"""FastAPI application factory."""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

from app.config import APP_ENV, APP_VERSION, CORS_ORIGINS, LEGAL_DISCLAIMER, OPENAI_BASE_URL, OPENAI_MODEL
from app.exceptions import AnalysisError, ConfigurationError, LLMError
from app.llm import llm_available
from app.logging_config import setup_logging
from app.schemas import AnalyzeRequest, AnalyzeResponse, HealthResponse, ReadyResponse
from app.service import initialize, readiness, run_case_analysis


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    stats = initialize()
    app.state.startup_stats = stats
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="Legal Intelligence API",
        description="Production multi-agent LangGraph system with critique loops.",
        version=APP_VERSION,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", uuid.uuid4().hex)
        request.state.request_id = request_id
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time-Ms"] = str(elapsed_ms)
        return response

    @app.exception_handler(AnalysisError)
    async def analysis_error_handler(_: Request, exc: AnalysisError):
        return JSONResponse(status_code=500, content={"detail": str(exc), "type": "AnalysisError"})

    @app.exception_handler(ConfigurationError)
    async def config_error_handler(_: Request, exc: ConfigurationError):
        return JSONResponse(status_code=503, content={"detail": str(exc), "type": "ConfigurationError"})

    @app.exception_handler(LLMError)
    async def llm_error_handler(_: Request, exc: LLMError):
        return JSONResponse(status_code=503, content={"detail": str(exc), "type": "LLMError"})

    @app.get("/health", response_model=HealthResponse)
    @app.get("/v1/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok", version=APP_VERSION, environment=APP_ENV)

    @app.get("/ready", response_model=ReadyResponse)
    @app.get("/v1/ready", response_model=ReadyResponse)
    def ready() -> ReadyResponse:
        try:
            stats = readiness()
        except (ConfigurationError, LLMError) as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return ReadyResponse(
            ready=stats["ready"],
            case_index_count=stats["case_index_count"],
            courtlistener_enabled=stats["courtlistener_enabled"],
            llm_enabled=stats.get("llm_ready", False),
            llm_model=OPENAI_MODEL if llm_available() else None,
            llm_base_url=OPENAI_BASE_URL if llm_available() else None,
            warnings=stats.get("warnings") or [],
        )

    @app.post("/analyze", response_model=AnalyzeResponse)
    @app.post("/v1/analyze", response_model=AnalyzeResponse)
    def analyze(body: AnalyzeRequest) -> AnalyzeResponse:
        try:
            result = run_case_analysis(
                body.case.model_dump(exclude_none=True),
                thread_id=body.thread_id,
                log=not body.skip_log,
            )
        except (ConfigurationError, LLMError) as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return AnalyzeResponse(**result)

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def home() -> str:
        from pathlib import Path

        examples = sorted(Path("input_examples").glob("*.json"))
        links = "".join(
            f'<li><a href="#" onclick="loadExample(\'{p.name}\')">{p.name}</a></li>'
            for p in examples
        )
        return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Legal Intelligence</title>
<style>
body {{ font-family: system-ui, sans-serif; max-width: 960px; margin: 2rem auto; padding: 0 1rem; }}
textarea {{ width: 100%; height: 240px; font-family: monospace; }}
pre {{ background: #f4f4f4; padding: 1rem; overflow: auto; max-height: 520px; font-size: 12px; }}
.warn {{ background: #fff3cd; padding: 1rem; border-left: 4px solid #ffc107; margin-bottom: 1rem; }}
.badge {{ display: inline-block; background: #198754; color: #fff; padding: 2px 8px; border-radius: 4px; font-size: 12px; }}
</style></head><body>
<h1>Legal Intelligence <span class="badge">v{APP_VERSION} + critique</span></h1>
<div class="warn"><strong>Disclaimer:</strong> {LEGAL_DISCLAIMER}</div>
<p>Multi-agent pipeline with critic peer review and supervisor revision loops.</p>
<p>Sample inputs: <ul>{links}</ul></p>
<textarea id="input"></textarea><br>
<button onclick="run()">Analyze Case</button>
<pre id="out">Results appear here.</pre>
<script>
async function loadExample(name) {{
  const r = await fetch('/examples/' + name);
  document.getElementById('input').value = JSON.stringify(await r.json(), null, 2);
}}
async function run() {{
  const out = document.getElementById('out');
  out.textContent = 'Running multi-agent pipeline with critique loop...';
  try {{
    const caseData = JSON.parse(document.getElementById('input').value);
    const res = await fetch('/v1/analyze', {{
      method: 'POST',
      headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{ case: caseData }})
    }});
    out.textContent = JSON.stringify(await res.json(), null, 2);
  }} catch (e) {{ out.textContent = 'Error: ' + e; }}
}}
</script></body></html>"""

    @app.get("/examples/{name}")
    def get_example(name: str) -> dict[str, Any]:
        from pathlib import Path
        import json

        if ".." in name or "/" in name or "\\" in name:
            raise HTTPException(status_code=400, detail="Invalid example name")
        path = Path("input_examples") / name
        if not path.exists():
            raise HTTPException(status_code=404, detail="Example not found")
        return json.loads(path.read_text(encoding="utf-8"))

    return app
