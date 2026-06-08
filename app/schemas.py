"""Pydantic schemas for API and validation."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


class PartyModel(BaseModel):
    plaintiff: str = Field(..., min_length=1, max_length=500)
    defendant: str = Field(..., min_length=1, max_length=500)


class CaseInputModel(BaseModel):
    title: str = Field(..., min_length=3, max_length=1000)
    jurisdiction: str = Field(..., min_length=2, max_length=200)
    court_type: str = Field(..., min_length=2, max_length=200)
    parties: PartyModel
    claims: list[str] = Field(..., min_length=1, max_length=20)
    relief_sought: list[str] = Field(default_factory=list, max_length=20)
    key_facts: list[str] = Field(default_factory=list, max_length=50)
    legal_issues: list[str] = Field(default_factory=list, max_length=20)
    contract_clauses: list[Any] = Field(default_factory=list, max_length=30)
    venue: str = Field(default="", max_length=200)
    your_role: str = Field(default="plaintiff", pattern=r"^(plaintiff|defendant)$")
    procedural_posture: str = Field(default="pre-filing", max_length=100)
    case_id: str | None = Field(default=None, max_length=100)

    @field_validator("claims", "key_facts", "legal_issues", "relief_sought")
    @classmethod
    def strip_items(cls, v: list[str]) -> list[str]:
        return [item.strip() for item in v if item and item.strip()]


class AnalyzeRequest(BaseModel):
    case: CaseInputModel
    thread_id: str | None = Field(
        default=None,
        description="Optional thread ID for checkpointed multi-turn sessions.",
        max_length=128,
    )
    run_id: str | None = Field(
        default=None,
        description="Caller-supplied run identifier (echoed back). Auto-generated if absent.",
        max_length=128,
    )
    skip_log: bool = False


class HealthResponse(BaseModel):
    status: str
    version: str
    environment: str


class ReadyResponse(BaseModel):
    ready: bool
    case_index_count: int
    courtlistener_enabled: bool
    llm_enabled: bool
    llm_model: str | None = None
    llm_base_url: str | None = None
    warnings: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Canonical run envelope (success / error)
# ---------------------------------------------------------------------------


class AgentInfo(BaseModel):
    name: str
    role: str


class OutputBlock(BaseModel):
    summary: str
    recommendations: list[str] = Field(default_factory=list)
    artifacts: list[dict[str, Any]] = Field(default_factory=list)


class ErrorBlock(BaseModel):
    type: str
    message: str
    recoverable: bool = False


class RunSuccessResponse(BaseModel):
    run_id: str
    status: str = Field(default="success", pattern="^success$")
    output: OutputBlock
    agents: list[AgentInfo] = Field(default_factory=list)
    trace_id: str | None = None
    log_file: str | None = None
    execution_time_seconds: float


class RunErrorResponse(BaseModel):
    run_id: str
    status: str = Field(default="error", pattern="^error$")
    error: ErrorBlock
    trace_id: str | None = None
    log_file: str | None = None
    execution_time_seconds: float


# Convenience union for OpenAPI docs.
RunResponse = RunSuccessResponse | RunErrorResponse
