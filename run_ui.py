#!/usr/bin/env python3
"""Optional FastAPI UI for Legal Intelligence (port 8001)."""

from app.api.app import create_app

app = create_app()

if __name__ == "__main__":
    import uvicorn

    from app.config import API_HOST, API_PORT

    uvicorn.run("run_ui:app", host=API_HOST, port=API_PORT, reload=False)
