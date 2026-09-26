"""Vercel Serverless Function entrypoint for IQWAVE FastAPI application."""

import os
import sys

# Ensure repository root is on sys.path so 'backend' can be imported anywhere
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

try:
    from backend.app.main import app
except Exception as e:
    import traceback
    err_str = traceback.format_exc()
    print("FATAL: Failed to initialize backend.app.main:\n", err_str, file=sys.stderr)
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse
    app = FastAPI(title="IQWAVE Error Handler")
    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD"])
    def fallback(path: str):
        return JSONResponse(
            status_code=500,
            content={
                "status": "INITIALIZATION_ERROR",
                "error": "FastAPI initialization failed on Vercel runtime.",
                "details": err_str
            }
        )

# Expose 'app' for ASGI/WSGI serverless runtimes
__all__ = ["app"]
