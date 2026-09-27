"""FastAPI Main Application Entry Point."""

import os
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from backend.app.api.routes_module1 import router as module1_router
from backend.app.api.routes_module2 import router as module2_router
from backend.app.api.routes_module3 import router as module3_router
from backend.app.api.routes_module4 import router as module4_router
from backend.app.api.routes_module5 import router as module5_router
from backend.app.api.routes_module6 import router as module6_router
from backend.app.api.routes_module7 import router as module7_router
from backend.app.api.routes_module8 import router as module8_router
from backend.app.api.routes_module9 import router as module9_router
from backend.app.api.routes_module10 import router as module10_router
from backend.app.api.routes_pipeline import router as pipeline_router

app = FastAPI(
    title="Blind Signal Analysis System (SIH26147)",
    description="Automated model for analysis of .IQ and .wav files along with signal parameter extraction.",
    version="1.0.0"
)

# CORS middleware for local frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def normalize_api_path(request, call_next):
    # When deployed on Vercel or behind reverse proxies, path may arrive stripped of '/api'
    # or passed through headers like 'x-matched-path'
    orig_path = request.headers.get("x-matched-path") or request.headers.get("x-vercel-matched-path")
    if orig_path and orig_path.startswith("/api/"):
        request.scope["path"] = orig_path
    else:
        path = request.scope.get("path", "")
        if not path.startswith("/api/") and (path.startswith("/pipeline") or path.startswith("/module")):
            request.scope["path"] = f"/api{path}"
    response = await call_next(request)
    return response

# Register API routes
app.include_router(module1_router)
app.include_router(module2_router)
app.include_router(module3_router)
app.include_router(module4_router)
app.include_router(module5_router)
app.include_router(module6_router)
app.include_router(module7_router)
app.include_router(module8_router)
app.include_router(module9_router)
app.include_router(module10_router)
app.include_router(pipeline_router)

@app.get("/api")
@app.get("/api/")
@app.get("/api/index.py")
def api_root():
    return {
        "status": "ONLINE",
        "service": "IQWAVE Blind Signal Analysis Backend",
        "version": "1.0.0",
        "endpoints": {
            "status": "/api/pipeline/status",
            "run": "/api/pipeline/run",
            "fixture": "/api/pipeline/fixture",
            "upload": "/api/pipeline/upload",
            "report": "/api/pipeline/report"
        }
    }



# Mount frontend static directory if exists
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "frontend")
if not os.path.exists(frontend_dir):
    candidate = os.path.join(os.getcwd(), "frontend")
    if os.path.exists(candidate):
        frontend_dir = candidate

if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

    @app.get("/")
    def serve_frontend_index():
        return FileResponse(os.path.join(frontend_dir, "index.html"))

@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)
