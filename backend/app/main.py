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
from backend.app.api.routes_pipeline import router as pipeline_router, experimental_router

app = FastAPI(
    title="DAWC — Digital Automated Waveform Classifier (SIH26147)",
    description="Automated model for analysis of .IQ and .wav files along with signal parameter extraction, digital waveform classification, and blind protocol recovery.",
    version="1.0.0",
    redirect_slashes=False
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
    # When deployed on Vercel or behind reverse proxies, inspect routing headers first
    headers = dict(request.scope.get("headers", []))
    matched_path = None
    for h_name, h_val in headers.items():
        if h_name.lower() in [b"x-matched-path", b"x-forwarded-uri", b"x-real-origin-url", b"x-vercel-sc-path"]:
            matched_path = h_val.decode("utf-8", errors="ignore")
            break

    path = matched_path or request.scope.get("path", "")
    if "?" in path:
        path = path.split("?")[0]

    # Strip any serverless file prefix (/api/index.py, /index.py, /api/index, /index)
    for prefix in ["/api/index.py", "/index.py", "/api/index", "/index"]:
        if path.startswith(prefix):
            path = path[len(prefix):]
            break

    if not path.startswith("/"):
        path = "/" + path

    # If stripped path is empty or root
    if path in ["", "/"]:
        method = request.scope.get("method", "GET").upper()
        if method == "POST":
            path = "/api/experimental/pipeline/run"
        else:
            path = "/api"

    # Ensure /api prefix if routed directly to submodule or experimental
    if not path.startswith("/api/") and (path.startswith("/pipeline") or path.startswith("/module") or path.startswith("/experimental")):
        path = f"/api{path}"

    request.scope["path"] = path
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
from backend.app.api.routes_pipeline import (
    router as pipeline_router,
    experimental_router,
    post_run_pipeline,
    PipelineRunRequest,
    get_integration_fixture
)
app.include_router(pipeline_router)
app.include_router(experimental_router)

@app.get("/api")
@app.get("/api/")
@app.get("/api/index.py")
def api_root():
    return {
        "status": "ONLINE",
        "service": "DAWC — Digital Automated Waveform Classifier Backend",
        "version": "1.0.0",
        "endpoints": {
            "status": "/api/pipeline/status",
            "run": "/api/pipeline/run",
            "fixture": "/api/pipeline/fixture",
            "upload": "/api/pipeline/upload",
            "report": "/api/pipeline/report",
            "experimental_run": "/api/experimental/pipeline/run",
            "experimental_fixture": "/api/experimental/pipeline/fixture",
            "experimental_status": "/api/experimental/pipeline/status"
        }
    }

@app.post("/api")
@app.post("/api/")
@app.post("/api/index.py")
@app.post("/index.py")
def api_root_post(req: PipelineRunRequest):
    return post_run_pipeline(req)

@app.get("/api/fixture")
@app.get("/fixture")
def api_direct_fixture(preset: str = "qpsk_1200"):
    return get_integration_fixture(preset=preset)




# Mount frontend static directory if exists
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "frontend")
if not os.path.exists(frontend_dir):
    candidate = os.path.join(os.getcwd(), "frontend")
    if os.path.exists(candidate):
        frontend_dir = candidate

if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

    @app.get("/")
    @app.get("/exp")
    @app.get("/experimental")
    @app.get("/radar")
    @app.get("/radar-labview")
    @app.get("/Frontend-Experimental")
    def serve_frontend_index():
        return FileResponse(os.path.join(frontend_dir, "index.html"))

    @app.get("/app.js")
    def serve_frontend_js():
        return FileResponse(os.path.join(frontend_dir, "app.js"), media_type="application/javascript")

    @app.get("/style.css")
    def serve_frontend_css():
        return FileResponse(os.path.join(frontend_dir, "style.css"), media_type="text/css")

# Also mount Frontend-Experimental / Radar LabView static directory for alias compatibility
candidates = [
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "Frontend-Experimental"),
    os.path.join(os.getcwd(), "Frontend-Experimental"),
    os.path.join(os.getcwd(), "public", "experimental"),
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "public", "experimental"),
]
experimental_dir = next((c for c in candidates if os.path.exists(c)), None)

if experimental_dir:
    app.mount("/experimental", StaticFiles(directory=experimental_dir, html=True), name="experimental")

@app.get("/favicon.ico")
def favicon():
    ico_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "favicon.ico")
    if os.path.exists(ico_path):
        return FileResponse(ico_path, media_type="image/x-icon")
    candidate = os.path.join(os.getcwd(), "favicon.ico")
    if os.path.exists(candidate):
        return FileResponse(candidate, media_type="image/x-icon")
    return Response(status_code=204)

@app.get("/favicon.svg")
def favicon_svg():
    svg_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "favicon.svg")
    if os.path.exists(svg_path):
        return FileResponse(svg_path, media_type="image/svg+xml")
    candidate = os.path.join(os.getcwd(), "favicon.svg")
    if os.path.exists(candidate):
        return FileResponse(candidate, media_type="image/svg+xml")
    return Response(status_code=204)

