"""Vercel Serverless Function entrypoint for IQWAVE FastAPI application."""

import os
import sys

# Ensure repository root is on sys.path so 'backend' can be imported anywhere
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from backend.app.main import app

# Expose 'app' for ASGI/WSGI serverless runtimes
__all__ = ["app"]
