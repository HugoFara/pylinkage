"""FastAPI application entry point for pylinkage web API."""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .config import settings
from .routers import examples, linkages, simulation, websocket


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    yield
    # Shutdown


app = FastAPI(
    title="Pylinkage API",
    description="REST API for building and simulating planar linkages",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(linkages.router, prefix="/api")
app.include_router(simulation.router, prefix="/api")
app.include_router(examples.router, prefix="/api")
app.include_router(websocket.router, prefix="/api")


@app.get("/health")
def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/api")
def api_info() -> dict[str, str | list[str]]:
    """API information."""
    return {
        "name": "Pylinkage API",
        "version": "1.0.0",
        "endpoints": [
            "/api/linkages",
            "/api/linkages/{id}",
            "/api/linkages/{id}/simulate",
            "/api/linkages/{id}/trajectory",
            "/api/examples",
            "/api/examples/{name}",
            "/api/ws/simulation/{id}",
            "/api/ws/simulation-fast/{id}",
        ],
    }


# Serve frontend static files if the frontend directory exists
frontend_dir = Path(__file__).parent.parent / "frontend" / "dist"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
