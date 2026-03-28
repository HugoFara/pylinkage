"""FastAPI application entry point for pylinkage web API."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .config import settings
from .routers import examples, mechanisms, optimization, simulation, synthesis, websocket


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    yield
    # Shutdown


app = FastAPI(
    title="Pylinkage API",
    description="REST API for building and simulating planar linkages",
    version="2.0.0",
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
app.include_router(mechanisms.router, prefix="/api")
app.include_router(simulation.router, prefix="/api")
app.include_router(examples.router, prefix="/api")
app.include_router(synthesis.router, prefix="/api")
app.include_router(optimization.router, prefix="/api")
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
        "version": "2.0.0",
        "endpoints": [
            "/api/mechanisms",
            "/api/mechanisms/{id}",
            "/api/mechanisms/{id}/simulate",
            "/api/mechanisms/{id}/trajectory",
            "/api/examples",
            "/api/examples/{name}",
            "/api/optimization",
            "/api/ws/simulation/{id}",
            "/api/ws/simulation-fast/{id}",
        ],
    }
