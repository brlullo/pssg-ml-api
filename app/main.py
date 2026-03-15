from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.logging.db import get_usage_stats
from app.models.registry import registry
from app.routers import models_info, predict


@asynccontextmanager
async def lifespan(app: FastAPI):
    registry.load_all()
    loaded = registry.list_models()
    print(f"Loaded {len(loaded)} model(s): {loaded}")
    yield


app = FastAPI(
    title="PSSG ML API",
    description=(
        "Centralized API for machine learning models developed using the "
        "Pediatric Spine Study Group (PSSG) database. Provides standardized "
        "endpoints for clinical prediction and model documentation."
    ),
    version=settings.API_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict.router)
app.include_router(models_info.router)


@app.get("/", tags=["Info"])
def root():
    """API root with basic information and links."""
    return {
        "name": "PSSG ML API",
        "version": settings.API_VERSION,
        "description": "Machine learning models from the Pediatric Spine Study Group database",
        "docs_url": "/docs",
        "models_url": "/models",
    }


@app.get("/health", tags=["Info"])
def health():
    """Health check endpoint for deployment monitoring."""
    return {"status": "healthy", "models_loaded": len(registry.list_models())}


@app.get("/admin/usage", tags=["Admin"])
def usage():
    """API usage statistics: total queries, per-model counts, monthly trends."""
    return get_usage_stats()
