from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.logging.db import get_usage_stats
from app.models.registry import registry
from app.routers import models_info
from app.routers.predict import make_predict_handler
from app.schemas.predict import PredictionResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    registry.load_all()
    loaded = registry.list_models()
    print(f"Loaded {len(loaded)} model(s): {loaded}")

    # Register a dedicated prediction endpoint for every loaded model.
    # Each endpoint gets its own Swagger entry, display name, description,
    # and a pre-filled example request body derived from the model's metadata.
    existing_paths = {route.path for route in app.routes}  # type: ignore[attr-defined]
    for model_name in loaded:
        path = f"/predict/{model_name}"
        if path in existing_paths:
            continue
        metadata = registry.get_metadata(model_name)
        if metadata is None:
            continue
        display_name = metadata.get("display_name", model_name)
        request_model = registry.make_request_model(model_name)
        handler = make_predict_handler(model_name, request_model)
        app.add_api_route(
            path,
            handler,
            methods=["POST"],
            response_model=PredictionResponse,
            tags=[display_name],
            summary=f"Predict — {display_name}",
            description=metadata.get("description", ""),
        )

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
