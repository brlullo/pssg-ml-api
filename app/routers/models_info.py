from fastapi import APIRouter, HTTPException

from app.models.registry import registry
from app.schemas.models_info import ModelDetail, ModelSummary

router = APIRouter(tags=["Models"])


@router.get("/models", response_model=list[ModelSummary])
def list_models():
    """List all available ML models with summary information."""
    return registry.get_all_metadata()


@router.get("/models/{model_name}", response_model=ModelDetail)
def get_model_detail(model_name: str):
    """Get detailed information about a specific model, including features,
    validation metrics, cohort details, and limitations."""
    metadata = registry.get_metadata(model_name)
    if metadata is None:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' not found. Available models: {registry.list_models()}",
        )
    return {"name": model_name, **metadata}
