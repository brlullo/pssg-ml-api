from fastapi import APIRouter, HTTPException

from app.models.registry import registry
from app.schemas.predict import PredictionRequest, PredictionResponse

router = APIRouter(tags=["Predictions"])


@router.post("/predict/{model_name}", response_model=PredictionResponse)
def predict(model_name: str, request: PredictionRequest):
    """Run inference on a loaded ML model.

    Provide feature values as a JSON object. Use GET /models/{model_name}
    to see required features and their descriptions.
    """
    if model_name not in registry.list_models():
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' not found. Available models: {registry.list_models()}",
        )

    try:
        result = registry.predict(model_name, request.features)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    return result
