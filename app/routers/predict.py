import time

from fastapi import APIRouter, BackgroundTasks, HTTPException

from app.logging.db import log_request
from app.models.registry import registry
from app.schemas.predict import PredictionRequest, PredictionResponse

router = APIRouter(tags=["Predictions"])


@router.post("/predict/{model_name}", response_model=PredictionResponse)
def predict(
    model_name: str,
    request: PredictionRequest,
    background_tasks: BackgroundTasks,
):
    """Run inference on a loaded ML model.

    Provide feature values as a JSON object. Use GET /models/{model_name}
    to see required features and their descriptions.
    """
    start = time.perf_counter()

    if model_name not in registry.list_models():
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' not found. Available models: {registry.list_models()}",
        )

    try:
        result = registry.predict(model_name, request.features)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    elapsed_ms = (time.perf_counter() - start) * 1000
    background_tasks.add_task(
        log_request, model_name, f"/predict/{model_name}", elapsed_ms, 200
    )

    return result
