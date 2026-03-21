import time

from fastapi import BackgroundTasks, HTTPException

from app.logging.db import log_request
from app.models.registry import registry
from app.schemas.predict import PredictionResponse


def make_predict_handler(model_name: str, request_model: type):
    """Return a FastAPI route handler bound to a specific model."""

    async def predict(request: request_model, background_tasks: BackgroundTasks):
        start = time.perf_counter()
        try:
            result = registry.predict(model_name, request.features)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        elapsed_ms = (time.perf_counter() - start) * 1000
        background_tasks.add_task(
            log_request, model_name, f"/predict/{model_name}", elapsed_ms, 200
        )
        return result

    # Give each handler a unique name so FastAPI generates distinct operation IDs
    predict.__name__ = f"predict_{model_name}"
    return predict
