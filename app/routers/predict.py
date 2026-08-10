from fastapi import HTTPException

from app.models.registry import registry


def make_predict_handler(model_name: str, request_model: type):
    """Return a FastAPI route handler bound to a specific model.

    Request logging happens in the log_predict_requests middleware, which
    sees the real status code — handlers stay pure inference.
    """

    async def predict(request: request_model):
        try:
            return registry.predict(model_name, request.model_dump())
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))

    # Give each handler a unique name so FastAPI generates distinct operation IDs
    predict.__name__ = model_name
    return predict
