from pydantic import BaseModel


class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    threshold: float
    model_name: str
    model_version: str
