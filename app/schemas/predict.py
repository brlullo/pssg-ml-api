from pydantic import BaseModel


class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    threshold: float
    model_name: str
    model_version: str


class Scenario(BaseModel):
    label: str
    probability: float
    ci_lower: float
    ci_upper: float


class ScenarioPredictionResponse(BaseModel):
    """Response for models that compare treatment scenarios (prediction_type: scenario_comparison)."""

    model_name: str
    model_version: str
    scenarios: list[Scenario]
