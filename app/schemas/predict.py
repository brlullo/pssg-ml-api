from pydantic import BaseModel


class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    threshold: float
    model_name: str
    model_version: str


class LevelPrediction(BaseModel):
    """Cumulative probability for one ordinal outcome level, e.g. '1+' events."""

    level: str
    probability: float
    ci_lower: float
    ci_upper: float


class Scenario(BaseModel):
    label: str
    # Primary result: probability of any event (lowest cumulative level)
    probability: float
    ci_lower: float
    ci_upper: float
    # Full cumulative breakdown for ordinal models (1+, 2+, 3+ events)
    by_level: list[LevelPrediction] | None = None


class ScenarioPredictionResponse(BaseModel):
    """Response for models that compare treatment scenarios (prediction_type: scenario_comparison)."""

    model_name: str
    model_version: str
    scenarios: list[Scenario]
