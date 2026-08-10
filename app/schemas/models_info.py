from pydantic import BaseModel


class ModelSummary(BaseModel):
    name: str
    display_name: str
    description: str
    version: str
    algorithm: str


class ModelDetail(BaseModel):
    name: str
    display_name: str
    description: str
    version: str
    algorithm: str
    framework: str
    # "binary_threshold" (single probability vs cutoff) or
    # "scenario_comparison" (per-scenario probabilities with CIs)
    prediction_type: str = "binary_threshold"
    features: list[dict]
    metrics: dict
    cohort: dict
    last_updated: str
    limitations: str
