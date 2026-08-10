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
    # Optional display config for the web UI: field grouping/order, composite
    # controls (e.g. one dropdown driving multiple one-hot features), result note
    ui: dict | None = None
    metrics: dict
    cohort: dict
    last_updated: str
    limitations: str
