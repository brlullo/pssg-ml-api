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
    features: list[dict]
    metrics: dict
    cohort: dict
    last_updated: str
    limitations: str
