from pydantic import BaseModel


class PredictionRequest(BaseModel):
    features: dict[str, float]

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "features": {
                        "Post-Index_Inferior_Attachments_Spine_Based": 1.0,
                        "Pre-Index_Height_cm": 110.0,
                        "Post-Index_Device_Type_VEPTR": 0.0,
                        "Post-Index_Device_Type_TGR": 1.0,
                        "Pre-Index_Kyphosis": 45.0,
                        "age_at_surgery": 6.5,
                        "Post-Index_Number_of_Superior_Anchors": 4.0,
                        "Gender_Male": 1.0,
                    }
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    threshold: float
    model_name: str
    model_version: str
