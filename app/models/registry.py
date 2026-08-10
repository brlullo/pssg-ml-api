import json
import pickle
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
from pydantic import Field, create_model

from app.models.rms_lrm import RmsLrmScenarioModel


ARTIFACTS_DIR = Path(__file__).parent / "artifacts"


class ModelRegistry:
    """Discovers and loads ML models from the artifacts directory."""

    def __init__(self) -> None:
        self._models: dict[str, dict[str, Any]] = {}

    def load_all(self) -> None:
        if not ARTIFACTS_DIR.exists():
            return
        for model_dir in sorted(ARTIFACTS_DIR.iterdir()):
            if not model_dir.is_dir():
                continue
            self._load_model(model_dir)

    def _load_model(self, model_dir: Path) -> None:
        model_name = model_dir.name
        metadata_path = model_dir / "metadata.json"
        if not metadata_path.exists():
            return

        with open(metadata_path) as f:
            metadata = json.load(f)

        # JSON-exported R model (rms::lrm) — scored in pure Python
        json_file = model_dir / "model.json"
        if json_file.exists():
            predictor = RmsLrmScenarioModel(json_file)
            self._models[model_name] = {
                "backend": "rms_lrm",
                "predictor": predictor,
                "feature_names": predictor.feature_names,
                "metadata": metadata,
            }
            return

        # Load the model file (.pkl or .joblib)
        model_file = model_dir / "model.pkl"
        if not model_file.exists():
            model_file = model_dir / "model.joblib"
        if not model_file.exists():
            return

        if model_file.suffix == ".joblib":
            artifact = joblib.load(model_file)
        else:
            with open(model_file, "rb") as f:
                artifact = pickle.load(f)

        # Handle dict-style artifacts (like the UPROR model)
        if isinstance(artifact, dict):
            model = artifact.get("calibrated_model") or artifact.get("model")
            scaler = artifact.get("scaler")
            feature_names = artifact.get("feature_names", [])
        else:
            model = artifact
            scaler = None
            feature_names = (
                list(artifact.feature_names_in_)
                if hasattr(artifact, "feature_names_in_")
                else []
            )

        self._models[model_name] = {
            "backend": "sklearn",
            "model": model,
            "scaler": scaler,
            "feature_names": feature_names,
            "metadata": metadata,
            "artifact": artifact,
        }

    def make_request_model(self, model_name: str) -> type:
        """Create a model-specific Pydantic request class with individual feature fields."""
        entry = self._models[model_name]
        metadata = entry["metadata"]
        feature_names = entry["feature_names"]

        feature_meta = {f["name"]: f for f in metadata.get("features", [])}
        fields: dict[str, tuple] = {}
        for name in feature_names:
            meta = feature_meta.get(name, {})
            desc = meta.get("description", "")
            if meta.get("type") == "categorical":
                options = meta["options"]
                fields[name] = (
                    Literal[tuple(options)],
                    Field(description=desc, examples=[options[0]]),
                )
                continue
            if "example_value" in meta:
                example = meta["example_value"]
            elif meta.get("type") == "binary":
                example = 1.0
            else:
                example = 0.0
            fields[name] = (float, Field(description=desc, examples=[example]))

        class_name = (
            "".join(w.capitalize() for w in model_name.split("_"))
            + "PredictionRequest"
        )
        return create_model(class_name, **fields)

    def list_models(self) -> list[str]:
        return list(self._models.keys())

    def get_metadata(self, model_name: str) -> dict | None:
        entry = self._models.get(model_name)
        return entry["metadata"] if entry else None

    def get_all_metadata(self) -> list[dict]:
        return [
            {"name": name, **entry["metadata"]}
            for name, entry in self._models.items()
        ]

    def predict(self, model_name: str, features: dict[str, Any]) -> dict:
        entry = self._models.get(model_name)
        if entry is None:
            raise KeyError(f"Model '{model_name}' not found")

        if entry["backend"] == "rms_lrm":
            return entry["predictor"].predict(
                features,
                model_name,
                entry["metadata"].get("version", "unknown"),
            )

        feature_names = entry["feature_names"]
        missing = [f for f in feature_names if f not in features]
        if missing:
            raise ValueError(f"Missing features: {missing}")

        # Build DataFrame with correct feature names to avoid sklearn warnings
        X = pd.DataFrame([[features[f] for f in feature_names]], columns=feature_names)

        # Apply scaler if present
        scaler = entry["scaler"]
        if scaler is not None:
            X = pd.DataFrame(scaler.transform(X), columns=feature_names)

        model = entry["model"]
        probability = float(model.predict_proba(X)[0, 1])
        threshold = entry["metadata"].get("metrics", {}).get(
            "optimal_threshold", 0.5
        )
        prediction = int(probability >= threshold)

        return {
            "prediction": prediction,
            "probability": round(probability, 4),
            "threshold": threshold,
            "model_name": model_name,
            "model_version": entry["metadata"].get("version", "unknown"),
        }


# Singleton instance
registry = ModelRegistry()
