import json
import pickle
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from pydantic import ConfigDict, create_model


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
            "model": model,
            "scaler": scaler,
            "feature_names": feature_names,
            "metadata": metadata,
            "artifact": artifact,
        }

    def make_request_model(self, model_name: str) -> type:
        """Create a model-specific Pydantic request class with a pre-filled example."""
        entry = self._models[model_name]
        metadata = entry["metadata"]
        feature_names = entry["feature_names"]

        example_features: dict[str, float] = {}
        for f in metadata.get("features", []):
            fname = f["name"]
            example_features[fname] = 1.0 if f.get("type") == "binary" else 0.0
        if not example_features:
            example_features = {name: 0.0 for name in feature_names}

        class_name = (
            "".join(w.capitalize() for w in model_name.split("_"))
            + "PredictionRequest"
        )
        return create_model(
            class_name,
            __config__=ConfigDict(
                json_schema_extra={"examples": [{"features": example_features}]}
            ),
            features=(dict[str, float], ...),
        )

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

    def predict(self, model_name: str, features: dict[str, float]) -> dict:
        entry = self._models.get(model_name)
        if entry is None:
            raise KeyError(f"Model '{model_name}' not found")

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
