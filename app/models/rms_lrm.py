"""Pure-Python scorer for R rms::lrm models exported to model.json.

Reimplements prediction for a proportional-odds logistic regression with
3-knot restricted cubic splines and factor interactions, plus delta-method
confidence intervals. No R runtime required; validated against the authors'
hosted Shiny app (see tests/test_hgt_golden.py).
"""

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

Z_975 = 1.959964  # qnorm(0.975), matching rms


def rcs3_basis(x: float, knots: list[float]) -> float:
    """Second basis column of a 3-knot restricted cubic spline (Hmisc norm=2).

    The first column is x itself; this returns S(x), constructed so the
    spline is linear beyond the outer knots.
    """
    t1, t2, t3 = knots
    pos = lambda v: max(v, 0.0) ** 3
    return (
        pos(x - t1)
        - pos(x - t2) * (t3 - t1) / (t3 - t2)
        + pos(x - t3) * (t2 - t1) / (t3 - t2)
    ) / (t3 - t1) ** 2


def _expit(lp: float) -> float:
    return 1.0 / (1.0 + math.exp(-lp))


class RmsLrmScenarioModel:
    def __init__(self, model_json_path: Path):
        with open(model_json_path) as f:
            m = json.load(f)
        self.spec = m
        self.intercept_name = m["any_complication_intercept"]
        self.intercept = m["intercepts"][self.intercept_name]
        # All cumulative-probability intercepts, ordered: P(Y>=1), P(Y>=2), ...
        self.all_intercepts = list(m["intercepts"].items())
        self.ylevels = m["ylevels"]
        self.coef = m["coefficients"]  # ordered dict: column name -> beta
        self.cov_names: list[str] = m["covariance"]["names"]
        self.cov = np.array(m["covariance"]["matrix"])
        self.scenario = m["scenario_variable"]
        self.variables = m["variables"]
        # API-facing feature names (fixed inputs have api_name None)
        self.feature_names = [v["api_name"] for v in self.variables if v["api_name"]]

    def _main_effect_columns(
        self, features: dict[str, Any], scenario_level: str
    ) -> dict[str, float]:
        """Values of every main-effect design column, keyed by rms column name."""
        cols: dict[str, float] = {}

        # Scenario variable: dummies vs its reference level
        for level in self.scenario["r_levels"][1:]:
            cols[f"{self.scenario['r_name']}={level}"] = float(level == scenario_level)

        for v in self.variables:
            r_name = v["r_name"]
            if v["kind"] == "rcs3":
                if v.get("fixed_value") is not None:
                    x = float(v["fixed_value"])
                else:
                    x = float(features[v["api_name"]])
                transform = v.get("input_transform")
                if transform and transform["type"] == "excess_over":
                    x = max(x - transform["threshold"], 0.0)
                cols[r_name] = x
                cols[r_name + "'"] = rcs3_basis(x, v["knots"])
            else:  # factor: match API level by position -> R level dummy
                api_val = features[v["api_name"]]
                if api_val not in v["api_levels"]:
                    raise ValueError(
                        f"Invalid value {api_val!r} for {v['api_name']}; "
                        f"expected one of {v['api_levels']}"
                    )
                r_val = v["r_levels"][v["api_levels"].index(api_val)]
                for level in v["r_levels"][1:]:  # first level = reference
                    cols[f"{r_name}={level}"] = float(level == r_val)
        return cols

    def _design_vector(
        self, features: dict[str, Any], scenario_level: str
    ) -> np.ndarray:
        cols = self._main_effect_columns(features, scenario_level)
        x = []
        for name in self.coef:
            if " * " in name:
                left, right = name.split(" * ", 1)
                x.append(cols[left] * cols[right])
            else:
                x.append(cols[name])
        return np.array(x)

    def _score_level(
        self, x: np.ndarray, intercept_name: str, intercept: float
    ) -> dict:
        beta = np.array(list(self.coef.values()))
        lp = intercept + float(x @ beta)

        # Delta-method CI: z aligned by name to the covariance matrix
        by_name = dict(zip(self.coef.keys(), x))
        z = np.array(
            [
                1.0 if name == intercept_name
                else by_name.get(name, 0.0)  # other intercepts -> 0
                for name in self.cov_names
            ]
        )
        se = math.sqrt(float(z @ self.cov @ z))
        return {
            "probability": round(_expit(lp), 4),
            "ci_lower": round(_expit(lp - Z_975 * se), 4),
            "ci_upper": round(_expit(lp + Z_975 * se), 4),
        }

    def _score_scenario(self, features: dict[str, Any], scenario_level: str) -> dict:
        x = self._design_vector(features, scenario_level)
        # Primary result = P(Y >= first non-reference level), i.e. any event;
        # by_level carries every cumulative probability (matches the Shiny app)
        primary = self._score_level(x, self.intercept_name, self.intercept)
        by_level = [
            {"level": f"{self.ylevels[i + 1]}+", **self._score_level(x, name, value)}
            for i, (name, value) in enumerate(self.all_intercepts)
        ]
        return {**primary, "by_level": by_level}

    def predict(
        self, features: dict[str, Any], model_name: str, model_version: str
    ) -> dict:
        missing = [f for f in self.feature_names if f not in features]
        if missing:
            raise ValueError(f"Missing features: {missing}")

        scenarios = []
        for level, label in zip(self.scenario["r_levels"], self.scenario["labels"]):
            scenarios.append({"label": label, **self._score_scenario(features, level)})
        return {
            "model_name": model_name,
            "model_version": model_version,
            "scenarios": scenarios,
        }
