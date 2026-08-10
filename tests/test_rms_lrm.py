import math
from pathlib import Path

import numpy as np
import pytest

from app.models.rms_lrm import RmsLrmScenarioModel, rcs3_basis

MODEL_JSON = (
    Path(__file__).parent.parent
    / "app" / "models" / "artifacts" / "hgt_complication" / "model.json"
)

KNOTS = [46.0, 73.0, 102.0]

VALID_FEATURES = {
    "coronal_deformity": 73.0,
    "max_kyphosis": 52.0,
    "age_years": 6.5,
    "gender": "Male",
    "race": "White",
    "weight_kg": 18.3,
    "etiology": "Idiopathic",
    "ambulatory_status": "Ambulatory",
    "pelvic_attachments": "No",
    "rod_number": "Bilateral",
    "device_type": "VEPTR/TGR",
}


@pytest.fixture(scope="module")
def model():
    return RmsLrmScenarioModel(MODEL_JSON)


# --- RCS basis math ---

def test_rcs3_zero_before_first_knot():
    for x in [-10.0, 0.0, 45.99, 46.0]:
        assert rcs3_basis(x, KNOTS) == 0.0


def test_rcs3_linear_beyond_last_knot():
    # Second finite difference must vanish where the spline is linear.
    # This catches the swapped-knot formula, which stays quadratic past t3.
    h = 1.0
    for x in [110.0, 150.0, 300.0]:
        second_diff = (
            rcs3_basis(x + h, KNOTS)
            - 2 * rcs3_basis(x, KNOTS)
            + rcs3_basis(x - h, KNOTS)
        )
        assert abs(second_diff) < 1e-9, f"not linear at x={x}"


def test_rcs3_continuous_at_knots():
    eps = 1e-7
    for t in KNOTS:
        left = rcs3_basis(t - eps, KNOTS)
        right = rcs3_basis(t + eps, KNOTS)
        assert abs(left - right) < 1e-5


# --- design vector construction ---

def test_reference_levels_zero_dummies(model):
    cols = model._main_effect_columns(VALID_FEATURES, "w/o HGT")
    # all-reference categorical input -> every factor dummy is 0 except Rod.number
    assert cols["Gender=Female"] == 0.0
    assert cols["Race=Black"] == 0.0 and cols["Race=Other"] == 0.0
    assert cols["Etiology=Idiopathic"] == 1.0  # non-reference level selected
    assert cols["Rod.number=Bilateral"] == 1.0
    assert cols["Treatment=w/ HGT"] == 0.0


def test_corrected_level_spelling_maps_to_r_level(model):
    # API accepts "Ambulatory"; R reference level is the typo "Abmulatory".
    cols = model._main_effect_columns(VALID_FEATURES, "w/o HGT")
    assert cols["Ambulatory.status=Non-Ambulatory"] == 0.0
    assert cols["Ambulatory.status=Therapeutic"] == 0.0


def test_without_hgt_zeroes_treatment_and_interactions(model):
    x = model._design_vector(VALID_FEATURES, "w/o HGT")
    names = list(model.coef.keys())
    for i, name in enumerate(names):
        if "Treatment=" in name:
            assert x[i] == 0.0, name


def test_interaction_is_product(model):
    x = model._design_vector(VALID_FEATURES, "w/ HGT")
    names = list(model.coef.keys())
    idx = {n: i for i, n in enumerate(names)}
    assert x[idx["Treatment=w/ HGT * Coronal.deformity"]] == pytest.approx(
        x[idx["Treatment=w/ HGT"]] * x[idx["Coronal.deformity"]]
    )
    assert x[idx["Treatment=w/ HGT * Sagittal.deformity'"]] == pytest.approx(
        x[idx["Treatment=w/ HGT"]] * x[idx["Sagittal.deformity'"]]
    )


def test_fixed_days_from_surgery_used(model):
    # Prediction at time of surgery (days=0), matching the authors' Shiny app
    cols = model._main_effect_columns(VALID_FEATURES, "w/o HGT")
    assert cols["days.from.surgery"] == 0.0


def test_kyphosis_excess_transform(model):
    # The fitted variable is kyphosis in excess of 45 degrees
    cols = model._main_effect_columns(dict(VALID_FEATURES, max_kyphosis=52.0), "w/o HGT")
    assert cols["Sagittal.deformity"] == 7.0
    cols = model._main_effect_columns(dict(VALID_FEATURES, max_kyphosis=40.0), "w/o HGT")
    assert cols["Sagittal.deformity"] == 0.0


# --- prediction output ---

def test_predict_shape_and_bounds(model):
    out = model.predict(VALID_FEATURES, "hgt_complication", "1.0.0")
    assert [s["label"] for s in out["scenarios"]] == ["Without HGT", "With HGT"]
    for s in out["scenarios"]:
        assert 0.0 < s["probability"] < 1.0
        assert s["ci_lower"] <= s["probability"] <= s["ci_upper"]
        assert 0.0 < s["ci_lower"] and s["ci_upper"] < 1.0


def test_predict_missing_feature_raises(model):
    incomplete = {k: v for k, v in VALID_FEATURES.items() if k != "age_years"}
    with pytest.raises(ValueError, match="Missing features"):
        model.predict(incomplete, "hgt_complication", "1.0.0")


def test_predict_invalid_categorical_raises(model):
    bad = dict(VALID_FEATURES, gender="Malee")
    with pytest.raises(ValueError, match="Invalid value"):
        model.predict(bad, "hgt_complication", "1.0.0")


def test_ci_matches_hand_computed_delta_method(model):
    # Independent recomputation of the SE for one scenario
    x = model._design_vector(VALID_FEATURES, "w/o HGT")
    by_name = dict(zip(model.coef.keys(), x))
    z = np.array([
        1.0 if n == model.intercept_name else by_name.get(n, 0.0)
        for n in model.cov_names
    ])
    se = math.sqrt(float(z @ model.cov @ z))
    lp = model.intercept + float(x @ np.array(list(model.coef.values())))

    out = model.predict(VALID_FEATURES, "m", "v")["scenarios"][0]
    expit = lambda v: 1 / (1 + math.exp(-v))
    assert out["ci_lower"] == pytest.approx(expit(lp - 1.959964 * se), abs=1e-4)
    assert out["ci_upper"] == pytest.approx(expit(lp + 1.959964 * se), abs=1e-4)
