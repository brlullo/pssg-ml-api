"""Golden tests: the Python port must reproduce the authors' hosted Shiny app.

Fixture values were captured from https://vumcbiostat.shinyapps.io/hgtmodel/
(the authoritative rms implementation). The app displays 3 decimals, so the
tolerance is the display half-width (0.0005) plus a small numerical margin.
"""

import json
from pathlib import Path

import pytest

from app.models.rms_lrm import RmsLrmScenarioModel

FIXTURE = Path(__file__).parent / "fixtures" / "hgt_golden.json"
MODEL_JSON = (
    Path(__file__).parent.parent
    / "app" / "models" / "artifacts" / "hgt_complication" / "model.json"
)

TOL_P = 0.0015
TOL_CI = 0.0015

with open(FIXTURE) as f:
    CASES = json.load(f)["cases"]


@pytest.fixture(scope="module")
def model():
    return RmsLrmScenarioModel(MODEL_JSON)


# Shiny app's default case, rows "2+ UPRORs" and "3+ UPRORs" (captured 2026-08-10):
# columns are (without HGT), (with HGT); values as displayed with 95% CIs
HIGHER_LEVELS_CASE = CASES[0]["inputs"]
HIGHER_LEVELS_EXPECTED = {
    "2+": {"without": (0.018, 0.006, 0.050), "with": (0.011, 0.003, 0.041)},
    "3+": {"without": (0.004, 0.001, 0.013), "with": (0.003, 0.001, 0.011)},
}


def test_higher_severity_levels_match_shiny(model):
    out = model.predict(HIGHER_LEVELS_CASE, "hgt_complication", "1.0.0")
    by_label = {s["label"]: s for s in out["scenarios"]}
    for level, cols in HIGHER_LEVELS_EXPECTED.items():
        for label, key in [("Without HGT", "without"), ("With HGT", "with")]:
            got = next(l for l in by_label[label]["by_level"] if l["level"] == level)
            p, lo, hi = cols[key]
            assert got["probability"] == pytest.approx(p, abs=TOL_P)
            assert got["ci_lower"] == pytest.approx(lo, abs=TOL_CI)
            assert got["ci_upper"] == pytest.approx(hi, abs=TOL_CI)


def test_by_level_first_row_equals_primary(model):
    out = model.predict(CASES[0]["inputs"], "hgt_complication", "1.0.0")
    for s in out["scenarios"]:
        assert s["by_level"][0]["probability"] == s["probability"]
        assert len(s["by_level"]) == 3


@pytest.mark.parametrize("case", CASES, ids=lambda c: (
    f"cor{c['inputs']['coronal_deformity']}_kyp{c['inputs']['max_kyphosis']}"
))
def test_matches_shiny_app(model, case):
    out = model.predict(case["inputs"], "hgt_complication", "1.0.0")
    by_label = {s["label"]: s for s in out["scenarios"]}

    for label, key in [("Without HGT", "without"), ("With HGT", "with")]:
        expected = case.get(key)
        if not expected:
            continue
        got = by_label[label]
        assert got["probability"] == pytest.approx(expected["p"], abs=TOL_P), (
            f"{label} probability mismatch vs Shiny app"
        )
        if "lo" in expected:
            assert got["ci_lower"] == pytest.approx(expected["lo"], abs=TOL_CI)
            assert got["ci_upper"] == pytest.approx(expected["hi"], abs=TOL_CI)
