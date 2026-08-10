#!/usr/bin/env python
"""Assemble model.json for the HGT complication model from extract_hgt_model.R output.

Usage:
    python scripts/build_model_json.py <extract_dir> <output_model.json>

The api_name mapping and level-spelling corrections are defined here; the
scorer matches R levels by position, so the verbatim R levels (including the
source data's "Abmulatory" typo) are preserved in r_levels.
"""

import csv
import json
import sys
from pathlib import Path

# R predictor name -> API field spec. days.from.surgery and Treatment are
# handled separately (fixed input / scenario variable).
API_VARS = {
    "Coronal.deformity": {"api_name": "coronal_deformity"},
    # The fitted variable is kyphosis IN EXCESS of 45 degrees; the API accepts
    # raw max kyphosis (like the authors' Shiny app) and transforms internally.
    # Verified empirically against the Shiny app (exact 3-decimal agreement).
    "Sagittal.deformity": {
        "api_name": "max_kyphosis",
        "input_transform": {"type": "excess_over", "threshold": 45.0},
    },
    "Age": {"api_name": "age_years"},
    "Gender": {"api_name": "gender"},
    "Race": {"api_name": "race"},
    "Weight": {"api_name": "weight_kg"},
    "Etiology": {"api_name": "etiology"},
    "Ambulatory.status": {
        "api_name": "ambulatory_status",
        # index-aligned corrections of R level spellings for API display
        "level_fixes": {"Abmulatory": "Ambulatory"},
    },
    "Pelvic.attachments": {"api_name": "pelvic_attachments"},
    "Rod.number": {"api_name": "rod_number"},
    "Device.type": {"api_name": "device_type"},
}

# Prediction at time of surgery; the model's outcome window is 2 years.
# days.from.surgery = 0 matches the authors' Shiny app exactly (verified).
FIXED_INPUTS = {"days.from.surgery": 0.0}
SCENARIO_VAR = "Treatment"
SCENARIO_LABELS = {"w/o HGT": "Without HGT", "w/ HGT": "With HGT"}


def main(extract_dir: Path, out_path: Path) -> None:
    with open(extract_dir / "coef.csv") as f:
        coef_rows = [(r["name"], float(r["value"])) for r in csv.DictReader(f)]

    with open(extract_dir / "var.csv") as f:
        var_rows = list(csv.DictReader(f))
    cov_names = [r["name"] for r in var_rows]
    cov = [[float(r[f"X{j+1}"]) for j in range(len(var_rows))] for r in var_rows]

    with open(extract_dir / "design.csv") as f:
        design = list(csv.DictReader(f))

    misc = {}
    for line in (extract_dir / "misc.txt").read_text().splitlines():
        k, v = line.split("=", 1)
        misc[k] = v

    ylevels = misc["ylevels"].split("|")
    yfreq = [int(x) for x in misc["yfreq"].split("|")]
    n_intercepts = int(misc["non_slopes"])

    # --- assertions from the plan ---
    assert [n for n, _ in coef_rows] == cov_names, "covariance/coefficient name mismatch"
    intercepts = coef_rows[:n_intercepts]
    slopes = coef_rows[n_intercepts:]
    assert len(slopes) == 26, f"expected 26 slopes, got {len(slopes)}"
    assert all(n.startswith("y>=") for n, _ in intercepts)
    # lowest outcome level must be the no-complication category (most frequent, coded 0)
    assert ylevels[0] == "0" and yfreq[0] == max(yfreq), (
        "lowest ylevel is not the no-complication category; intercept choice invalid"
    )

    variables = []
    scenario = None
    for row in design:
        name, kind, values = row["name"], row["kind"], row["values"]
        if kind == "interaction":
            continue  # structure recovered from coefficient names by the scorer
        if name == SCENARIO_VAR:
            levels = values.split("|")
            scenario = {
                "r_name": name,
                "r_levels": levels,
                "labels": [SCENARIO_LABELS[l] for l in levels],
            }
            continue
        if name in FIXED_INPUTS:
            assert kind == "rcspline"
            variables.append({
                "api_name": None,
                "r_name": name,
                "kind": "rcs3",
                "knots": [float(x) for x in values.split("|")],
                "fixed_value": FIXED_INPUTS[name],
            })
            continue
        spec = API_VARS[name]
        if kind == "rcspline":
            var = {
                "api_name": spec["api_name"],
                "r_name": name,
                "kind": "rcs3",
                "knots": [float(x) for x in values.split("|")],
            }
            if "input_transform" in spec:
                var["input_transform"] = spec["input_transform"]
            variables.append(var)
        elif kind == "category":
            r_levels = values.split("|")
            fixes = spec.get("level_fixes", {})
            variables.append({
                "api_name": spec["api_name"],
                "r_name": name,
                "kind": "factor",
                "r_levels": r_levels,
                "api_levels": [fixes.get(l, l) for l in r_levels],
            })
        else:
            raise ValueError(f"unhandled kind {kind} for {name}")
    assert scenario is not None

    model = {
        "type": "rms_lrm_scenario",
        "source": "fit0.rds (VUMC HGT model, rms::lrm via fit.mult.impute, "
                  f"{misc['n_impute']} imputations)",
        "n": int(float(misc["n_obs"])),
        "c_statistic": float(misc["c_stat"]),
        "ylevels": ylevels,
        "intercepts": {n: v for n, v in intercepts},
        "any_complication_intercept": intercepts[0][0],
        "coefficients": {n: v for n, v in slopes},
        "covariance": {"names": cov_names, "matrix": cov},
        "fixed_inputs": FIXED_INPUTS,
        "scenario_variable": scenario,
        "variables": variables,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(model, f, indent=1)
    print(f"Wrote {out_path} ({out_path.stat().st_size} bytes, "
          f"{len(slopes)} slopes, {len(variables)} variables)")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
