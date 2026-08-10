# Adding a model to the PSSG ML API

Adding a model requires **no code changes**. The registry auto-discovers every
subfolder of `app/models/artifacts/` at startup, and the web UI auto-generates
its input form from the model's metadata. You only need two files:

```
app/models/artifacts/<model_name>/
├── model.pkl        (or model.joblib)
└── metadata.json
```

The folder name (`<model_name>`) becomes the API identifier:
`POST /predict/<model_name>`, `GET /models/<model_name>`. At startup the API
generates a dedicated, typed prediction endpoint for the model — the request
body is a flat JSON object with one field per feature, and Swagger (`/docs`)
gets a pre-filled example built from the metadata.

## 1. Export the pickle

Save a **dict** with these keys (see `app/models/registry.py:51-62`):

| key | required | contents |
|---|---|---|
| `model` | yes* | fitted estimator with `predict_proba` |
| `calibrated_model` | no | if present, used **instead of** `model` (e.g. `CalibratedClassifierCV`) |
| `scaler` | no | fitted scaler (e.g. `StandardScaler`); applied to inputs before prediction |
| `feature_names` | yes | ordered list of feature names, matching training column order |

*required unless `calibrated_model` is provided.

Export snippet — run this at the end of your model-development notebook:

```python
import pickle

artifact = {
    "model": clf,                      # or "calibrated_model": calibrated_clf
    "scaler": scaler,                  # omit if features are not scaled
    "feature_names": list(X_train.columns),
}
with open("model.pkl", "wb") as f:
    pickle.dump(artifact, f)
```

> **scikit-learn version warning:** pickles are only reliable when loaded under
> the same scikit-learn version they were saved with. Export your model in an
> environment with the **exact version pinned in `requirements.txt`**
> (currently `scikit-learn==1.8.0`). If the API's pin is ever upgraded,
> re-export every model under the new version.

## 2. Write metadata.json

The API serves this verbatim at `GET /models/<model_name>`, and the web UI
builds the input form from `features[]` — so `type` and `description` must be
accurate. `type` is `"binary"` (rendered as Yes/No) or `"continuous"`
(rendered as a number input). `example_value` (optional) seeds the Swagger
example for continuous features. `metrics.optimal_threshold` is used as the
classification cutoff (defaults to 0.5 if omitted).

Template:

```json
{
  "display_name": "Human-readable model name",
  "description": "One-sentence description of what the model predicts and for whom",
  "version": "1.0.0",
  "algorithm": "Logistic Regression",
  "framework": "scikit-learn",
  "features": [
    {
      "name": "feature_name_exactly_as_trained",
      "type": "binary",
      "description": "What this feature means, incl. coding (1 = X, 0 = Y)"
    },
    {
      "name": "another_feature",
      "type": "continuous",
      "description": "What this feature means, incl. units",
      "example_value": 42.0
    }
  ],
  "metrics": {
    "test_auc": 0.0,
    "cv_auc": 0.0,
    "brier_score": 0.0,
    "sensitivity": 0.0,
    "specificity": 0.0,
    "optimal_threshold": 0.5
  },
  "cohort": {
    "total_patients": 0,
    "train_patients": 0,
    "test_patients": 0
  },
  "last_updated": "YYYY-MM-DD",
  "limitations": "Cohort exclusions, discrimination caveats, calibration notes."
}
```

`app/models/artifacts/upror_growing_rod/metadata.json` is a complete working
example.

## 3. Verify

```bash
pytest -v                      # existing tests still pass
uvicorn app.main:app --reload  # then check:
```

- `GET /models` lists the new model
- The web UI at `/` shows a card for it and generates the correct form
- `/docs` shows a dedicated endpoint for the model with a pre-filled example
- `POST /predict/<model_name>` with a flat JSON body of feature values
  returns a probability
- A request with missing features returns a 422 naming the missing fields

Commit the folder (model pickles in `app/models/artifacts/` are tracked in
git) and push — Render redeploys and the model is live.
