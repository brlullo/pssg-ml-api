# PSSG ML API

A central web API for machine learning models built on the Pediatric Spine
Study Group (PSSG) database. Each model gets a typed prediction endpoint,
documentation served from its metadata, and a form in the bundled web UI.

## Models

| Model (`/predict/<name>`) | Predicts | Backend | Discrimination |
|---|---|---|---|
| `upror_growing_rod` | Probability of unplanned return to the OR (UPROR) after growing rod surgery in early-onset scoliosis (EOS); 8 features | scikit-learn logistic regression, isotonic calibration (`model.pkl`) | Test AUC 0.65 (n=2,106) |
| `hgt_complication` | 2-year probability of implant-related complication in EOS growing rod patients, **with vs. without** preoperative halo-gravity traction (95% CIs); 11 features | R `rms::lrm` proportional-odds model with restricted cubic splines, exported to `model.json` and scored in pure Python | C-statistic 0.70 (n=1,606) |

The HGT model was developed by VUMC Biostatistics and is hosted with the
authors' permission. The reference implementation is
<https://vumcbiostat.shinyapps.io/hgtmodel/>. `GET /models/<name>` returns
each model's full feature definitions, metrics, cohort and limitations.

## Quick Start

```bash
conda create -n pssg-ml-api python=3.12 -y
conda activate pssg-ml-api
pip install -r requirements.txt
uvicorn app.main:app --reload
```

- Web UI: <http://localhost:8000/>
- Interactive API docs (Swagger): <http://localhost:8000/docs>

## Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Web UI: a form per model; HGT shows a with/without-traction comparison table |
| GET | `/api` | API info and links |
| GET | `/health` | Health check (reports number of models loaded) |
| GET | `/models` | List available models |
| GET | `/models/{name}` | Model details: features, validation metrics, cohort, limitations |
| POST | `/predict/upror_growing_rod` | UPROR probability |
| POST | `/predict/hgt_complication` | Complication probability for both traction scenarios, with 95% CIs |
| GET | `/admin/usage` | Usage statistics (requires `X-API-Key` when `ADMIN_API_KEY` is set) |

At startup a dedicated `POST /predict/<name>` route is registered for every
model. The request body is a flat JSON object with one field per feature, and
Swagger shows a pre-filled example. A missing or invalid field returns a 422
response that names the field.

```bash
curl -X POST http://localhost:8000/predict/upror_growing_rod \
  -H 'Content-Type: application/json' \
  -d '{"Post-Index_Inferior_Attachments_Spine_Based": 1, "Pre-Index_Height_cm": 110,
       "Post-Index_Device_Type_VEPTR": 0, "Post-Index_Device_Type_TGR": 1,
       "Pre-Index_Kyphosis": 40, "age_at_surgery": 6,
       "Post-Index_Number_of_Superior_Anchors": 2, "Gender_Male": 1}'
# {"prediction": 0, "probability": 0.3844, "threshold": 0.4028, "model_name": "upror_growing_rod", "model_version": "1.0.0"}
```

`/docs` has a working example body for every model under **Try it out**.

## Configuration

Settings are read from environment variables or a `.env` file (see
`.env.example`):

| Variable | Default | Purpose |
|---|---|---|
| `DATABASE_URL` | empty | Postgres connection string for request logging; when empty, logs go to SQLite at `LOG_DB_PATH` |
| `LOG_DB_PATH` | `api_logs.db` | SQLite log file used when `DATABASE_URL` is unset |
| `ADMIN_API_KEY` | empty | When set, `/admin/usage` requires a matching `X-API-Key` header |
| `ENVIRONMENT`, `API_VERSION` | `development`, `0.1.0` | Informational |

Every `/predict/*` request is logged with its model, latency and status code.
Only this metadata is recorded; the patient input values are not.

## Running Tests

```bash
pytest -v
```

The HGT tests compare Python predictions against golden values produced by
the original R model (`tests/fixtures/hgt_golden.json`).

## Deployment

- **Docker:** `docker compose up --build`
- **Render:** `render.yaml` defines a Docker web service with a `/health`
  check. Set `DATABASE_URL` and `ADMIN_API_KEY` in the Render dashboard.
- **CI:** GitHub Actions runs `pytest` on every push and pull request.

## Adding a New Model

Add a folder `app/models/artifacts/<model_name>/` containing `metadata.json`
plus one of the following:

- `model.pkl` / `model.joblib`: a scikit-learn estimator (dict with `model` or
  `calibrated_model`, optional `scaler`, and `feature_names`)
- `model.json`: an R `rms::lrm` model exported with
  `scripts/extract_hgt_model.R` and `scripts/build_model_json.py`. No R
  runtime is needed in production.

The model is discovered at startup, with no code changes needed. See
[docs/adding-a-model.md](docs/adding-a-model.md) for the export convention,
metadata format, UI options and verification checklist.
