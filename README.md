# PSSG ML API

Centralized web API for machine learning models developed using the Pediatric Spine Study Group (PSSG) database.

## Overview

This API provides standardized endpoints for clinical prediction models built on the PSSG database, starting with a model to predict unplanned return to the operating room (UPROR) in early-onset scoliosis (EOS) patients.

## Quick Start

```bash
# Create and activate conda environment
conda create -n pssg-ml-api python=3.12 -y
conda activate pssg-ml-api

# Install dependencies
pip install -r requirements.txt

# Run the API locally
uvicorn app.main:app --reload

# Visit http://localhost:8000/docs for interactive API documentation
```

## Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API info |
| GET | `/health` | Health check |
| GET | `/models` | List available models |
| GET | `/models/{name}` | Model details and metadata |
| POST | `/predict/{name}` | Run prediction |
| GET | `/admin/usage` | Usage statistics |

## Running Tests

```bash
pytest -v
```

## Docker

```bash
docker compose up --build
```

## Adding a New Model

1. Create a folder in `app/models/artifacts/{model_name}/`
2. Add `model.pkl` (or `.joblib`) and `metadata.json`
3. Restart the API — the model is auto-discovered

See `app/models/artifacts/upror_growing_rod/metadata.json` for the metadata format.
