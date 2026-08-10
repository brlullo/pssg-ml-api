import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.background import BackgroundTask

from app.config import settings
from app.logging.db import get_usage_stats, init_db, log_request
from app.models.registry import registry
from app.routers import models_info
from app.routers.predict import make_predict_handler
from app.schemas.predict import PredictionResponse

STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    registry.load_all()
    loaded = registry.list_models()
    print(f"Loaded {len(loaded)} model(s): {loaded}")

    # Register a dedicated prediction endpoint for every loaded model.
    # Each endpoint gets its own Swagger entry, display name, description,
    # and a pre-filled example request body derived from the model's metadata.
    # getattr: app.routes can contain mount/router entries without a .path
    existing_paths = {getattr(route, "path", None) for route in app.routes}
    for model_name in loaded:
        path = f"/predict/{model_name}"
        if path in existing_paths:
            continue
        metadata = registry.get_metadata(model_name)
        if metadata is None:
            continue
        display_name = metadata.get("display_name", model_name)
        request_model = registry.make_request_model(model_name)
        handler = make_predict_handler(model_name, request_model)
        app.add_api_route(
            path,
            handler,
            methods=["POST"],
            response_model=PredictionResponse,
            tags=[display_name],
            summary=f"Predict — {display_name}",
            description=metadata.get("description", ""),
        )

    try:
        init_db()
    except Exception as e:
        print(f"[logging] could not initialize log database: {e}")
    yield


app = FastAPI(
    title="PSSG ML API",
    description=(
        "Centralized API for machine learning models developed using the "
        "Pediatric Spine Study Group (PSSG) database. Provides standardized "
        "endpoints for clinical prediction and model documentation."
    ),
    version=settings.API_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_predict_requests(request: Request, call_next):
    """Log every /predict/* request with its real status code (200, 404, 422, 500)."""
    if not request.url.path.startswith("/predict/"):
        return await call_next(request)

    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000

    model_name = request.url.path.removeprefix("/predict/").strip("/")
    task = BackgroundTask(
        log_request, model_name, request.url.path, elapsed_ms, response.status_code
    )
    # Chain rather than clobber any background task set by the endpoint
    if response.background is None:
        response.background = task
    else:
        existing = response.background
        response.background = BackgroundTask(
            lambda: (existing(), task())
        )
    return response


app.include_router(models_info.router)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def home():
    """Web interface for interacting with the models."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api", tags=["Info"])
def api_info():
    """API root with basic information and links."""
    return {
        "name": "PSSG ML API",
        "version": settings.API_VERSION,
        "description": "Machine learning models from the Pediatric Spine Study Group database",
        "docs_url": "/docs",
        "models_url": "/models",
    }


@app.get("/health", tags=["Info"])
def health():
    """Health check endpoint for deployment monitoring."""
    return {"status": "healthy", "models_loaded": len(registry.list_models())}


def require_admin(x_api_key: str | None = Header(default=None)):
    """Require X-API-Key when ADMIN_API_KEY is configured; open in local dev."""
    if settings.ADMIN_API_KEY and x_api_key != settings.ADMIN_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")


@app.get("/admin/usage", tags=["Admin"], dependencies=[Depends(require_admin)])
def usage():
    """API usage statistics: total queries, per-model counts, monthly trends."""
    return get_usage_stats()
