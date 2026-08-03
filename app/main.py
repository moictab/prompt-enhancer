import logging
import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request
from fastapi.staticfiles import StaticFiles

from .auth import require_auth
from .config import get_settings
from .logging_config import configure_logging
from .routes import admin as admin_routes
from .routes import api as api_routes
from .routes import pages as page_routes
from .seed import ensure_seed_data

request_logger = logging.getLogger("app.request")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.data_dir)
    if not settings.admin_password:
        raise RuntimeError(
            "ADMIN_PASSWORD is not set. Refusing to start with an unconfigured "
            "auth boundary — set ADMIN_PASSWORD in your .env before running the app."
        )
    ensure_seed_data(settings.data_dir)
    yield


app = FastAPI(title="Prompt Enhancer", dependencies=[Depends(require_auth)], lifespan=lifespan)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.monotonic()
    response = await call_next(request)
    duration_ms = (time.monotonic() - start) * 1000
    request_logger.info(
        "%s %s -> %d (%.1fms)",
        request.method, request.url.path, response.status_code, duration_ms,
    )
    return response


app.include_router(api_routes.router)
app.include_router(admin_routes.router)
app.include_router(page_routes.router)
app.mount("/static", StaticFiles(directory="app/static"), name="static")
