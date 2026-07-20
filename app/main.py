from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from fastapi.staticfiles import StaticFiles

from .auth import require_auth
from .config import get_settings
from .routes import admin as admin_routes
from .routes import api as api_routes
from .routes import pages as page_routes
from .seed import ensure_seed_data


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    if not settings.admin_password:
        raise RuntimeError(
            "ADMIN_PASSWORD is not set. Refusing to start with an unconfigured "
            "auth boundary — set ADMIN_PASSWORD in your .env before running the app."
        )
    ensure_seed_data(settings.data_dir)
    yield


app = FastAPI(title="Prompt Enhancer", dependencies=[Depends(require_auth)], lifespan=lifespan)

app.include_router(api_routes.router)
app.include_router(admin_routes.router)
app.include_router(page_routes.router)
app.mount("/static", StaticFiles(directory="app/static"), name="static")
