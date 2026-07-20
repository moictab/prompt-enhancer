from fastapi import Depends, FastAPI
from fastapi.staticfiles import StaticFiles

from .auth import require_auth
from .config import get_settings
from .routes import admin as admin_routes
from .routes import api as api_routes
from .routes import pages as page_routes
from .seed import ensure_seed_data

app = FastAPI(title="Prompt Enhancer", dependencies=[Depends(require_auth)])

app.include_router(api_routes.router)
app.include_router(admin_routes.router)
app.include_router(page_routes.router)
app.mount("/static", StaticFiles(directory="app/static"), name="static")


@app.on_event("startup")
def on_startup():
    ensure_seed_data(get_settings().data_dir)
