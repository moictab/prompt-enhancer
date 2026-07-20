from fastapi import Depends, FastAPI

from .auth import require_auth
from .routes import admin as admin_routes
from .routes import api as api_routes

app = FastAPI(title="Prompt Enhancer", dependencies=[Depends(require_auth)])

app.include_router(api_routes.router)
app.include_router(admin_routes.router)
