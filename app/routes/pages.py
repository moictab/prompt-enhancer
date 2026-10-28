from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from .. import families
from ..config import get_settings
from ..prompts import VARIATION_AXES

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")

# Which axes the Variaciones form starts with set to "varies"; the rest start fixed.
DEFAULT_VARYING_AXES = {"outfit", "setting", "lighting", "pose"}


@router.get("/", response_class=HTMLResponse)
def index(request: Request):
    settings = get_settings()
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "families": families.list_families(),
            "variation_axes": [
                {"key": key, "label": label, "vary": key in DEFAULT_VARYING_AXES}
                for key, label in VARIATION_AXES.items()
            ],
            "default_llm_model": settings.default_llm_model,
            "default_vision_model": settings.default_vision_model,
        },
    )


@router.get("/historial", response_class=HTMLResponse)
def historial_page(request: Request):
    return templates.TemplateResponse(request, "historial.html", {})


@router.get("/admin", response_class=HTMLResponse)
def admin_page(request: Request):
    return templates.TemplateResponse(request, "admin.html", {})
