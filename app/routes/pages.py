from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from .. import claude_code, comfy, families, references
from ..config import get_settings

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")
# Every page shows times in the app's zone (base.html: <html data-time-zone>, common.TIME_ZONE).
templates.env.globals["time_zone"] = lambda: get_settings().time_zone

@router.get("/")
def index():
    # Conversar is the app's home since the one-shot Crear page went.
    return RedirectResponse("/conversar")


@router.get("/conversar", response_class=HTMLResponse)
def conversar_page(request: Request):
    return templates.TemplateResponse(
        request,
        "conversar.html",
        {
            "families": families.list_families(),
            "default_llm_model": get_settings().default_llm_model,
            "comfy_configured": comfy.client() is not None,
            "claude_code_available": claude_code.available(),
            "claude_code_models": claude_code.MODELS,
            "default_claude_model": get_settings().claude_code_model,
            # Offered when "@" is typed in the composer: a message that names one gets its study.
            "reference_folders": [f["name"] for f in references.list_folders()],
        },
    )


@router.get("/peticiones", response_class=HTMLResponse)
def peticiones_page(request: Request):
    return templates.TemplateResponse(request, "peticiones.html", {})


@router.get("/historial", response_class=HTMLResponse)
def historial_page(request: Request):
    return templates.TemplateResponse(request, "historial.html", {})


@router.get("/admin", response_class=HTMLResponse)
def admin_page(request: Request):
    return templates.TemplateResponse(request, "admin.html", {
        # Who can tidy the memory (Ordenar la memoria): Claude Code, when installed, or an OpenRouter model.
        "claude_code_available": claude_code.available(),
        "default_llm_model": get_settings().default_llm_model,
    })
