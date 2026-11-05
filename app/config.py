import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODEL = "anthropic/claude-sonnet-4.6"


@dataclass(frozen=True)
class Settings:
    openrouter_api_key: str
    admin_username: str
    admin_password: str
    data_dir: str
    default_llm_model: str
    # comfy-gateway, and ComfyUI itself as a fallback; both unset turns ComfyUI runs off.
    comfy_gateway_url: str = ""
    comfy_direct_url: str = ""
    # How long a job may run in ComfyUI without a result before it's given up (runs.py): counted
    # from when it starts, never while it waits its turn. A run's own `timeout_seconds` overrides it.
    comfy_job_timeout_minutes: int = 45
    # Civitai's API key: images shared by URL are read with it (optional).
    civitai_api_key: str = ""
    # Claude Code as Conversar's bot (claude_code.py): the command, the configuration folder whose
    # login it uses (an account of its own, not the one Claude Code uses elsewhere on the PC), the
    # model new conversations start with, and the address its MCP calls come back to (this app).
    claude_code_command: str = "claude"
    claude_code_config_dir: str = ""
    claude_code_model: str = "opus"
    self_url: str = "http://127.0.0.1:8090"
    # ComfyUI's models folder (its loras, checkpoints and diffusion_models): where downloads from
    # Civitai go, each one accepted by the user (downloads.py). Unset, downloads are off.
    comfy_models_dir: str = ""
    # The time zone the pages show times in (the PC's; a browser may report another, or UTC).
    time_zone: str = "Europe/Madrid"


def _positive_int(value: str | None, default: int) -> int:
    """A whole number above zero from the environment, or the default (unset, empty or not one)."""
    try:
        number = int((value or "").strip())
    except ValueError:
        return default
    return number if number > 0 else default


def get_settings() -> Settings:
    return Settings(
        openrouter_api_key=os.environ.get("OPENROUTER_API_KEY", ""),
        admin_username=os.environ.get("ADMIN_USERNAME", "admin"),
        admin_password=os.environ.get("ADMIN_PASSWORD", ""),
        data_dir=os.environ.get("DATA_DIR", "data"),
        default_llm_model=os.environ.get("DEFAULT_LLM_MODEL", DEFAULT_MODEL),
        comfy_gateway_url=os.environ.get("COMFY_GATEWAY_URL", "").strip(),
        comfy_direct_url=os.environ.get("COMFY_DIRECT_URL", "").strip(),
        comfy_job_timeout_minutes=_positive_int(os.environ.get("COMFY_JOB_TIMEOUT_MINUTES"), 45),
        civitai_api_key=os.environ.get("CIVITAI_API_KEY", "").strip(),
        claude_code_command=os.environ.get("CLAUDE_CODE_COMMAND", "claude").strip() or "claude",
        claude_code_config_dir=os.path.expanduser(
            os.environ.get("CLAUDE_CODE_CONFIG_DIR", "").strip() or "~/.claude-prompt-enhancer"),
        claude_code_model=os.environ.get("CLAUDE_CODE_MODEL", "").strip() or "opus",
        self_url=(os.environ.get("SELF_URL", "").strip() or "http://127.0.0.1:8090").rstrip("/"),
        comfy_models_dir=os.environ.get("COMFY_MODELS_DIR", "").strip(),
        time_zone=os.environ.get("TIME_ZONE", "").strip() or "Europe/Madrid",
    )
