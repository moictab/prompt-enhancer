import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODEL = "anthropic/claude-sonnet-4"


@dataclass(frozen=True)
class Settings:
    openrouter_api_key: str
    admin_username: str
    admin_password: str
    data_dir: str
    default_llm_model: str
    default_vision_model: str


def get_settings() -> Settings:
    return Settings(
        openrouter_api_key=os.environ.get("OPENROUTER_API_KEY", ""),
        admin_username=os.environ.get("ADMIN_USERNAME", "admin"),
        admin_password=os.environ.get("ADMIN_PASSWORD", ""),
        data_dir=os.environ.get("DATA_DIR", "data"),
        default_llm_model=os.environ.get("DEFAULT_LLM_MODEL", DEFAULT_MODEL),
        default_vision_model=os.environ.get("DEFAULT_VISION_MODEL", DEFAULT_MODEL),
    )
