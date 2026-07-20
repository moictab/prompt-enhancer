from fastapi import Depends, FastAPI

from .auth import require_auth

app = FastAPI(title="Prompt Enhancer", dependencies=[Depends(require_auth)])
