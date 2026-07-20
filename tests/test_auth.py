import pytest
from fastapi import HTTPException
from fastapi.security import HTTPBasicCredentials

from app.auth import require_auth


def test_require_auth_accepts_matching_credentials(monkeypatch):
    monkeypatch.setenv("ADMIN_USERNAME", "admin")
    monkeypatch.setenv("ADMIN_PASSWORD", "secret")
    creds = HTTPBasicCredentials(username="admin", password="secret")

    assert require_auth(creds) == "admin"


def test_require_auth_rejects_wrong_password(monkeypatch):
    monkeypatch.setenv("ADMIN_USERNAME", "admin")
    monkeypatch.setenv("ADMIN_PASSWORD", "secret")
    creds = HTTPBasicCredentials(username="admin", password="wrong")

    with pytest.raises(HTTPException) as exc_info:
        require_auth(creds)
    assert exc_info.value.status_code == 401


def test_require_auth_rejects_wrong_username(monkeypatch):
    monkeypatch.setenv("ADMIN_USERNAME", "admin")
    monkeypatch.setenv("ADMIN_PASSWORD", "secret")
    creds = HTTPBasicCredentials(username="someone-else", password="secret")

    with pytest.raises(HTTPException) as exc_info:
        require_auth(creds)
    assert exc_info.value.status_code == 401
