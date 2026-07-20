from fastapi import FastAPI

from app.main import app


def test_app_is_a_fastapi_instance():
    assert isinstance(app, FastAPI)
