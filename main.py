# Render start command: uvicorn main:app --host 0.0.0.0 --port $PORT
from app import app

__all__ = ["app"]
