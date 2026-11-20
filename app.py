from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.config import CORS_ORIGINS, validate_settings
from src.orchestration import handle_question

app = FastAPI(
    title="Dami Socratic RAG API",
    version="1.0.0",
    description="Amas Socratic pipeline adapted to Dami, Gemini and Supabase.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)


class ChatResponse(BaseModel):
    answer: str


@app.get("/")
def root():
    return {"service": "Dami Socratic RAG", "status": "online"}


@app.get("/health")
def health():
    try:
        validate_settings()
        return {"status": "ok", "configuration": "present"}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest):
    try:
        answer = handle_question(payload.question)
        return ChatResponse(answer=answer)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        # Render logs retain the technical exception. The client receives
        # a generic message so secrets/provider payloads are not exposed.
        print(f"[DAMI_CHAT_ERROR] {type(exc).__name__}: {exc}", flush=True)
        raise HTTPException(
            status_code=502,
            detail="در پردازش پیام مشکلی پیش آمد. لطفاً کمی بعد دوباره تلاش کن.",
        )
