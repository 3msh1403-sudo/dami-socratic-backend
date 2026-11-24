from __future__ import annotations

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ValidationError

from src.config import CORS_ORIGINS, validate_settings
from src.orchestration import handle_question
from src.request_auth import RequestAuthError, verify_signed_request

app = FastAPI(
    title="Dami Socratic RAG API",
    version="1.1.0",
    description="Private HMAC-authenticated Dami Socratic RAG API.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=[
        "Content-Type", "Authorization",
        "x-dami-timestamp", "x-dami-nonce", "x-dami-signature",
    ],
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
async def chat(request: Request):
    # Authenticate the exact raw bytes before parsing or running the expensive RAG flow.
    raw_body = await request.body()
    if not raw_body or len(raw_body) > 16 * 1024:
        raise HTTPException(status_code=413, detail="Invalid request body.")

    try:
        verify_signed_request(
            method=request.method,
            path=request.url.path,
            raw_body=raw_body,
            timestamp=request.headers.get("x-dami-timestamp"),
            nonce=request.headers.get("x-dami-nonce"),
            signature=request.headers.get("x-dami-signature"),
        )
    except RequestAuthError as exc:
        reason = str(exc)
        status = 503 if ("not configured" in reason or "unavailable" in reason) else 401
        raise HTTPException(status_code=status, detail="Request authentication failed.")

    try:
        payload = ChatRequest.model_validate_json(raw_body)
    except ValidationError:
        raise HTTPException(status_code=422, detail="Invalid chat payload.")

    try:
        answer = handle_question(payload.question)
        return ChatResponse(answer=answer)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        print(f"[DAMI_CHAT_ERROR] {type(exc).__name__}: {exc}", flush=True)
        raise HTTPException(
            status_code=502,
            detail="در پردازش پیام مشکلی پیش آمد. لطفاً کمی بعد دوباره تلاش کن.",
        )
