"""Minimal, human-friendly FastAPI server for local LLM inference.

This module intentionally keeps the surface area small and boring:
- A single `/v1/chat/completions` endpoint that streams or returns a full
  response in an OpenAI-like shape.
- Conservative concurrency using a global generation lock to avoid GPU/CPU
  thrash on consumer hardware.
- Clear error paths and a health endpoint for quick checks.

Design notes (for humans):
- We lazily create a single `Llama` instance and reuse it across requests. This
  trades first-token latency for simpler lifecycle management and lower memory
  churn. If you need multi-model or true concurrency, prefer an orchestrator
  process per model and a queue in front of them.
- The fallback from chat-completions to plain completions narrows API surface
  area while still supporting older GGUFs without chat templates.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field

import threading
import asyncio
import concurrent.futures

from . import config
from . import metrics

try:
    # Import may fail in CI or before the wheel is installed. That's OK:
    # tests monkeypatch `get_llama` so we don't require the dependency there.
    from llama_cpp import Llama
except Exception as e:  # pragma: no cover
    Llama = None  # type: ignore
    _import_err = e
else:
    _import_err = None


app = FastAPI(title="Gemma Local Model Server", version="0.1.0")

# A coarse lock for model construction and a generation lock to serialize
# token generation on modest hardware. This avoids OOMs and context thrashing.
_llama_lock = threading.Lock()
_generation_lock = threading.Lock()
_llama_instance: Optional["Llama"] = None
_req_semaphore = threading.BoundedSemaphore(value=max(1, config.MAX_CONCURRENT_REQUESTS))


def get_llama() -> "Llama":
    """Return a process-wide singleton Llama instance.

    We construct lazily and keep the instance hot. If `llama_cpp` is not
    importable, provide a clear, actionable error to the caller.
    """
    global _llama_instance
    if _llama_instance is None:
        if Llama is None:
            raise RuntimeError(
                (
                    f"llama-cpp-python not available: {_import_err}.\n"
                    "Install from model_server/requirements.txt and ensure the wheel matches your platform."
                )
            )
        with _llama_lock:
            if _llama_instance is None:
                _llama_instance = Llama(
                    model_path=config.MODEL_PATH,
                    n_ctx=config.N_CTX,
                    n_threads=config.N_THREADS,
                    n_gpu_layers=config.N_GPU_LAYERS,
                    verbose=False,
                )
    return _llama_instance


class ChatMessage(BaseModel):
    """A minimal chat message with a role and content."""

    role: str
    content: str


class ChatRequest(BaseModel):
    """Request payload mirroring a subset of OpenAI's API."""

    model: Optional[str] = "gemma"
    messages: List[ChatMessage] = Field(..., min_length=1)
    stream: bool = True
    temperature: float = config.TEMPERATURE_DEFAULT
    max_tokens: int = Field(
        default=config.MAX_TOKENS_DEFAULT, gt=0, le=config.MAX_TOKENS_LIMIT
    )
    stop: Optional[List[str]] = None
    seed: Optional[int] = None


def sse_event(data: Dict[str, Any]) -> bytes:
    """Encode a JSON-compatible dict as an SSE data line."""

    return ("data: " + json.dumps(data, ensure_ascii=False) + "\n\n").encode("utf-8")


@app.get("/healthz")
def healthz():
    """Cheap health signal for uptime checks and local sanity tests."""

    status = {
        "ok": True,
        "model_path": config.MODEL_PATH,
        "has_llama": Llama is not None,
    }
    return JSONResponse(status)


def _format_prompt(messages: List[Dict[str, str]]) -> str:
    """Format a generic prompt from chat-style messages.

    Many base models lack a chat template. This formatting is deliberately
    simple and readable when inspecting prompts in logs.
    """

    lines: List[str] = []
    for m in messages:
        role = m.get("role", "user").capitalize()
        if role == "System":
            lines.append(f"System: {m.get('content','')}")
        elif role == "User":
            lines.append(f"User: {m.get('content','')}")
        elif role == "Assistant":
            lines.append(f"Assistant: {m.get('content','')}")
        else:
            lines.append(f"{role}: {m.get('content','')}")
    lines.append("Assistant:")
    return "\n".join(lines)


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest, request: Request):
    """Chat endpoint with streaming and completion fallback.

    The output shape tracks OpenAI's responses closely enough to drop into
    lightweight clients and demos without extra glue code.
    """

    started = time.time()
    # Rate limit: acquire within timeout or return 429
    loop = asyncio.get_running_loop()
    timeout_s = max(0.001, config.ACQUIRE_TIMEOUT_MS / 1000.0)
    def _acq() -> bool:
        return _req_semaphore.acquire(timeout=timeout_s)
    acquired = await loop.run_in_executor(None, _acq)
    if not acquired:
        metrics.record_error()
        return PlainTextResponse("Too Many Requests", status_code=429)

    try:
        llama = get_llama()
    except Exception as e:
        metrics.record_error()
        return PlainTextResponse(str(e), status_code=500)

    # Prepend optional system prompt if not already present
    messages: List[Dict[str, str]] = [m.model_dump() for m in req.messages]
    if not messages or messages[0].get("role") != "system":
        messages = [{"role": "system", "content": config.SYSTEM_PROMPT}] + messages

    created = int(time.time())

    if req.stream:
        def token_stream():
            try:
                with _generation_lock:
                    try:
                        iterator = llama.create_chat_completion(
                            messages=messages,
                            temperature=req.temperature,
                            max_tokens=req.max_tokens,
                            stream=True,
                            stop=req.stop,
                            seed=req.seed,
                        )
                        deadline = time.time() + max(1, config.STREAMING_TIMEOUT_S)
                        for chunk in iterator:
                            if time.time() > deadline:
                                break
                            chunk["id"] = chunk.get("id", f"chatcmpl-{created}")
                            chunk["object"] = "chat.completion.chunk"
                            chunk["created"] = created
                            chunk["model"] = req.model or "gemma"
                            yield sse_event(chunk)
                    except Exception:
                        # Fallback to plain completion with our prompt formatter
                        prompt = _format_prompt(messages)
                        for chunk in llama.create_completion(
                            prompt=prompt,
                            temperature=req.temperature,
                            max_tokens=req.max_tokens,
                            stream=True,
                            stop=req.stop,
                            seed=req.seed,
                        ):
                            # Normalize to chat.completion.chunk shape
                            text = ""
                            try:
                                text = chunk["choices"][0]["text"]
                            except Exception:
                                text = str(chunk)
                            payload = {
                                "id": f"chatcmpl-{created}",
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": req.model or "gemma",
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {"content": text},
                                        "finish_reason": None,
                                    }
                                ],
                            }
                            yield sse_event(payload)
            except Exception as e:
                err = {"error": {"message": str(e)}}
                metrics.record_error()
                yield sse_event(err)
            finally:
                # Explicitly signal the end of stream for simple clients
                yield b"data: [DONE]\n\n"
                metrics.record_request(started, streaming=True)
                try:
                    _req_semaphore.release()
                except Exception:
                    pass

        headers = {
            "Cache-Control": "no-cache",
            "Content-Type": "text/event-stream",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
        return StreamingResponse(token_stream(), headers=headers)

    # Non-streaming
    try:
        with _generation_lock:
            try:
                result = llama.create_chat_completion(
                    messages=messages,
                    temperature=req.temperature,
                    max_tokens=req.max_tokens,
                    stream=False,
                    stop=req.stop,
                    seed=req.seed,
                )
                result.setdefault("id", f"chatcmpl-{created}")
                result.setdefault("object", "chat.completion")
                result.setdefault("created", created)
                result.setdefault("model", req.model or "gemma")
                return JSONResponse(result)
            except Exception:
                prompt = _format_prompt(messages)
                result = llama.create_completion(
                    prompt=prompt,
                    temperature=req.temperature,
                    max_tokens=req.max_tokens,
                    stream=False,
                    stop=req.stop,
                    seed=req.seed,
                )
                # Build OpenAI-like response
                text = ""
                try:
                    text = result["choices"][0]["text"]
                except Exception:
                    text = str(result)
                payload = {
                    "id": f"chatcmpl-{created}",
                    "object": "chat.completion",
                    "created": created,
                    "model": req.model or "gemma",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": text},
                            "finish_reason": result.get("choices", [{}])[0].get("finish_reason", "stop"),
                        }
                    ],
                }
                return JSONResponse(payload)
    except Exception as e:
        # Return a clear 500 so clients can surface errors cleanly
        metrics.record_error()
        return PlainTextResponse(str(e), status_code=500)
    finally:
        try:
            _req_semaphore.release()
        except Exception:
            pass
        if not req.stream:
            metrics.record_request(started, streaming=False)


@app.get("/")
def root():
    """Tiny index for quick discovery during local testing."""

    return {
        "name": "Gemma Local Model Server",
        "endpoints": ["/healthz", "/v1/chat/completions"],
        "model_path": config.MODEL_PATH,
    }


@app.get("/metrics")
def prometheus_metrics():
    return PlainTextResponse(metrics.prometheus_text(), media_type="text/plain; version=0.0.4")
