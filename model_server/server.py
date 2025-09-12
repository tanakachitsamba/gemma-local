from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel

import os
import threading

from . import config

try:
    from llama_cpp import Llama
except Exception as e:  # pragma: no cover - hints for missing dependency during first setup
    Llama = None  # type: ignore
    _import_err = e
else:
    _import_err = None


app = FastAPI(title="Gemma Local Model Server", version="0.1.0")

_llama_lock = threading.Lock()
_generation_lock = threading.Lock()
_llama_instance: Optional["Llama"] = None


def get_llama() -> "Llama":
    global _llama_instance
    if _llama_instance is None:
        if Llama is None:
            raise RuntimeError(
                f"llama-cpp-python not available: {_import_err}.\n"
                "Install dependencies from model_server/requirements.txt and ensure the correct wheel for your platform."
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
    role: str
    content: str


class ChatRequest(BaseModel):
    model: Optional[str] = "gemma"
    messages: List[ChatMessage]
    stream: bool = True
    temperature: float = config.TEMPERATURE_DEFAULT
    max_tokens: int = config.MAX_TOKENS_DEFAULT
    stop: Optional[List[str]] = None
    seed: Optional[int] = None


def sse_event(data: Dict[str, Any]) -> bytes:
    return ("data: " + json.dumps(data, ensure_ascii=False) + "\n\n").encode("utf-8")


@app.get("/healthz")
def healthz():
    status = {
        "ok": True,
        "model_path": config.MODEL_PATH,
        "has_llama": Llama is not None,
    }
    return JSONResponse(status)


def _format_prompt(messages: List[Dict[str, str]]) -> str:
    # Simple generic chat-style prompt for models without chat templates
    # Format:
    # System: ...\nUser: ...\nAssistant: ...
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
    llama = get_llama()

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
                        for chunk in iterator:
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
                            # normalize to chat.completion.chunk shape
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
                yield sse_event(err)
            finally:
                yield b"data: [DONE]\n\n"

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
        return PlainTextResponse(str(e), status_code=500)


# Optional root for a quick info page
@app.get("/")
def root():
    return {
        "name": "Gemma Local Model Server",
        "endpoints": ["/healthz", "/v1/chat/completions"],
        "model_path": config.MODEL_PATH,
    }
