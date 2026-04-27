# -*- coding: utf-8 -*-
"""
PILAR LLM Engine — Qwen3 4B via llama-cpp-python
=================================================
Singleton model loader. Runs locally on CPU, zero external dependency,
zero token cost, works fully offline after first model download.

Public API:
    is_available() -> bool
    query(system, user, max_tokens, temperature) -> str | None
    preload()   -- call at startup to warm the model before first request
"""

import os
import re
import threading
from pathlib import Path
from typing import Optional

try:
    from llama_cpp import Llama
    HAS_LLAMA_CPP = True
except ImportError:
    HAS_LLAMA_CPP = False

# ── Model location ─────────────────────────────────────────────────────────────
MODEL_FILENAME = "Qwen3-4B-Q4_K_M.gguf"
MODEL_DIR      = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "PILAR" / "models"
MODEL_URL      = (
    "https://huggingface.co/bartowski/Qwen3-4B-GGUF"
    "/resolve/main/Qwen3-4B-Q4_K_M.gguf"
)

# ── Singleton state ────────────────────────────────────────────────────────────
_lock:            threading.Lock     = threading.Lock()
_llm:             Optional["Llama"]  = None
_load_attempted:  bool               = False

# Strip Qwen3 chain-of-thought blocks before returning to caller
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


# ── Helpers ────────────────────────────────────────────────────────────────────

def model_path() -> Path:
    return MODEL_DIR / MODEL_FILENAME


def is_available() -> bool:
    """True only when llama-cpp-python is installed AND the GGUF file exists."""
    return HAS_LLAMA_CPP and model_path().exists()


def _load() -> Optional["Llama"]:
    global _llm, _load_attempted
    if _load_attempted:
        return _llm
    _load_attempted = True

    if not is_available():
        return None

    try:
        n_threads = max(1, (os.cpu_count() or 4) - 1)
        _llm = Llama(
            model_path=str(model_path()),
            n_ctx=4096,
            n_threads=n_threads,
            n_gpu_layers=0,        # CPU-only — no GPU assumed
            chat_format="chatml",  # Qwen3 uses ChatML
            verbose=False,
        )
        print(f"[LLMEngine] Qwen3 4B ready ({n_threads} threads, CPU)")
    except Exception as exc:
        print(f"[LLMEngine] load failed: {exc}")
        _llm = None

    return _llm


def _get() -> Optional["Llama"]:
    with _lock:
        if _llm is not None:
            return _llm
        return _load()


def _strip_think(text: str) -> str:
    """Remove <think>…</think> reasoning blocks Qwen3 may emit."""
    return _THINK_RE.sub("", text).strip()


# ── Public API ─────────────────────────────────────────────────────────────────

def preload() -> bool:
    """
    Eagerly load the model at startup so the first real request is fast.
    Safe to call from a background thread. Returns True if model loaded.
    """
    return _get() is not None


def query(system: str, user: str,
          max_tokens: int = 512,
          temperature: float = 0.3) -> Optional[str]:
    """
    Send a prompt to Qwen3 4B and return the cleaned response text.

    Returns None if the model is not available (caller should use rule-based
    fallback). The <think> reasoning block is stripped automatically.

    Note: llama-cpp-python is not thread-safe — calls are serialized via lock.
    For a desktop app this is fine; agents run in a single background thread.
    """
    llm = _get()
    if llm is None:
        return None

    try:
        with _lock:
            result = llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                stop=["<|im_end|>", "<|endoftext|>"],
            )
        text = result["choices"][0]["message"]["content"]
        return _strip_think(text).strip()
    except Exception as exc:
        print(f"[LLMEngine] inference error: {exc}")
        return None
