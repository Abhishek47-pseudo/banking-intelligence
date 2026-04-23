"""
LLM Factory — resolves the best available LLM provider with runtime fallback.

Priority:
  1. OpenAI  (gpt-4o)           — if OPENAI_API_KEY is set and non-empty
  2. Groq    (llama-3.3-70b)    — if GROQ_API_KEY is set (fallback)

Fallback operates at TWO levels:
  - Init-time : if OpenAI key is missing/invalid, Groq is used from the start.
  - Runtime   : if OpenAI raises any exception during a call (quota, network,
                rate-limit, etc.) the request is transparently retried on Groq.

Both providers expose an identical LangChain BaseChatModel interface, so the
rest of the codebase is completely provider-agnostic.
"""

import logging
import os
from typing import Any, Iterator, List, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult

log = logging.getLogger(__name__)

# Ensure `.env` is loaded for CLI scripts/tests (FastAPI already does this in
# `backend/main.py`, but `test_pipeline.py` imports the factory directly).
try:
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv(), override=False)
except Exception:
    # Keep import-time behavior safe even if python-dotenv isn't installed
    # or the .env file isn't present in the current environment.
    pass

# LangChain callbacks to record token usage + tool calls
try:
    from backend.observability.usage_logger import get_usage_callbacks

    _USAGE_CALLBACKS = get_usage_callbacks()
except Exception:
    _USAGE_CALLBACKS = []

# ── Provider constants ────────────────────────────────────────────────────────

OPENAI_MODEL = "gpt-4o"
GROQ_MODEL   = "llama-3.3-70b-versatile"   # best general Groq model
TEMPERATURE  = 0.2


def _has_key(env_var: str) -> bool:
    val = os.getenv(env_var, "").strip()
    return bool(val) and not val.startswith("sk-your")


# ── Runtime-fallback wrapper ──────────────────────────────────────────────────

class FallbackLLM(BaseChatModel):
    """
    Wraps a primary LLM with a fallback.  Any exception raised during
    _generate / _stream is caught and the request is retried on the fallback.
    Fully compatible with LangChain's BaseChatModel interface.
    """

    primary: BaseChatModel
    fallback: BaseChatModel

    class Config:
        arbitrary_types_allowed = True

    @property
    def _llm_type(self) -> str:
        return f"fallback({self.primary._llm_type} → {self.fallback._llm_type})"

    # ── Core call ─────────────────────────────────────────────────────────────

    def _generate(self, messages: List[BaseMessage], **kwargs) -> ChatResult:
        try:
            return self.primary._generate(messages, **kwargs)
        except Exception as exc:
            log.warning(
                f"Primary LLM ({self.primary._llm_type}) failed at call-time "
                f"({type(exc).__name__}: {exc}). Switching to fallback."
            )
            return self.fallback._generate(messages, **kwargs)

    async def _agenerate(self, messages: List[BaseMessage], **kwargs) -> ChatResult:
        try:
            return await self.primary._agenerate(messages, **kwargs)
        except Exception as exc:
            log.warning(
                f"Primary LLM ({self.primary._llm_type}) failed at call-time "
                f"({type(exc).__name__}: {exc}). Switching to fallback."
            )
            return await self.fallback._agenerate(messages, **kwargs)

    # ── Tool / function call pass-through ─────────────────────────────────────
    # LangChain agents inspect these attributes; delegate to primary first,
    # then fallback so tool-calling schemas are preserved correctly.

    def bind_tools(self, tools, **kwargs):
        """Return a new FallbackLLM whose primary+fallback both have tools bound."""
        return FallbackLLM(
            primary=self.primary.bind_tools(tools, **kwargs),
            fallback=self.fallback.bind_tools(tools, **kwargs),
        )

    # ── Identifying params (required by BaseChatModel) ────────────────────────

    @property
    def _identifying_params(self) -> dict:
        return {
            "primary": self.primary._identifying_params,
            "fallback": self.fallback._identifying_params,
        }


# ── Builder ───────────────────────────────────────────────────────────────────

def _build_openai() -> Optional[BaseChatModel]:
    """Attempt to build a ChatOpenAI instance. Returns None on failure."""
    if not _has_key("OPENAI_API_KEY"):
        return None
    try:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model=OPENAI_MODEL,
            temperature=TEMPERATURE,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            callbacks=_USAGE_CALLBACKS,
        )
        log.info(f"LLM provider: OpenAI ({OPENAI_MODEL})")
        return llm
    except Exception as e:
        log.warning(f"OpenAI init failed ({e}).")
        return None


def _build_groq() -> Optional[BaseChatModel]:
    """Attempt to build a ChatGroq instance. Returns None on failure."""
    if not _has_key("GROQ_API_KEY"):
        return None
    try:
        from langchain_groq import ChatGroq
        llm = ChatGroq(
            model=GROQ_MODEL,
            temperature=TEMPERATURE,
            groq_api_key=os.getenv("GROQ_API_KEY"),
            callbacks=_USAGE_CALLBACKS,
        )
        log.info(f"LLM provider: Groq ({GROQ_MODEL})")
        return llm
    except Exception as e:
        log.warning(f"Groq init failed: {e}")
        return None


def build_llm() -> BaseChatModel:
    """
    Build and return the best available chat model.

    - If both OpenAI and Groq are available: returns a FallbackLLM that
      calls OpenAI first and automatically retries on Groq on any error.
    - If only Groq is available: returns Groq directly.
    - If only OpenAI is available: returns OpenAI directly (no fallback).
    - If neither is available: returns a stub that will raise a clear error.
    """
    openai_llm = _build_openai()
    groq_llm   = _build_groq()

    if openai_llm and groq_llm:
        log.info("Both providers available — OpenAI primary, Groq runtime fallback.")
        return FallbackLLM(primary=openai_llm, fallback=groq_llm)

    if groq_llm:
        log.info("Using Groq as sole provider (no OpenAI key).")
        return groq_llm

    if openai_llm:
        log.info("Using OpenAI as sole provider (no Groq key).")
        return openai_llm

    # ── No provider available — warn loudly ───────────────────────────────────
    log.error(
        "No LLM provider available! Set OPENAI_API_KEY or GROQ_API_KEY in .env. "
        "Returning a stub that will raise on invocation."
    )
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model=OPENAI_MODEL,
        temperature=TEMPERATURE,
        openai_api_key="MISSING",
    )


# ── Singleton ─────────────────────────────────────────────────────────────────

_llm: Optional[BaseChatModel] = None


def get_llm() -> BaseChatModel:
    """Return the cached LLM singleton (built on first call)."""
    global _llm
    if _llm is None:
        _llm = build_llm()
    return _llm


def reset_llm() -> None:
    """Force re-initialisation on next get_llm() call (useful in tests)."""
    global _llm
    _llm = None
