"""Minimal OpenAI-compatible chat client used by WebAlign and the baselines.

Every model is reached through the OpenAI chat-completions protocol, so a local
vLLM server and any hosted OpenAI-compatible endpoint are handled the same way.
Backends and models are declared in ``configs/models.yaml``. Responses are cached
on disk, keyed by the full request, so reruns are deterministic and free.
"""
from __future__ import annotations

import os
import re
import time
from pathlib import Path

import yaml
from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI, RateLimitError

from .cache import LLMCache

REPO = Path(__file__).resolve().parent.parent
_THINK_RE = re.compile(r"<think>.*?</think>", re.S)


class LLMClient:
    def __init__(self, config_path: str | os.PathLike | None = None,
                 cache_dir: str | os.PathLike | None = None, use_cache: bool = True):
        path = Path(config_path or os.environ.get("WEBALIGN_CONFIG", REPO / "configs" / "models.yaml"))
        cfg = yaml.safe_load(path.read_text())
        self.backends = cfg["backends"]
        self.models = {**cfg.get("roles", {}), **{a["name"]: a for a in cfg.get("agents", [])}}
        self.cache = LLMCache(cache_dir) if use_cache else None
        self._clients: dict[str, OpenAI] = {}

    def _client(self, backend: str) -> OpenAI:
        if backend not in self._clients:
            b = self.backends[backend]
            base_url = b.get("base_url") or os.environ.get(b.get("base_url_env", ""), "")
            api_key = b.get("api_key") or os.environ.get(b.get("api_key_env", ""), "")
            if not base_url:
                raise RuntimeError(f"backend '{backend}': set {b.get('base_url_env')}")
            self._clients[backend] = OpenAI(base_url=base_url, api_key=api_key or "EMPTY",
                                            timeout=300.0)
        return self._clients[backend]

    def _params(self, spec: dict, params: dict) -> dict:
        b = self.backends[spec["backend"]]
        p = dict(params)
        budget = int(p.pop("max_tokens", 0) or 0)
        if spec.get("reasoning"):
            # Reasoning models spend tokens before answering and reject a custom temperature.
            p.pop("temperature", None)
            budget = max(budget, b.get("min_tokens_reasoning", 6000))
        else:
            p.setdefault("temperature", 0.0)
            budget = max(budget, b.get("min_tokens", 512))
        p[b.get("token_param", "max_tokens")] = budget
        return p

    def chat(self, name: str, messages: list[dict], **params) -> str:
        """Send one chat request to the model registered under ``name``."""
        if name not in self.models:
            raise KeyError(f"unknown model '{name}'; add it to configs/models.yaml")
        spec = self.models[name]
        req = self._params(spec, params)
        key = None
        if self.cache is not None:
            key = self.cache.key(spec["backend"], spec["model"], messages, req)
            hit = self.cache.get(key)
            if hit is not None:
                return hit["content"]
        content = self._call(spec["backend"], spec["model"], messages, req)
        if key is not None:
            self.cache.set(key, {"model": spec["model"], "content": content})
        return content

    def _call(self, backend: str, model: str, messages: list[dict], req: dict,
              attempts: int = 6) -> str:
        delay, last = 2.0, None
        for _ in range(attempts):
            try:
                resp = self._client(backend).chat.completions.create(
                    model=model, messages=messages, **req)
                if resp.choices:
                    msg = resp.choices[0].message
                    text = msg.content or getattr(msg, "reasoning", None) or ""
                    return _THINK_RE.sub("", text).strip()
                last = RuntimeError("response contained no choices")
            except (RateLimitError, APIStatusError) as e:
                last = e
                # Some servers only accept one of the two token-limit parameter names.
                if "max_completion_tokens" in str(e) and "max_completion_tokens" in req:
                    req["max_tokens"] = req.pop("max_completion_tokens")
                    continue
                if getattr(e, "status_code", None) not in (408, 429, 500, 502, 503, 504):
                    raise
            except (APIConnectionError, APITimeoutError) as e:
                last = e
            time.sleep(delay)
            delay = min(delay * 2, 30)
        raise RuntimeError(f"{model}: request failed after {attempts} attempts: {last}")


_default: LLMClient | None = None


def get_client(**kw) -> LLMClient:
    global _default
    if _default is None:
        _default = LLMClient(**kw)
    return _default
