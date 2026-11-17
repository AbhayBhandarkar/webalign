"""Content-addressed disk cache for LLM calls.

Keyed by a hash of (backend, model, messages, generation params) so identical
requests return byte-identical results across reruns and never re-bill an API.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Optional

DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache" / "llm"


class LLMCache:
    def __init__(self, cache_dir: str | os.PathLike | None = None):
        self.dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
        self.dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def key(backend: str, model: str, messages: list[dict], params: dict) -> str:
        payload = {
            "backend": backend,
            "model": model,
            "messages": messages,
            "params": {k: v for k, v in sorted(params.items())},
        }
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _path(self, key: str) -> Path:
        shard = self.dir / key[:2]
        shard.mkdir(exist_ok=True)
        return shard / f"{key}.json"

    def get(self, key: str) -> Optional[dict[str, Any]]:
        p = self._path(key)
        if p.exists():
            try:
                return json.loads(p.read_text())
            except json.JSONDecodeError:
                return None
        return None

    def set(self, key: str, value: dict[str, Any]) -> None:
        tmp = self._path(key).with_suffix(".tmp")
        tmp.write_text(json.dumps(value, ensure_ascii=False))
        tmp.replace(self._path(key))
