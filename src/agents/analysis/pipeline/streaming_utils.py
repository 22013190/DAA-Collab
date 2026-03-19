"""Utilities for Analysis Agent A2A streaming.

These helpers are intentionally pure / transport-layer focused:
- No LangGraph state mutation
- No A2A SDK dependencies

They exist to make the service-layer streaming loop testable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


_THINK_RE = re.compile(r"<think>[\s\S]*?</think>", re.IGNORECASE)


def strip_think_blocks(text: str) -> str:
    """Remove Qwen-style <think>...</think> blocks.

    This mirrors the intent of the pipeline runner's internal hygiene but is kept
    here so the A2A streaming layer can sanitize incremental chunks.
    """
    if not text:
        return ""
    try:
        return _THINK_RE.sub("", text).strip()
    except Exception:
        return text


def ascii_sanitize(text: str, *, keep_newlines: bool = True) -> str:
    """Best-effort ASCII sanitization for Windows cp1252 consoles.

    - Drops non-ASCII characters.
    - Optionally normalizes newlines.
    """
    if not text:
        return ""
    try:
        cleaned = text.encode("ascii", errors="ignore").decode("ascii", errors="ignore")
        if keep_newlines:
            return cleaned
        return cleaned.replace("\r", " ").replace("\n", " ")
    except Exception:
        return text


@dataclass
class StatusAccumulator:
    """Accumulate streaming text with a hard max size.

    Contract-A snapshot updates want the full accumulated text so far.
    To prevent unbounded growth, keep only the most recent `max_chars`.
    """

    max_chars: int = 80_000
    _buf: str = ""

    def append(self, text: str) -> str:
        if not text:
            return self._buf
        self._buf += text
        if self.max_chars > 0 and len(self._buf) > self.max_chars:
            self._buf = self._buf[-self.max_chars :]
        return self._buf

    def get(self) -> str:
        return self._buf

    def __len__(self) -> int:
        return len(self._buf)


@dataclass
class StreamThrottle:
    """Throttle emissions to avoid flooding A2A with per-token updates."""

    min_interval_s: float = 0.35
    min_chars_delta: int = 120
    _last_emit_time: Optional[float] = None
    _last_emit_len: int = 0

    def should_emit(self, *, now_s: float, current_len: int, force: bool = False) -> bool:
        if force:
            self._last_emit_time = now_s
            self._last_emit_len = current_len
            return True

        if self._last_emit_time is None:
            # First emission: allow as soon as there's something to show.
            if current_len <= 0:
                return False
            self._last_emit_time = now_s
            self._last_emit_len = current_len
            return True

        if (now_s - self._last_emit_time) >= self.min_interval_s:
            self._last_emit_time = now_s
            self._last_emit_len = current_len
            return True

        if (current_len - self._last_emit_len) >= self.min_chars_delta:
            self._last_emit_time = now_s
            self._last_emit_len = current_len
            return True

        return False
