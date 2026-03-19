"""Security guardrails for the Analysis Agent pipeline.

These guardrails are enforced centrally in the pipeline runner before invoking MCP tools.
They are intentionally lightweight (no external deps) and Windows-safe.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


class GuardrailViolation(Exception):
    def __init__(self, message: str, *, kind: str = "policy", details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.kind = kind
        self.details = details or {}


_PATH_KEYS_READ = {
    "file_path",
    "dataset_path",
    "csv_path",
    "input_path",
    "path",
}

_PATH_KEYS_WRITE_DIR = {
    "output_dir",
    "out_dir",
    "save_dir",
}

_PATH_KEYS_WRITE_FILE = {
    "output_path",
    "out_path",
    "save_path",
}


def _has_parent_traversal(p: str) -> bool:
    try:
        parts = PurePath(str(p)).parts
        return any(part == ".." for part in parts)
    except Exception:
        return ".." in str(p)


def _is_under_root(path: Path, root: Path) -> bool:
    try:
        path_res = path.resolve(strict=False)
        root_res = root.resolve(strict=False)
        try:
            return path_res.is_relative_to(root_res)  # py3.9+
        except Exception:
            return str(path_res).lower().startswith(str(root_res).lower().rstrip("\\/") + "\\")
    except Exception:
        return False


def validate_and_resolve_path(
    raw_path: str,
    *,
    repo_root: Path,
    allowed_roots: Sequence[Path],
) -> Path:
    """Resolve a user/tool-supplied path and ensure it stays under an allowlisted root.

    Rules:
    - Reject any path containing parent traversal (`..`).
    - Relative paths are interpreted as relative to repo_root.
    - Absolute paths are allowed only if they are within allowed_roots.
    """
    if raw_path is None:
        raise GuardrailViolation("Path is missing", kind="path")

    raw = str(raw_path).strip()
    if not raw:
        raise GuardrailViolation("Path is empty", kind="path")

    if _has_parent_traversal(raw):
        raise GuardrailViolation(f"Path traversal is not allowed: {raw}", kind="path")

    p = Path(raw)
    if not p.is_absolute():
        p = (repo_root / p).resolve(strict=False)
    else:
        p = p.resolve(strict=False)

    for root in allowed_roots:
        if _is_under_root(p, root):
            return p

    allowed = ", ".join(str(r.resolve(strict=False)) for r in allowed_roots)
    raise GuardrailViolation(
        f"Path is outside allowed roots: {p}. Allowed roots: {allowed}",
        kind="path",
        details={"path": str(p), "allowed_roots": [str(r) for r in allowed_roots]},
    )


def score_prompt_injection(text: str) -> Tuple[int, List[str]]:
    """Heuristic risk score for prompt-injection attempts.

    Returns (score, flags). Score is an integer where >= 6 is considered high risk.
    """
    if not text:
        return 0, []

    s = str(text).lower()
    flags: List[str] = []
    score = 0

    def _hit(weight: int, label: str) -> None:
        nonlocal score
        score += weight
        flags.append(label)

    # Core override patterns
    if "ignore previous" in s or "ignore all previous" in s:
        _hit(4, "ignore_previous")
    if "system prompt" in s or "developer message" in s:
        _hit(3, "prompt_leak_attempt")
    if "do not follow" in s and "instructions" in s:
        _hit(3, "override_instructions")

    # Exfiltration / secrets
    if "api key" in s or "secret" in s or "token" in s or "password" in s:
        _hit(2, "secrets_language")

    # File-system targeting hints (Windows + Unix)
    if "c:\\windows" in s or "system32" in s or "/etc/" in s:
        _hit(4, "sensitive_path")
    if "read file" in s or "open file" in s:
        _hit(2, "file_read_intent")
    if "write file" in s or "overwrite" in s or "delete" in s:
        _hit(2, "file_write_intent")

    # Tool manipulation hints
    if "call the tool" in s or "use the tool" in s and "regardless" in s:
        _hit(1, "tool_manipulation")

    # Cap flags length for logging safety
    if len(flags) > 12:
        flags = flags[:12]

    return score, flags


@dataclass(frozen=True)
class ToolCallPolicy:
    """Central tool-call firewall + path/output policy."""

    repo_root: Path
    allowed_roots: Tuple[Path, ...]
    forced_output_dir: Path

    # Generic caps
    max_list_items: int = 50
    max_str_len: int = 4000

    # Common numeric caps
    max_top_n: int = 2000

    def sanitize_args(self, tool_name: str, args: Dict[str, Any], *, tool_obj: Any = None) -> Tuple[Dict[str, Any], List[str]]:
        if not isinstance(args, dict):
            raise GuardrailViolation("Tool args must be a dict", kind="args")

        warnings: List[str] = []
        cleaned: Dict[str, Any] = {}

        # Best-effort: use LangChain args_schema if available to drop unknown keys.
        allowed_keys: Optional[set[str]] = None
        try:
            schema = getattr(tool_obj, "args_schema", None)
            if schema is not None and hasattr(schema, "model_fields"):
                allowed_keys = set(schema.model_fields.keys())
        except Exception:
            allowed_keys = None

        for k, v in args.items():
            key = str(k)
            if allowed_keys is not None and key not in allowed_keys:
                warnings.append(f"Dropped unknown arg '{key}' for tool '{tool_name}'")
                continue

            # Cap lists
            if isinstance(v, list):
                if len(v) > self.max_list_items:
                    warnings.append(f"Capped list arg '{key}' to {self.max_list_items} items")
                    cleaned[key] = v[: self.max_list_items]
                else:
                    cleaned[key] = v
                continue

            # Cap strings
            if isinstance(v, str):
                if len(v) > self.max_str_len:
                    warnings.append(f"Trimmed string arg '{key}' to {self.max_str_len} chars")
                    cleaned[key] = v[: self.max_str_len]
                else:
                    cleaned[key] = v
                continue

            cleaned[key] = v

        # Common numeric caps
        if "top_n" in cleaned:
            try:
                n = int(cleaned["top_n"])
                if n > self.max_top_n:
                    warnings.append(f"Capped top_n from {n} to {self.max_top_n}")
                    cleaned["top_n"] = self.max_top_n
                if n < 1:
                    cleaned["top_n"] = 1
            except Exception:
                warnings.append("Invalid top_n; forcing to safe default 20")
                cleaned["top_n"] = 20

        # Path policy: validate and normalize any path-like args
        for key in list(cleaned.keys()):
            if key in _PATH_KEYS_READ or key in _PATH_KEYS_WRITE_FILE:
                try:
                    resolved = validate_and_resolve_path(
                        str(cleaned[key]),
                        repo_root=self.repo_root,
                        allowed_roots=self.allowed_roots,
                    )
                    cleaned[key] = str(resolved)
                except GuardrailViolation:
                    raise
                except Exception as e:
                    raise GuardrailViolation(f"Invalid path for '{key}': {e}", kind="path")

        # Output dir is forced under reports_path
        for key in list(cleaned.keys()):
            if key in _PATH_KEYS_WRITE_DIR:
                cleaned[key] = str(self.forced_output_dir)
                warnings.append(f"Forced '{key}' to safe output dir")

        return cleaned, warnings


def build_default_policy(*, repo_root: Path, allowed_roots: Iterable[Path], forced_output_dir: Path) -> ToolCallPolicy:
    roots = tuple(Path(r) for r in allowed_roots)
    return ToolCallPolicy(
        repo_root=Path(repo_root),
        allowed_roots=roots,
        forced_output_dir=Path(forced_output_dir),
    )
