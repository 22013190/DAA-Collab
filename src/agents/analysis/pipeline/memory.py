"""Persistence utilities for analysis run metadata and long-term memory."""

from __future__ import annotations

import asyncio
import os
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from pydantic import Field
from pydantic.dataclasses import dataclass

from .state import AnalysisPipelineState
from ..analysis_config import AgentConfig

# Align metadata storage with AgentConfig.reports_path under the analysis agent directory.
# Fall back to a path relative to this file if config resolution fails.
try:
    DEFAULT_METADATA_DIR = AgentConfig().reports_path / "metadata"
except Exception:
    DEFAULT_METADATA_DIR = Path(__file__).resolve().parent.parent / "reportDemo" / "metadata"
DEFAULT_HISTORY_LIMIT = 10
ASYNC_WRITE_TIMEOUT = 5.0


@dataclass
class RunMetadata:
    """Structured summary of a single analysis run."""

    dataset_path: Optional[str] = Field(default=None)
    instruction: str = Field(default="")
    findings_summary: str = Field(default="")
    plan_summary: str = Field(default="")
    artifacts: List[str] = Field(default_factory=list)
    issues: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    telemetry: Dict[str, Any] = Field(default_factory=dict)
    hitl_plan_review: Optional[Dict[str, Any]] = Field(default=None)
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict:
        return {
            "dataset_path": self.dataset_path,
            "instruction": self.instruction,
            "findings_summary": self.findings_summary,
            "plan_summary": self.plan_summary,
            "artifacts": list(self.artifacts),
            "issues": list(self.issues),
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "telemetry": dict(self.telemetry or {}),
            "hitl_plan_review": self.hitl_plan_review,
            "timestamp": self.timestamp,
        }


class MemoryStore:
    """File-backed metadata storage with simple retrieval and pruning."""

    def __init__(
        self,
        storage_dir: Path = DEFAULT_METADATA_DIR,
        history_limit: int = DEFAULT_HISTORY_LIMIT,
    ) -> None:
        self._storage_dir = storage_dir
        self._history_limit = history_limit
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        # One-time migration: if legacy repo-root metadata exists, copy into the configured storage_dir
        try:
            legacy_dir = Path("src/agents/analysis/reportDemo/metadata")
            if legacy_dir.exists():
                # Only migrate if target is effectively empty (no *.json files)
                has_target_files = any(self._storage_dir.glob("*.json"))
                if not has_target_files:
                    for src_file in legacy_dir.glob("*.json"):
                        try:
                            dst = self._storage_dir / src_file.name
                            if not dst.exists():
                                dst.write_bytes(src_file.read_bytes())
                        except Exception:
                            continue
        except Exception:
            pass

    def _metadata_file(self, dataset_key: str) -> Path:
        return self._storage_dir / f"{dataset_key}.json"

    def _read_records(self, dataset_key: str) -> List[Dict]:
        file_path = self._metadata_file(dataset_key)
        if not file_path.exists():
            return []
        try:
            with file_path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            return data if isinstance(data, list) else []
        except Exception:
            return []

    def _write_records(self, dataset_key: str, records: List[Dict]) -> None:
        file_path = self._metadata_file(dataset_key)
        with file_path.open("w", encoding="utf-8") as fh:
            json.dump(records[: self._history_limit], fh, indent=2)

    def load_history(self, dataset_path: Optional[str]) -> List[Dict]:
        if not dataset_path:
            return []
        dataset_key = self._normalise_dataset_key(dataset_path)
        return self._read_records(dataset_key)

    def schedule_append(self, metadata: RunMetadata) -> None:
        dataset_key = self._normalise_dataset_key(metadata.dataset_path or "generic")
        records = [metadata.to_dict()] + self._read_records(dataset_key)

        async def _async_write() -> None:
            await asyncio.to_thread(self._write_records, dataset_key, records)

        # Only schedule an async task when an event loop is running; otherwise write synchronously.
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is None:
            self._write_records(dataset_key, records)
            return

        loop.create_task(asyncio.wait_for(_async_write(), timeout=ASYNC_WRITE_TIMEOUT))

    def prune(self, dataset_path: Optional[str] = None) -> None:
        targets: Iterable[str]
        if dataset_path:
            targets = [self._normalise_dataset_key(dataset_path)]
        else:
            targets = [p.stem for p in self._storage_dir.glob("*.json")]
        for key in targets:
            records = self._read_records(key)
            if len(records) > self._history_limit:
                self._write_records(key, records)

    @staticmethod
    def _normalise_dataset_key(dataset_path: str) -> str:
        return Path(dataset_path).stem.replace(" ", "_")


def build_run_metadata(state: AnalysisPipelineState, report: str) -> RunMetadata:
    """Build run metadata with relative paths for portability.

    - dataset_path: stored relative to the project (best effort), else basename
    - artifacts: stored relative to AgentConfig().reports_path when possible, else basename
    """
    cfg = AgentConfig()
    reports_base = cfg.reports_path

    # Compute a best-effort repository root: prefer four levels up from analysis_agent_dir (repo root),
    # fall back to three levels up (agent_system root) if needed.
    try:
        analysis_dir = cfg.analysis_agent_dir
        candidates = []
        try:
            candidates.append(analysis_dir.parents[3])  # repo root (Multi-Agent)
        except Exception:
            pass
        try:
            candidates.append(analysis_dir.parents[2])  # agent_system root
        except Exception:
            pass
        repo_root = next((c for c in candidates if isinstance(c, Path) and c.exists()), analysis_dir)
    except Exception:
        repo_root = reports_base if isinstance(reports_base, Path) else Path('.')

    def _rel_or_name(p: str, base: Path) -> str:
        try:
            if not p:
                return ""
            # Use os.path.relpath to safely compute relative even if outside subtree
            rel = os.path.relpath(p, start=str(base))
            # Normalise to POSIX style for markdown portability
            return rel.replace('\\', '/')
        except Exception:
            try:
                return Path(p).name
            except Exception:
                return str(p)

    # Dataset path: prefer relative to repo_root; fallback to basename
    dataset_rel = _rel_or_name(state.dataset_path or "", repo_root)

    # Artifacts: prefer relative to reports_base; fallback to basename
    try:
        artifacts_rel = []
        for a in (state.artifact_log or []):
            artifacts_rel.append(_rel_or_name(str(a), reports_base))
    except Exception:
        artifacts_rel = list(state.artifact_log or [])

    def _clip_value(val: Any, *, max_str: int = 240, max_list: int = 25, max_depth: int = 3) -> Any:
        if max_depth <= 0:
            return None
        if val is None:
            return None
        if isinstance(val, (int, float, bool)):
            return val
        if isinstance(val, str):
            s = val.strip()
            return s if len(s) <= max_str else (s[: max_str - 3] + "...")
        if isinstance(val, dict):
            out: Dict[str, Any] = {}
            for k, v in list(val.items())[:50]:
                key = str(k)
                out[key] = _clip_value(v, max_str=max_str, max_list=max_list, max_depth=max_depth - 1)
            return out
        if isinstance(val, (list, tuple)):
            clipped = []
            for item in list(val)[:max_list]:
                clipped.append(_clip_value(item, max_str=max_str, max_list=max_list, max_depth=max_depth - 1))
            return clipped
        try:
            return _clip_value(str(val), max_str=max_str, max_list=max_list, max_depth=max_depth - 1)
        except Exception:
            return None

    # HITL plan review (Phase 4): persist the reviewed plan + user decision/feedback
    hitl_plan_review: Optional[Dict[str, Any]] = None
    try:
        decision = getattr(state, "hitl_plan_decision", None)
        feedback = getattr(state, "hitl_plan_feedback", None)
        reviewed = bool(getattr(state, "hitl_plan_reviewed", False))
        plan_hash = getattr(state, "plan_hash", None)
        plan_steps = getattr(state, "plan_steps", None)

        if decision or feedback or reviewed:
            hitl_plan_review = {
                "decision": str(decision) if decision is not None else None,
                "feedback": _clip_value(feedback, max_str=600, max_list=25, max_depth=2) if feedback else None,
                "reviewed": reviewed,
                "plan_hash": str(plan_hash) if plan_hash else None,
                "plan_steps": _clip_value(plan_steps or [], max_str=240, max_list=10, max_depth=4),
                "reviewed_at": datetime.now(timezone.utc).isoformat(),
            }
    except Exception:
        hitl_plan_review = None

    return RunMetadata(
        dataset_path=dataset_rel or (Path(state.dataset_path).name if state.dataset_path else None),
        instruction=state.instruction,
        findings_summary=report,
        # Use new structured plan_steps (tool + why). Include up to last 5 reasons.
        plan_summary="\n".join(
            f"{step.get('tool')}: {step.get('why','')}" for step in state.plan_steps[-5:]
        ),
        artifacts=artifacts_rel,
        issues=[entry.get("output") for entry in state.tool_transcript if entry.get("status") == "error"],
        warnings=list(getattr(state, "warnings", []) or []),
        errors=list(getattr(state, "errors", []) or []),
        telemetry=_clip_value(getattr(state, "telemetry", {}) or {}, max_str=240, max_list=50, max_depth=3),
        hitl_plan_review=hitl_plan_review,
    )
