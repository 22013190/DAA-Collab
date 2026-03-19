"""Unified state models for the LangGraph-based analysis pipeline."""

from __future__ import annotations

import hashlib
import os
from typing import Annotated, Any, Dict, List, Optional

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, ConfigDict, Field, field_validator


# Expert-level token budget for Qwen3-14B (32k context window)
# Conservative limit to prevent context overflow and excessive LLM calls
DEFAULT_CONTEXT_TOKEN_BUDGET = 8_000  # ~25% of 32k context window
DEFAULT_MESSAGE_WINDOW = 4  # Reduced message history for efficiency
try:
    MAX_PLAN_STEPS = int(os.getenv("ANALYSIS_PIPELINE_STEP_CAP", "6"))
except Exception:
    MAX_PLAN_STEPS = 6  # Limit planning iterations (fallback)

# Event history caps to prevent memory explosion
# Make insight cap configurable and generous by default to allow comprehensive output.
MAX_TOOL_TRANSCRIPT_ENTRIES = 50
try:
    MAX_INSIGHTS = int(os.getenv("ANALYSIS_MAX_INSIGHTS_STATE", "1000"))
except Exception:
    MAX_INSIGHTS = 1000
MAX_ARTIFACT_LOG = 100

# Text truncation caps (defaults preserve prior behavior)
try:
    INSIGHT_TEXT_MAX_CHARS = int(os.getenv("ANALYSIS_INSIGHT_TEXT_MAX_CHARS", "400"))
except Exception:
    INSIGHT_TEXT_MAX_CHARS = 400
try:
    COVERAGE_MISSING_ITEM_MAX_CHARS = int(os.getenv("ANALYSIS_COVERAGE_MISSING_ITEM_MAX_CHARS", "300"))
except Exception:
    COVERAGE_MISSING_ITEM_MAX_CHARS = 300


# ========== LangGraph Reducers ==========

def _replace_latest(_: Optional[str], new_val: Optional[str]) -> Optional[str]:
    """Replace with latest value for scalar channels"""
    return new_val


def _replace_latest_int(_: Optional[int], new_val: Optional[int]) -> int:
    """Replace with latest value for int channels"""
    return new_val if new_val is not None else (_ or 0)


def _replace_latest_bool(_: Optional[bool], new_val: Optional[bool]) -> bool:
    """Replace with latest value for bool channels"""
    return new_val if new_val is not None else (_ or False)


def _merge_token_metrics(
    prev: Optional[Dict[str, int]], new_val: Optional[Dict[str, int]]
) -> Dict[str, int]:
    """Merge token metrics by summing per component"""
    prev = prev or {}
    if not new_val:
        return dict(prev)
    merged: Dict[str, int] = dict(prev)
    for k, v in new_val.items():
        try:
            inc = int(v) if v is not None else 0
        except Exception:
            inc = 0
        merged[k] = merged.get(k, 0) + inc
    return merged


def _replace_latest_any(_: Optional[Any], new_val: Optional[Any]) -> Optional[Any]:
    """Replace with latest value for any type"""
    return new_val


def _replace_latest_dict(_: Optional[Dict], new_val: Optional[Dict]) -> Dict:
    """Replace with latest value for dict channels"""
    return new_val if new_val is not None else (_ or {})


def _merge_unique_str_list(prev: Optional[List[str]], new_val: Optional[List[str]]) -> List[str]:
    """Merge list[str] channels allowing multiple updates per graph step.

    - Preserves original order of first occurrence.
    - Deduplicates values (string identity) to prevent bloat.
    - Safe when LangGraph delivers multiple partial updates in a single step.
    """
    out: List[str] = []
    seen = set()
    for source in (prev, new_val):
        if not source:
            continue
        for item in source:
            if not isinstance(item, str):
                continue
            if item not in seen:
                seen.add(item)
                out.append(item)
    return out


def _merge_tool_transcript(
    prev: Optional[List[Dict[str, Any]]], 
    new_val: Optional[List[Dict[str, Any]]]
) -> List[Dict[str, Any]]:
    """Merge tool execution events with deduplication and capping.
    
    Deduplicates by content hash to avoid logging identical retried calls.
    Preserves chronological order.
    Caps at MAX_TOOL_TRANSCRIPT_ENTRIES to prevent memory explosion.
    """
    def _event_signature(evt: Dict[str, Any]) -> str:
        """Generate stable hash for event deduplication"""
        try:
            # Use tool, args, and output for signature
            key_parts = [
                str(evt.get("tool", "")),
                str(evt.get("args", {})),
                str(evt.get("output", ""))[:500],  # First 500 chars of output
            ]
            return hashlib.sha256("".join(key_parts).encode()).hexdigest()[:16]
        except Exception:
            # Fallback: use full dict str (slower but safe)
            return hashlib.sha256(str(evt).encode()).hexdigest()[:16]
    
    merged: List[Dict[str, Any]] = []
    seen_sigs = set()
    
    for source in (prev, new_val):
        if not source:
            continue
        for evt in source:
            if not isinstance(evt, dict):
                continue
            sig = _event_signature(evt)
            if sig not in seen_sigs:
                seen_sigs.add(sig)
                merged.append(evt)
    
    # Cap length (keep most recent)
    if len(merged) > MAX_TOOL_TRANSCRIPT_ENTRIES:
        merged = merged[-MAX_TOOL_TRANSCRIPT_ENTRIES:]
    
    return merged


def _merge_insights(
    prev: Optional[List[Dict[str, Any]]], 
    new_val: Optional[List[Dict[str, Any]]]
) -> List[Dict[str, Any]]:
    """Merge insights with deduplication by (statement, evidence).
    
    Preserves first occurrence and merges refs lists for duplicates.
    Caps at MAX_INSIGHTS to control memory.
    """
    def _insight_key(ins: Dict[str, Any]) -> tuple:
        """Generate deduplication key"""
        try:
            stmt = str(ins.get("statement", "")).strip()[:200]
            ev = str(ins.get("evidence", "")).strip()[:200]
            return (stmt, ev)
        except Exception:
            return (str(ins), "")
    
    merged_dict: Dict[tuple, Dict[str, Any]] = {}
    
    for source in (prev, new_val):
        if not source:
            continue
        for ins in source:
            if not isinstance(ins, dict):
                continue
            
            key = _insight_key(ins)
            if key not in merged_dict:
                # First occurrence: store it
                merged_dict[key] = dict(ins)
            else:
                # Duplicate: merge refs if present
                existing = merged_dict[key]
                new_refs = ins.get("refs", [])
                if isinstance(new_refs, list):
                    existing_refs = existing.get("refs", [])
                    if not isinstance(existing_refs, list):
                        existing_refs = []
                    # Merge refs (deduplicate)
                    combined_refs = []
                    seen_refs = set()
                    for ref in existing_refs + new_refs:
                        ref_str = str(ref)
                        if ref_str not in seen_refs:
                            seen_refs.add(ref_str)
                            combined_refs.append(ref)
                    existing["refs"] = combined_refs[:20]  # Cap refs per insight
    
    # Convert back to list, preserving insertion order
    merged = list(merged_dict.values())
    
    # Cap total insights
    if len(merged) > MAX_INSIGHTS:
        merged = merged[-MAX_INSIGHTS:]
    
    return merged


def _merge_dict_list(
    prev: Optional[List[Dict[str, Any]]], 
    new_val: Optional[List[Dict[str, Any]]]
) -> List[Dict[str, Any]]:
    """Generic merge for list[dict] with robust deduplication.

    Uses stable JSON serialization (sort_keys=True) for hashing so that logically
    identical dictionaries with different key orders are deduplicated reliably.
    """
    import json

    prev = prev or []
    new_val = new_val or []
    merged: List[Dict[str, Any]] = []
    seen_hashes = set()

    def _stable_hash(item: Dict[str, Any]) -> str:
        try:
            serialized = json.dumps(item, sort_keys=True, default=str)
        except Exception:
            serialized = str(item)
        return hashlib.sha256(serialized.encode()).hexdigest()[:16]

    for source in (prev, new_val):
        for item in source:
            if not isinstance(item, dict):
                continue
            item_hash = _stable_hash(item)
            if item_hash in seen_hashes:
                continue
            seen_hashes.add(item_hash)
            merged.append(item)

    return merged


def _merge_str_list_capped(
    prev: Optional[List[str]], 
    new_val: Optional[List[str]],
    cap: int = 100
) -> List[str]:
    """Merge list[str] with deduplication and capping.
    
    Similar to _merge_unique_str_list but with a length cap.
    """
    out: List[str] = []
    seen = set()
    
    for source in (prev, new_val):
        if not source:
            continue
        for item in source:
            if not isinstance(item, str):
                continue
            if item not in seen:
                seen.add(item)
                out.append(item)
    
    # Cap length (keep most recent)
    if len(out) > cap:
        out = out[-cap:]
    
    return out


# Reducer factories and bound reducers for capped string lists
def make_str_list_capped_reducer(cap: int):
    """Create a reducer that merges list[str] with deduplication and a fixed cap."""
    def _reducer(prev: Optional[List[str]], new_val: Optional[List[str]]):
        return _merge_str_list_capped(prev, new_val, cap)
    return _reducer


# Pre-bound reducers for commonly capped channels
ARTIFACT_LOG_REDUCER = make_str_list_capped_reducer(MAX_ARTIFACT_LOG)
REFINEMENT_REQUEST_REDUCER = make_str_list_capped_reducer(20)
WARNINGS_REDUCER = make_str_list_capped_reducer(50)
ERRORS_REDUCER = make_str_list_capped_reducer(50)


# ========== State Models ==========

class AnalysisPipelineState(BaseModel):
    """
    Expert LangGraph state with Pydantic validation and business logic.
    Optimized for Qwen3-14B with token budget controls.
    
    ALL fields use Annotated reducers to prevent INVALID_CONCURRENT_GRAPH_UPDATE errors.
    List fields now use proper merge reducers instead of _replace_latest_any.
    """
    
    model_config = ConfigDict(
        arbitrary_types_allowed=True,  # Allow BaseMessage types
        validate_assignment=True,      # Validate on field updates
        extra='forbid'                 # Prevent accidental field additions
    )

    # Core workflow state with LangGraph reducers
    messages: Annotated[List[BaseMessage], add_messages] = Field(
        default_factory=list, 
        description="LangChain message history"
    )
    dataset_path: Annotated[Optional[str], _replace_latest] = Field(
        default="", 
        description="Current dataset path under analysis"
    )
    
    # Analysis configuration
    instruction: Annotated[str, _replace_latest] = Field(
        default="",
        description="User-provided analysis instruction"
    )

    # Output mode classification (LLM determined)
    output_mode: Annotated[Optional[str], _replace_latest] = Field(
        default=None,
        description="LLM-classified output mode: visualization_only | insights_only | analysis | mixed"
    )
    visualization_only: Annotated[bool, _replace_latest_bool] = Field(
        default=False,
        description="Simplified flag derived from output_mode for downstream synthesis formatting"
    )
    classification_confidence: Annotated[Optional[float], _replace_latest_any] = Field(
        default=None,
        description="Reported confidence (0–1) from planner output mode classification"
    )
    classification_rationale: Annotated[Optional[str], _replace_latest] = Field(
        default=None,
        description="Short rationale for chosen output mode"
    )
    
    # Workflow tracking
    context_summary: Annotated[str, _replace_latest] = Field(
        default="",
        description="Condensed context for LLM prompts"
    )
    
    # plan_steps is single-writer (planner). Use replace to allow clearing.
    plan_steps: Annotated[List[Dict[str, Any]], _replace_latest_any] = Field(
        default_factory=list,
        description="Structured plan steps: list of {tool, args, why} objects"
    )
    
    next_step_index: Annotated[int, _replace_latest_int] = Field(
        default=0,
        ge=0,
        description="Index of next plan_step to execute"
    )

    # Planner/reflect loop bookkeeping (must be explicit fields; extra='forbid')
    plan_hash: Annotated[Optional[str], _replace_latest] = Field(
        default=None,
        description="Hash of current plan_steps for replan no-change detection"
    )
    replan_no_change: Annotated[bool, _replace_latest_bool] = Field(
        default=False,
        description="Internal: last replanning attempt produced identical plan"
    )
    plan_retries: Annotated[int, _replace_latest_int] = Field(
        default=0,
        ge=0,
        description="Internal: consecutive replan-without-change attempts"
    )
    finalize: Annotated[bool, _replace_latest_bool] = Field(
        default=False,
        description="Internal: force finalization to synthesis"
    )
    stop_after_first_synthesis: Annotated[bool, _replace_latest_bool] = Field(
        default=False,
        description="Internal: stop after the first synthesis/persist"
    )

    # Human-in-the-loop (HITL) plan review
    hitl_plan_reviewed: Annotated[bool, _replace_latest_bool] = Field(
        default=False,
        description="HITL: whether the current plan_steps has been reviewed/approved"
    )
    hitl_plan_decision: Annotated[Optional[str], _replace_latest] = Field(
        default=None,
        description="HITL: last plan review decision (approve|revise|abort)"
    )
    hitl_plan_feedback: Annotated[Optional[str], _replace_latest] = Field(
        default=None,
        description="HITL: optional feedback provided during plan review"
    )
    
    # FIXED: tool_transcript uses custom merge reducer
    tool_transcript: Annotated[List[Dict[str, Any]], _merge_tool_transcript] = Field(
        default_factory=list, 
        description="Tool execution history with deduplication and capping"
    )
    
    # FIXED: artifact_log already had _merge_unique_str_list, now capped
    artifact_log: Annotated[List[str], ARTIFACT_LOG_REDUCER] = Field(
        default_factory=list,
        description="Generated artifact paths (merged, unique, capped)"
    )
    
    should_synthesize: Annotated[bool, _replace_latest_bool] = Field(
        default=False,
        description="Flag to indicate when ready for synthesis"
    )
    
    # Memory and context
    historical_snippet: Annotated[Optional[str], _replace_latest] = Field(
        default=None, 
        description="Relevant historical context"
    )
    token_metrics: Annotated[Dict[str, int], _merge_token_metrics] = Field(
        default_factory=dict,
        description="Token usage tracking per component"
    )

    # Lightweight counters for run telemetry (merged by summation)
    telemetry: Annotated[Dict[str, int], _merge_token_metrics] = Field(
        default_factory=dict,
        description="Run telemetry counters (e.g., tool_calls, tool_errors, planner_calls, refinement_rounds)"
    )

    # Security/operational warnings (non-fatal) and errors (fatal or policy violations)
    warnings: Annotated[List[str], WARNINGS_REDUCER] = Field(
        default_factory=list,
        description="Non-fatal warnings (merged, capped)"
    )
    errors: Annotated[List[str], ERRORS_REDUCER] = Field(
        default_factory=list,
        description="Errors encountered (merged, capped)"
    )
    
    # FIXED: insights uses custom merge reducer
    insights: Annotated[List[Dict[str, Any]], _merge_insights] = Field(
        default_factory=list, 
        description="Structured insights with deduplication by (statement, evidence)"
    )
    
    coverage: Annotated[Dict[str, Any], _replace_latest_dict] = Field(
        default_factory=lambda: {"answers_query": False, "missing_info": []}, 
        description="Coverage assessment: {answers_query: bool, missing_info: list[str]}"
    )
    
    # Synthesis outputs
    report_text: Annotated[str, _replace_latest] = Field(
        default="",
        description="Final synthesized report text"
    )
    summary: Annotated[str, _replace_latest] = Field(
        default="",
        description="Concise distilled summary (1-2 sentences)"
    )
    final_artifacts: Annotated[Dict[str, Any], _replace_latest_dict] = Field(
        default_factory=dict,
        description="Structured artifact index produced at synthesis phase"
    )
    
    # FIXED: refinement_request uses capped str list merge
    refinement_request: Annotated[List[str], REFINEMENT_REQUEST_REDUCER] = Field(
        default_factory=list,
        description="Missing information items driving refinement (capped at 20)"
    )

    refinement_round: Annotated[int, _replace_latest_int] = Field(
        default=0,
        ge=0,
        description="Number of refinement loops executed"
    )

    reinterpret_attempts: Annotated[int, _replace_latest_int] = Field(
        default=0,
        ge=0,
        description=(
            "Internal: number of times reflect routed back to interpret_results to recover missed insights "
            "from existing tool outputs (loop-capped)."
        ),
    )
    
    preprocess_profile: Annotated[Optional[Dict[str, Any]], _replace_latest_any] = Field(
        default=None,
        description="Dataset column profile for tool argument inference"
    )

    capability_gap: Annotated[Optional[Dict[str, Any]], _replace_latest_any] = Field(
        default=None,
        description=(
            "Optional structured diagnosis explaining why the current toolset cannot fully answer the instruction, "
            "including proposed additional tool capability specifications."
        ),
    )
    
    # LLM-generated intelligent column mappings (semantic purpose → actual column names)
    column_mappings: Annotated[Optional[Dict[str, List[str]]], _replace_latest_any] = Field(
        default=None,
        description="Intelligent column mappings from LLM analysis (e.g., datetime_column → [crash_date, xyzabc])"
    )

    @field_validator('plan_steps')
    @classmethod
    def limit_plan_steps(cls, v):
        """Enforce plan step limit to control LLM calls"""
        if len(v) > MAX_PLAN_STEPS:
            return v[-MAX_PLAN_STEPS:]
        return v

    # Business logic methods
    def log_tool_event(self, event: Dict[str, Any]) -> None:
        """Record tool execution event"""
        if event:
            self.tool_transcript.append(event)
            # Auto-register artifact if present in event
            art = event.get("artifact")
            if isinstance(art, str) and art:
                self.register_artifact(art)

    def register_artifact(self, artifact_path: str) -> None:
        """Track generated artifacts (deduplication handled by reducer)"""
        if artifact_path:
            self.artifact_log.append(artifact_path)

    def update_context_summary(self, summary: str) -> None:
        """Update context summary for LLM prompts"""
        self.context_summary = summary.strip()

    def record_token_usage(self, component: str, tokens: int) -> None:
        """Track token usage per component"""
        if component and tokens >= 0:
            current = self.token_metrics.get(component, 0)
            self.token_metrics[component] = current + tokens

    def total_tokens(self) -> int:
        """Get total token consumption"""
        return sum(self.token_metrics.values())

    def within_token_budget(self, budget: int = DEFAULT_CONTEXT_TOKEN_BUDGET) -> bool:
        """Check if within token budget"""
        return self.total_tokens() <= budget

    def trim_message_window(self, max_messages: int = DEFAULT_MESSAGE_WINDOW) -> None:
        """Trim message history for context control"""
        if max_messages <= 0:
            return
        if len(self.messages) > max_messages:
            self.messages = self.messages[-max_messages:]

    def is_ready_for_synthesis(self) -> bool:
        """Check if enough analysis has been done for synthesis"""
        return (len(self.tool_transcript) > 0 or 
                len(self.insights) > 0 or 
                len(self.plan_steps) >= MAX_PLAN_STEPS)

    def advance_step_index(self) -> None:
        """Increment next_step_index safely."""
        if self.next_step_index < len(self.plan_steps):
            self.next_step_index += 1

    def record_insight(self, statement: str, evidence: str, refs: Optional[List[str]] = None) -> None:
        """Append a structured insight item (deduplication handled by reducer)."""
        statement = (statement or "").strip()
        evidence = (evidence or "").strip()
        if not statement:
            return
        self.insights.append({
            "statement": statement[:INSIGHT_TEXT_MAX_CHARS],
            "evidence": evidence[:INSIGHT_TEXT_MAX_CHARS],
            "refs": (refs or [])[:20]
        })

    def update_coverage(self, answers_query: Optional[bool] = None, missing_info: Optional[List[str]] = None) -> None:
        """Update coverage dict with validation."""
        cov = self.coverage or {"answers_query": False, "missing_info": []}
        if answers_query is not None:
            cov["answers_query"] = bool(answers_query)
        if missing_info is not None:
            # Normalize and de-duplicate missing info items
            cleaned = []
            for item in missing_info:
                if isinstance(item, str):
                    it = item.strip()
                    if it and it not in cleaned:
                        cleaned.append(it[:COVERAGE_MISSING_ITEM_MAX_CHARS])
            cov["missing_info"] = cleaned
        self.coverage = cov

    def set_report(self, report: str, summary: Optional[str] = None, artifacts: Optional[Dict[str, Any]] = None) -> None:
        """Populate synthesis fields in state."""
        self.report_text = (report or "").strip()
        if summary is not None:
            self.summary = summary.strip()
        if artifacts:
            self.final_artifacts.update(artifacts)

    def should_continue_analysis(self) -> bool:
        """Determine if more analysis steps are needed"""
        return (
            self.within_token_budget()
            and len(self.plan_steps) < MAX_PLAN_STEPS
            and not self.is_ready_for_synthesis()
        )


class PipelineOutputState(BaseModel):
    """Output envelope for the analysis pipeline with LangGraph-safe reducers.

    All list fields now use proper merge reducers to prevent INVALID_CONCURRENT_GRAPH_UPDATE.
    """

    model_config = ConfigDict(
        validate_assignment=True,
        extra='forbid'
    )

    status: Annotated[str, _replace_latest] = Field(
        default="ok",
        pattern="^(ok|error)$",
        description="Run status: 'ok' or 'error'"
    )
    title: Annotated[Optional[str], _replace_latest] = Field(
        default=None,
        description="Optional short title"
    )

    report_text: Annotated[str, _replace_latest] = Field(
        default="",
        description="Final analysis report"
    )
    summary: Annotated[Optional[str], _replace_latest] = Field(
        default=None,
        description="Optional concise summary"
    )

    artifacts: Annotated[Dict[str, Any], _replace_latest_dict] = Field(
        default_factory=dict,
        description="Produced artifacts"
    )
    dataset_path: Annotated[Optional[str], _replace_latest] = Field(
        default=None,
        description="Primary dataset path"
    )

    # FIXED: steps and tool_summaries use merge reducers
    steps: Annotated[List[Dict[str, Any]], _merge_dict_list] = Field(
        default_factory=list,
        description="Executed plan steps (merged)"
    )
    tool_summaries: Annotated[List[Dict[str, Any]], _merge_dict_list] = Field(
        default_factory=list,
        description="Key tool call summaries (merged)"
    )

    token_metrics: Annotated[Dict[str, int], _merge_token_metrics] = Field(
        default_factory=dict,
        description="Token usage by component"
    )
    
    # FIXED: warnings and errors use capped merge
    warnings: Annotated[List[str], WARNINGS_REDUCER] = Field(
        default_factory=list,
        description="Non-fatal warnings (merged, capped)"
    )
    errors: Annotated[List[str], ERRORS_REDUCER] = Field(
        default_factory=list,
        description="Errors encountered (merged, capped)"
    )

    # Business logic methods
    def add_artifact(self, name: str, path: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Register an artifact with optional metadata"""
        if name and path:
            self.artifacts[name] = {
                "path": path,
                "metadata": metadata or {}
            }

    def add_warning(self, warning: str) -> None:
        """Add a non-fatal warning (deduplication handled by reducer)"""
        if warning and warning.strip():
            self.warnings.append(warning.strip())

    def add_error(self, error: str) -> None:
        """Add an error and set status to error"""
        if error and error.strip():
            self.errors.append(error.strip())
            self.status = "error"

    def record_step(self, step: Dict[str, Any]) -> None:
        """Record a plan step execution"""
        if step and isinstance(step, dict):
            self.steps.append(step)

    def add_tool_summary(self, tool_name: str, result_summary: str) -> None:
        """Add a tool execution summary"""
        if tool_name and result_summary:
            self.tool_summaries.append({
                "tool": tool_name,
                "summary": result_summary.strip()
            })

    def merge_token_metrics(self, metrics: Dict[str, int]) -> None:
        """Merge additional token metrics"""
        for component, tokens in metrics.items():
            if component and tokens >= 0:
                current = self.token_metrics.get(component, 0)
                self.token_metrics[component] = current + tokens

    def total_tokens(self) -> int:
        """Get total token consumption"""
        return sum(self.token_metrics.values())

    def is_successful(self) -> bool:
        """Check if execution was successful"""
        return self.status == "ok" and len(self.errors) == 0

    def has_content(self) -> bool:
        """Check if output has meaningful content"""
        return (len(self.report_text.strip()) > 0 or 
                len(self.artifacts) > 0 or 
                self.summary is not None)

    def to_a2a_response(self) -> Dict[str, Any]:
        """Convert to A2A protocol response format"""
        return {
            "status": self.status,
            "title": self.title,
            "report": self.report_text,
            "summary": self.summary,
            "artifacts": self.artifacts,
            "metadata": {
                "steps": self.steps,
                "tools": self.tool_summaries,
                "tokens": self.token_metrics,
                "warnings": self.warnings,
                "errors": self.errors
            }
        }


# Backwards-compatible aliases
PipelineBaseState = AnalysisPipelineState
PipelineOverallState = AnalysisPipelineState