"""LangGraph-based analysis pipeline orchestration."""

from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime
import hashlib
import inspect
from pathlib import Path
from typing import Any, Dict, Literal, Optional, List
import difflib
import shutil

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.config import get_stream_writer
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.types import Command, interrupt
from typing import AsyncIterator, Iterable, Tuple

from agents.utils import llm_with_fallbacks
from mcp_servers.config import analysis_server_config, data_process_server_config
from packages.simple_py_logger.src.logger import Logger
from agents.analysis.analysis_config import AgentConfig
from pathlib import Path
from datetime import datetime

from .memory import MemoryStore, build_run_metadata
from .security_guards import GuardrailViolation, build_default_policy, score_prompt_injection
from .prompts import (
    build_planner_prompt,
    build_reflection_prompt,
    build_synthesis_prompt,
    build_interpretation_prompt,
)
from .state import AnalysisPipelineState, PipelineOutputState
# NOTE: Removed utils.args imports - normalization is anti-pattern for LLM intelligence

# ------------------------------------------------------------------
# DEBUG PRINT TOGGLE
# Set ANALYSIS_DEBUG_PRINTS=1 in environment to enable console prints.
# All debug prints are marked with 'DEBUG_PRINT' for easy removal.
# ------------------------------------------------------------------
DEBUG_PRINTS = str(os.getenv("ANALYSIS_DEBUG_PRINTS", "0")).lower() in ("1", "true", "yes")
# Deprecated: STRICT_EXECUTE_ALL previously forced execution of all plan steps.
# The pipeline now naturally executes sequential steps until exhaustion or reflection-driven synthesis.
# Removing the toggle prevents confusion; behavior is unchanged (always executes planned steps).


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


# Text truncation caps (defaults preserve prior behavior)
INSIGHT_TEXT_MAX_CHARS = _int_env("ANALYSIS_INSIGHT_TEXT_MAX_CHARS", 400)
LLM_CONTEXT_SNIPPET_MAX_CHARS = _int_env("ANALYSIS_LLM_CONTEXT_SNIPPET_MAX_CHARS", 400)
TOOL_OUTPUT_MAX_CHARS = _int_env("ANALYSIS_TOOL_OUTPUT_MAX_CHARS", 500)


def _strip_think_blocks(text: str) -> str:
    """Remove Qwen-style <think>...</think> blocks to avoid polluting history.

    Keeps only the final visible content; use when storing AIMessage content into state.
    """
    try:
        return re.sub(r"<think>[\s\S]*?</think>", "", text).strip()
    except Exception:
        return text


class ModularAnalysisPipelineAgent:
    @staticmethod
    def _iter_a2a_data_parts(parts):
        """Compatibility shim for A2A SDK variants.

        Some a2a versions do not expose `a2a.utils.message.get_data_parts`.
        This helper yields dict-like payload objects from `message.parts`.
        """
        try:
            from a2a.utils.message import get_data_parts  # type: ignore

            for obj in get_data_parts(parts):
                yield obj
            return
        except Exception:
            pass

        for part in (parts or []):
            if part is None:
                continue
            if isinstance(part, dict):
                if isinstance(part.get("data"), dict):
                    yield part.get("data")
                else:
                    yield part
                continue
            data = getattr(part, "data", None)
            if isinstance(data, dict):
                yield data
                continue
            if hasattr(part, "model_dump"):
                try:
                    dumped = part.model_dump()
                    if isinstance(dumped, dict) and isinstance(dumped.get("data"), dict):
                        yield dumped.get("data")
                    elif isinstance(dumped, dict):
                        yield dumped
                except Exception:
                    pass
                continue
            if hasattr(part, "dict"):
                try:
                    dumped = part.dict()
                    if isinstance(dumped, dict) and isinstance(dumped.get("data"), dict):
                        yield dumped.get("data")
                    elif isinstance(dumped, dict):
                        yield dumped
                except Exception:
                    pass
    """LangGraph wrapper that will coordinate planner, tools, and synthesis."""

    def __init__(
        self,
        step_cap: int = 6,
        memory_store: Optional[MemoryStore] = None,
        config: Optional[AgentConfig] = None,
    ) -> None:
        self._logger = Logger("ModularAnalysisPipeline").get_current_logger()
        self._graph: Optional[StateGraph] = None
        self.agent: Optional[CompiledStateGraph] = None
        self._step_cap = step_cap
        self._memory_store = memory_store or MemoryStore()
        self._config = config
        
        # Use langchain_mcp_adapters for proper LangChain tool compatibility
        self._mcp_client: Optional[MultiServerMCPClient] = None
        self._langchain_tools: List = []
        self._react_agent = None
        
        # Tool management
        self._tool_descriptions: str = ""
        self._tool_map = {}
        self._tool_desc_map = {}
        # DEBUG_PRINT: cache flag
        self._debug = DEBUG_PRINTS
        # Idempotency run cache (signature -> {ts, response})
        self._run_cache = {}
        # LLM handles (initialized in _connect_mcp)
        self._llm = None
        self._aux_llm = None
        # Per-invocation run id for tracing (set in receive_a2a_message)
        self._current_run_id: Optional[str] = None
        # Run-level guard to prevent duplicate report writes within a single invocation
        self._reports_written = set()
        # DataFrame cache: {dataset_path: (df, timestamp)}
        self._dataframe_cache = {}
        self._cache_ttl = 300  # 5 minutes TTL for cached DataFrames
        # -------------------------------------------------------------
        # SINGLE RUN GUARD
        # Prevents multiple full synthesis passes within a single invocation.
        # Enabled by default (ANALYSIS_SINGLE_RUN_GUARD=1). After first successful
        # synthesis the guard is latched and subsequent node executions will
        # short-circuit to a cached final output.
        # -------------------------------------------------------------
        try:
            self._single_run_guard_enabled = str(os.getenv("ANALYSIS_SINGLE_RUN_GUARD", "1")).lower() in ("1", "true", "yes")
        except Exception:
            self._single_run_guard_enabled = True
        self._single_run_completed: bool = False
        self._cached_final_output: Optional[PipelineOutputState] = None

        # Global safety cap to prevent runaway executions across replan cycles
        try:
            self._max_total_steps = int(os.getenv("ANALYSIS_MAX_TOTAL_STEPS", "60"))
        except Exception:
            self._max_total_steps = 60
        self._steps_executed = 0

        # Best-known synthesis cache for resilience: if final pass fails, reuse last good
        self._last_good_report_text: Optional[str] = None

        # Best-known structured JSON cache for resilience
        self._last_good_structured_json: Optional[str] = None

        # Traceability toggle: enable detailed state/tool I/O logging when set
        try:
            self._trace_enabled = str(os.getenv("ANALYSIS_TRACE", "0")).lower() in ("1", "true", "yes")
        except Exception:
            self._trace_enabled = False

    def _get_config(self) -> AgentConfig:
        return self._config or AgentConfig()

    # DEBUG_PRINT: helper to emit gated console prints
    def _dprint(self, msg: str) -> None:
        if self._debug:
            print(f"[ANALYSIS_DEBUG] {msg}")

    # Safe ASCII sanitization to avoid Windows cp1252 UnicodeEncodeError (emojis etc.)
    def _ascii_sanitize(self, text: str, keep_newlines: bool = True) -> str:
        try:
            if not isinstance(text, str):
                text = str(text)
            sanitized = text.encode("ascii", "ignore").decode()
            if not keep_newlines:
                sanitized = sanitized.replace("\n", " ")
            return sanitized
        except Exception:
            return text if isinstance(text, str) else str(text)

    def _debug_ascii(self, text: str) -> None:
        try:
            self._logger.debug(self._ascii_sanitize(text))
        except Exception:
            pass

    # Compact state snapshot for trace logs
    def _snapshot_state(self, state: AnalysisPipelineState) -> Dict[str, Any]:
        try:
            return {
                "dataset_path": state.dataset_path,
                "instruction": state.instruction,
                "output_mode": state.output_mode,
                "visualization_only": state.visualization_only,
                "context_summary": (state.context_summary or "")[:800],
                "plan_steps": state.plan_steps[-6:],
                "next_step_index": state.next_step_index,
                "artifact_log": state.artifact_log[-10:],
                "token_metrics": state.token_metrics,
                "telemetry": getattr(state, "telemetry", {}),
            }
        except Exception:
            return {"snapshot_error": True}

    # Heuristic extraction of artifact paths (plots, csvs) from tool outputs
    def _parse_artifact_paths(self, text: Optional[str]) -> list[str]:
        if not text:
            return []
        out: list[str] = []
        try:
            # 1) Markdown image/link syntax ![alt](path)
            for m in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text, flags=re.IGNORECASE):
                out.append(m.strip())
            # 2) Common phrases like "saved to", "written to", "output file:"
            for m in re.findall(r"(?:saved\s+(?:to|as)|written\s+to|output(?:\s+file)?:)\s+([^\s)]+)", text, flags=re.IGNORECASE):
                out.append(m.strip())
            # 3) Bare filenames that look like artifacts (.png/.jpg/.csv/.svg/.pdf)
            for m in re.findall(r"[\w\-./\\]+\.(?:png|jpg|jpeg|svg|csv|pdf)", text, flags=re.IGNORECASE):
                out.append(m.strip())
        except Exception:
            return []
        # De-duplicate while preserving order
        seen = set()
        unique: list[str] = []
        for p in out:
            if p not in seen:
                seen.add(p)
                unique.append(p)
        return unique

    async def setup(self) -> None:
        await self._connect_mcp()
        await self._build_graph()

    # Structured node logging for traceability across a single run
    def _log_node(self, node: str, state: Optional[AnalysisPipelineState] = None, extra: str = "") -> None:
        try:
            rid = self._current_run_id or "-"
            step_idx = getattr(state, "next_step_index", None) if state is not None else None
            plan_len = len(getattr(state, "plan_steps", []) or []) if state is not None else None
            ref_round = getattr(state, "refinement_round", None)
            transcript_len = len(getattr(state, "tool_transcript", []) or []) if state is not None else None
            msg = f"run={rid} node={node} step_index={step_idx} plan_len={plan_len} refine_round={ref_round} transcript_len={transcript_len} {extra}".strip()
            self._logger.info(msg)

            # Phase 3 streaming: emit a small custom progress event so the
            # A2A service can show progress even when LLM token streaming is quiet.
            # This does not mutate LangGraph state.
            self._emit_custom_progress(
                {
                    "type": "progress",
                    "event": "node_start",
                    "run_id": rid,
                    "node": node,
                    "step_index": step_idx,
                    "refinement_round": ref_round,
                    "ts": datetime.now().isoformat(),
                }
            )
            if self._trace_enabled and state is not None:
                try:
                    snap = self._snapshot_state(state)
                    self._logger.info("TRACE_STATE_BEFORE::" + node + "::" + json.dumps(snap, ensure_ascii=True))
                except Exception:
                    pass
        except Exception:
            # Logging must not break pipeline
            pass

    def _emit_custom_progress(self, payload: Dict[str, Any]) -> None:
        """Emit a LangGraph custom stream event (no-op when not streaming custom).

        Uses LangGraph's StreamWriter, which is only meaningful when the caller
        requested stream_mode includes "custom". Safe to call otherwise.
        """
        try:
            writer = get_stream_writer()
            # Keep payload JSON-serializable and small
            if isinstance(payload, dict):
                writer(payload)
            else:
                writer({"type": "progress", "event": "unknown", "payload": str(payload)[:500]})
        except Exception:
            # Never let missing runtime context break pipeline execution.
            return

    async def _connect_mcp(self) -> None:
        """Initialize MCP tools using langchain_mcp_adapters like the successful original agent."""
        try:
            self._logger.info("Initializing MCP tools using langchain_mcp_adapters...")
            
            # Initialize LLM first
            self._llm = llm_with_fallbacks()
            # Auxiliary model for utility tasks (fallback to main if not configured)
            try:
                self._aux_llm = self._llm  # Use same handle unless a split is configured elsewhere
            except Exception:
                self._aux_llm = self._llm
            
            def _normalize_mcp_server_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
                normalized = dict(entry)
                try:
                    cmd = str(normalized.get("command", "")).strip().lower()
                    if normalized.get("transport") == "stdio" and cmd in ("python", "python.exe", "py"):
                        normalized["command"] = sys.executable
                except Exception:
                    pass
                return normalized

            # Build server configuration dict (but ensure stdio servers use our current interpreter)
            server_config: Dict[str, Dict[str, Any]] = {}
            for key, val in analysis_server_config.items():
                server_config[key] = _normalize_mcp_server_entry(val)
            for key, val in data_process_server_config.items():
                server_config[key] = _normalize_mcp_server_entry(val)
            
            self._logger.info(f"Starting MCP servers: {list(server_config.keys())}")
            
            # Create MCP client with better error handling like original
            try:
                self._mcp_client = MultiServerMCPClient(server_config)
                
                # Load tools from MCP server with connection error handling
                try:
                    self._langchain_tools = await self._mcp_client.get_tools()
                    self._logger.info(f"Loaded {len(self._langchain_tools)} LangChain-compatible tools")
                    self._logger.info(f"Active servers: {', '.join(server_config.keys())}")
                except (
                    ConnectionError,
                    ConnectionResetError,
                    OSError,
                ) as conn_error:
                    self._logger.warning(f"Connection error loading tools: {conn_error}")
                    # Try to continue with partial tools if any were loaded
                    if not self._langchain_tools:
                        raise
                        
            except Exception as mcp_error:
                self._logger.error(f"MCP client initialization failed: {mcp_error}")
                # Clean up any partial connections
                if hasattr(self, "_mcp_client") and self._mcp_client:
                    try:
                        await self._mcp_client.close()
                    except Exception:
                        pass  # Ignore cleanup errors
                    self._mcp_client = None
                raise
            
            # Build tool map and descriptions for compatibility
            for tool in self._langchain_tools:
                self._tool_map[tool.name] = tool
                self._tool_desc_map[tool.name] = tool.description or "No description available"
            
            # Build tool descriptions string
            descriptions = []
            for tool in self._langchain_tools:
                descriptions.append(f"- {tool.name}: {tool.description or 'No description'}")
            self._tool_descriptions = "\n".join(descriptions)
            
            # Create React agent with LangChain tools
            self._react_agent = create_react_agent(self._llm, self._langchain_tools)
            self._logger.info("Created React agent with LangChain-compatible tools")
            
        except Exception as e:
            self._logger.error(f"Failed to initialize MCP tools: {e}")
            # FAIL FAST: No fallback, raise the error immediately
            raise RuntimeError(f"MCP tool initialization failed: {e}") from e

    # ----------------------------
    # Tool resolution utilities
    # ----------------------------
    def _resolve_tool(self, candidate: Optional[str], intent: Optional[str]) -> Optional[str]:
        """Resolve a candidate tool name or intent to a known MCP tool name.

        Strategy:
        - Exact match if candidate in tool_map.
        - Case-insensitive and close-match using difflib.
        - If intent provided, score name/description for keyword overlap and pick best.
        """
        if not self._tool_map:
            return None

        names = list(self._tool_map.keys())

        # Exact match
        if candidate and candidate in self._tool_map:
            return candidate

        cand_norm = (candidate or "").strip().lower()
        # Case-insensitive exact
        for n in names:
            if cand_norm and n.lower() == cand_norm:
                return n

        # Close match by name
        if cand_norm:
            close = difflib.get_close_matches(cand_norm, [n.lower() for n in names], n=1, cutoff=0.72)
            if close:
                # Map back to original casing
                for n in names:
                    if n.lower() == close[0]:
                        return n

        # Intent/description-based scoring
        if intent:
            intent_tokens = [tok for tok in re.split(r"[^a-zA-Z0-9]+", intent.lower()) if tok]
            best_name = None
            best_score = 0.0
            for n in names:
                desc = (self._tool_desc_map.get(n, "") or "").lower()
                name_tokens = re.split(r"[^a-zA-Z0-9]+", n.lower())
                # Token overlap score
                overlap = len(set(intent_tokens) & set(name_tokens))
                desc_overlap = sum(1 for tok in intent_tokens if tok in desc)
                name_ratio = difflib.SequenceMatcher(None, cand_norm, n.lower()).ratio() if cand_norm else 0.0
                score = overlap * 1.5 + desc_overlap * 0.5 + name_ratio
                if score > best_score:
                    best_score = score
                    best_name = n
            if best_name and best_score >= 2.0:  # require minimal confidence
                return best_name

        return None

    async def _build_graph(self) -> None:
        """Build graph with explicit edges for Command-based routing."""
        graph = StateGraph(AnalysisPipelineState, input_schema=AnalysisPipelineState, output_schema=PipelineOutputState)

        graph.add_node("ingest_request", self.ingest_request)
        graph.add_node("planner", self.planner)
        graph.add_node("execute_step", self.execute_step)
        graph.add_node("interpret_results", self.interpret_results)
        graph.add_node("delegate_code_interpreter", self.delegate_code_interpreter)
        # NOTE: reflect returns Command with goto targets; provide ends for accurate graph rendering.
        graph.add_node(
            "reflect",
            self.reflect,
            ends=["planner", "execute_step", "interpret_results", "delegate_code_interpreter", "synthesis", END],
        )
        graph.add_node("synthesis", self.synthesis)
        # Ensure report persistence always runs even if synthesis defers writing
        graph.add_node("persist_cleanup", self.persist_cleanup)

        # Basic static transitions
        graph.add_edge(START, "ingest_request")
        graph.add_edge("ingest_request", "planner")
        graph.add_edge("planner", "execute_step")
        # NOTE: Do not add a second outgoing edge from planner (e.g., planner->synthesis).
        # Multiple outgoing edges can cause premature synthesis before planned tools run.

        # Conditional routers to eliminate uncontrolled cyclical edge expansion which was
        # causing post-final recursion beyond the single-run guard.
        def _route_execute(state: AnalysisPipelineState):
            try:
                # If guard latched, terminate immediately.
                if getattr(self, "_single_run_guard_enabled", False) and getattr(self, "_single_run_completed", False):
                    self._logger.debug("route_execute: single-run completed -> END")
                    return END
                # Finalize flag forces synthesis.
                if getattr(state, "finalize", False):
                    return "interpret_results"  # ensure insights extraction with existing transcript
                # If no plan, go straight to interpretation -> synthesis path.
                if not state.plan_steps:
                    return "interpret_results"
                # Continue executing while steps remain.
                if state.next_step_index < len(state.plan_steps):
                    return "execute_step"
                # All steps done -> interpret.
                return "interpret_results"
            except Exception:
                return "interpret_results"

        def _route_interpret(state: AnalysisPipelineState):
            try:
                if getattr(self, "_single_run_guard_enabled", False) and getattr(self, "_single_run_completed", False):
                    self._logger.debug("route_interpret: single-run completed -> END")
                    return END
                # reflect owns the decision to execute/replan/synthesize.
                return "reflect"
            except Exception:
                return "reflect"

        # Visualization aid: explicit conditional edges from reflect so Mermaid shows
        # the possible Command(goto=...) targets, even though reflect usually returns Command.
        def _route_reflect(state: AnalysisPipelineState):
            try:
                # If guard latched, terminate immediately.
                if getattr(self, "_single_run_guard_enabled", False) and getattr(self, "_single_run_completed", False):
                    return "synthesis"

                # Mirror reflect() high-level decisions (no LLM calls here).
                disable_reflect = str(os.getenv("ANALYSIS_DISABLE_REFLECT", "0")).lower() in ("1", "true", "yes")
                try:
                    if str(os.getenv("ANALYSIS_SIMPLE_MODE", "0")).lower() in ("1", "true", "yes"):
                        disable_reflect = True
                except Exception:
                    pass

                if disable_reflect:
                    transcript = state.tool_transcript or []
                    has_ok = any((e.get("status") == "ok") for e in transcript if isinstance(e, dict))
                    has_errors = any((e.get("status") == "error") for e in transcript if isinstance(e, dict))
                    current_round = getattr(state, "refinement_round", 0) or 0
                    try:
                        max_ref = int(os.getenv("ANALYSIS_MAX_REFINEMENTS", "1"))
                    except Exception:
                        max_ref = 1
                    if (not has_ok) and has_errors and current_round < max_ref:
                        return "planner"
                    remaining = len(state.plan_steps or []) - state.next_step_index
                    return "execute_step" if remaining > 0 else "synthesis"

                # Finalize flag forces synthesis.
                if getattr(state, "finalize", False):
                    return "synthesis"

                # If answered or nothing missing, synthesize.
                answered = bool(state.coverage.get("answers_query")) if isinstance(state.coverage, dict) else False
                missing = state.coverage.get("missing_info") if isinstance(state.coverage, dict) else []
                missing = [m for m in (missing or []) if isinstance(m, str) and m.strip()]

                try:
                    end_on_answer = str(os.getenv("ANALYSIS_END_ON_ANSWER", "1")).lower() in ("1", "true", "yes")
                except Exception:
                    end_on_answer = True
                if end_on_answer and (answered or not missing):
                    return "synthesis"

                # Execute remaining planned steps first.
                remaining_steps = len(state.plan_steps or []) - state.next_step_index
                if remaining_steps > 0:
                    return "execute_step"

                # PLACEHOLDER FOR INTEGRATION WITH CI
                # Visualization-only: reflect() may route to delegate_code_interpreter when missing_info is non-actionable
                # and/or _diagnose_tool_gap indicates a capability gap. We avoid duplicating full actionable filtering
                # logic here (router must be deterministic and side-effect free), and instead use a conservative proxy.
                try:
                    delegation_enabled = str(os.getenv("ANALYSIS_ENABLE_CODE_INTERPRETER_DELEGATION", "0")).lower() in (
                        "1",
                        "true",
                        "yes",
                    )
                except Exception:
                    delegation_enabled = False
                if delegation_enabled and missing:
                    try:
                        max_delegations = int(os.getenv("ANALYSIS_MAX_CI_DELEGATIONS", "1"))
                    except Exception:
                        max_delegations = 1
                    attempts = int(getattr(state, "delegation_attempts", 0) or 0)
                    if attempts < max_delegations:
                        return "delegate_code_interpreter"

                # If still missing, replan until capped.
                try:
                    max_refinements = int(os.getenv("ANALYSIS_MAX_REFINEMENTS", "2"))
                except Exception:
                    max_refinements = 2
                current_round = getattr(state, "refinement_round", 0) or 0
                if current_round >= max_refinements:
                    return "synthesis"

                return "planner" if missing else "synthesis"
            except Exception:
                return "synthesis"

        # NOTE: Provide explicit path maps so graph rendering includes conditional edges
        # even when router functions lack Literal return annotations.
        graph.add_conditional_edges(
            "execute_step",
            _route_execute,
            {
                "execute_step": "execute_step",
                "interpret_results": "interpret_results",
                END: END,
            },
        )
        # After interpretation, always go through reflect to decide: execute remaining steps,
        # replan, or synthesize.
        graph.add_edge("interpret_results", "reflect")
        graph.add_conditional_edges(
            "reflect",
            _route_reflect,
            {
                "planner": "planner",
                "execute_step": "execute_step",
                "interpret_results": "interpret_results",
                "delegate_code_interpreter": "delegate_code_interpreter",
                "synthesis": "synthesis",
                END: END,
            },
        )
        # Always run a final persistence step to write Markdown and metadata
        graph.add_edge("synthesis", "persist_cleanup")
        graph.add_edge("persist_cleanup", END)

        self._graph = graph

        # HITL requires a checkpointer for interrupt/resume.
        try:
            hitl_enabled = str(os.getenv("ANALYSIS_HITL_PLAN_REVIEW", "0")).lower() in ("1", "true", "yes")
            if not hitl_enabled:
                hitl_enabled = str(os.getenv("ANALYSIS_HITL_ENABLED", "0")).lower() in ("1", "true", "yes")
        except Exception:
            hitl_enabled = False

        compiled = None
        if hitl_enabled:
            try:
                checkpointer = InMemorySaver()
                compiled = graph.compile(checkpointer=checkpointer)
                try:
                    self._logger.info("HITL enabled: using InMemorySaver checkpointer")
                except Exception:
                    pass
            except Exception as _e:
                compiled = None
                try:
                    self._logger.warning(f"HITL enabled but failed to set checkpointer: {_e}")
                except Exception:
                    pass

        if compiled is None:
            compiled = graph.compile()

        class _CompiledPipelineWrapper:
            """Duck-typed wrapper around the compiled graph.

            LangGraph often returns plain dicts even when output_schema is provided.
            Tests (and some calling code) expect attribute access like output.status.
            This wrapper converts dict outputs to PipelineOutputState.
            """

            def __init__(self, inner):
                self._inner = inner

            def __getattr__(self, name: str):
                return getattr(self._inner, name)

            @staticmethod
            def _coerce_output(result):
                if isinstance(result, PipelineOutputState):
                    return result
                if isinstance(result, dict):
                    try:
                        # Pydantic v2
                        return PipelineOutputState.model_validate(result)
                    except Exception:
                        return PipelineOutputState(**result)
                return result

            async def ainvoke(self, *args, **kwargs):
                res = await self._inner.ainvoke(*args, **kwargs)
                return self._coerce_output(res)

            def invoke(self, *args, **kwargs):
                res = self._inner.invoke(*args, **kwargs)
                return self._coerce_output(res)

        self.agent = _CompiledPipelineWrapper(compiled)
        # Avoid non-ASCII to prevent Windows cp1252 logging errors
        self._logger.info("LangGraph analysis pipeline compiled")

    # ------------------------------------------------------------------
    # Graph nodes (Phase 5.2 implementations)
    # ------------------------------------------------------------------
    async def ingest_request(self, state: AnalysisPipelineState) -> Command:
        """Initial state hygiene: trim message window, load historical context if available."""
        self._log_node("ingest_request", state)
        # New run: reset single-run latch and cached output to avoid cross-run short-circuits
        try:
            self._single_run_completed = False
            self._cached_final_output = None
        except Exception:
            pass
        state.trim_message_window()

        injection_warning: Optional[str] = None
        injection_telemetry: Dict[str, int] = {}
        # Guardrail Option 4 (detection only): prompt-injection risk scoring (instruction only at ingress)
        try:
            score, flags = score_prompt_injection(state.instruction or "")
            if score >= 6 and flags:
                injection_warning = f"Prompt-injection risk detected (score={score}, flags={','.join(flags)})"
                injection_telemetry = {"injection_flags": 1}
        except Exception:
            injection_warning = None
        
        # Attempt to load historical memory for this dataset
        if state.dataset_path:
            history = self._memory_store.load_history(state.dataset_path)
            if history:
                # Build a concise snippet from the most recent run
                last_run = history[0] if isinstance(history, list) and history else {}
                snippet_parts = []
                if last_run.get("findings_summary"):
                    snippet_parts.append(f"Previous findings: {last_run['findings_summary'][:200]}")
                # Human-in-the-loop plan review context (if present)
                try:
                    hitl = last_run.get("hitl_plan_review") if isinstance(last_run, dict) else None
                    if isinstance(hitl, dict):
                        decision = str(hitl.get("decision") or "").strip()
                        feedback = hitl.get("feedback")
                        if decision:
                            snippet_parts.append(f"Previous plan review decision: {decision}")
                        if isinstance(feedback, str) and feedback.strip():
                            fb = feedback.strip()
                            if len(fb) > 300:
                                fb = fb[:297] + "..."
                            snippet_parts.append(f"Previous plan feedback: {fb}")
                        steps = hitl.get("plan_steps")
                        if isinstance(steps, list) and steps:
                            lines = []
                            for s in steps[:5]:
                                if not isinstance(s, dict):
                                    continue
                                tool = str(s.get("tool") or s.get("name") or "").strip()
                                why = str(s.get("why") or s.get("description") or "").strip()
                                if tool and why:
                                    lines.append(f"- {tool}: {why}")
                                elif tool:
                                    lines.append(f"- {tool}")
                            if lines:
                                snippet_parts.append("Previously reviewed plan steps:\n" + "\n".join(lines))
                except Exception:
                    pass
                if last_run.get("issues"):
                    snippet_parts.append(f"Known issues: {', '.join(last_run['issues'][:3])}")
                snippet = "\n".join(snippet_parts) if snippet_parts else None
                try:
                    if isinstance(snippet, str) and len(snippet) > 1400:
                        snippet = snippet[:1397] + "..."
                except Exception:
                    pass
                # Try to load a persisted dataset profile to reuse across runs
                persisted_profile = None
                try:
                    persisted_profile = self._load_persisted_profile(state.dataset_path)
                except Exception:
                    persisted_profile = None
                updates = {"historical_snippet": snippet}
                if isinstance(persisted_profile, dict) and persisted_profile.get("columns"):
                    updates["preprocess_profile"] = persisted_profile
                if injection_warning:
                    updates["warnings"] = [injection_warning]
                if injection_telemetry:
                    updates["telemetry"] = injection_telemetry
                return Command(update=updates, goto="planner")
        updates: Dict[str, Any] = {}
        if injection_warning:
            updates["warnings"] = [injection_warning]
        if injection_telemetry:
            updates["telemetry"] = injection_telemetry
        return Command(update=updates, goto="planner")

    async def _build_column_mappings_with_llm(self, dataset_snapshot: Dict[str, Any]) -> Dict[str, List[str]]:
        """Build intelligent mappings using LLM to analyze column names, types, and sample data.
        
        The LLM examines actual data to identify column purposes, not just keyword matching.
        This handles edge cases like 'xyzabc' being a datetime column.
        
        Returns dict like:
        {
            "datetime_column": ["crash_date", "xyzabc"],  # LLM detected xyzabc has dates
            "text_column": ["description", "notes"],
            "numeric_column": ["severity", "count"]
        }
        """
        mappings = {}
        
        columns = dataset_snapshot.get('columns', [])
        if not columns:
            return mappings
        
        # Prepare context for LLM
        column_info = []
        for col in columns[:20]:  # Limit to first 20 columns to avoid token bloat
            info = {"name": col}
            
            # Add type information
            dtypes = dataset_snapshot.get('dtypes', {})
            if col in dtypes:
                info["dtype"] = dtypes[col]
            
            # Add sample values
            sample_values = dataset_snapshot.get('sample_values', {})
            if col in sample_values:
                info["samples"] = sample_values[col][:3]  # First 3 samples
            
            column_info.append(info)
        
        # Build prompt for LLM
        prompt = f"""Analyze these dataset columns and classify them by PURPOSE (not just name).

COLUMNS TO ANALYZE:
{json.dumps(column_info, indent=2, ensure_ascii=False)}

TASK: For each column, determine its semantic purpose by examining:
1. Column name (but don't rely only on keywords!)
2. Data type (object/int64/float64/datetime64)
3. Sample values (MOST IMPORTANT - what does the data look like?)

CLASSIFICATION CATEGORIES:
- datetime_columns: Contains dates, times, or timestamps (check sample values!)
- text_columns: Contains free-form text descriptions or narratives
- categorical_columns: Contains categories, labels, or groupings (limited unique values)
- numeric_columns: Contains numeric measurements or counts
- location_columns: Contains geographic data (lat/long, addresses, places)
- identifier_columns: Contains IDs, codes, or unique identifiers

CRITICAL: A column named 'xyzabc' with samples like ['2024-01-15', '2024-02-20'] IS a datetime column!
Don't just match keywords - analyze the ACTUAL DATA.

Return ONLY valid JSON (no markdown):
{{
  "datetime_columns": ["col1", "col2"],
  "text_columns": ["col3"],
  "categorical_columns": ["col4", "col5"],
  "numeric_columns": ["col6"],
  "location_columns": ["col7"],
  "identifier_columns": ["col8"]
}}"""

        try:
            # Call LLM for intelligent classification
            from langchain_core.messages import HumanMessage, SystemMessage
            
            system_msg = SystemMessage(content="You are a data analysis expert. Classify columns by PURPOSE based on their data, not just names.")
            user_msg = HumanMessage(content=prompt)
            
            response = await self._llm.ainvoke([system_msg, user_msg])
            raw = response.content if hasattr(response, 'content') else str(response)
            
            # Parse LLM response
            classification = self._extract_json(raw)
            
            if isinstance(classification, dict):
                # Map to our parameter naming convention
                if classification.get('datetime_columns'):
                    mappings['datetime_column'] = classification['datetime_columns']
                    mappings['date_column'] = classification['datetime_columns']
                    mappings['temporal_column'] = classification['datetime_columns']
                
                if classification.get('text_columns'):
                    mappings['text_column'] = classification['text_columns']
                
                if classification.get('categorical_columns'):
                    mappings['category_column'] = classification['categorical_columns']
                    mappings['categorical_column'] = classification['categorical_columns']
                    mappings['group_column'] = classification['categorical_columns']
                
                if classification.get('numeric_columns'):
                    mappings['numeric_column'] = classification['numeric_columns']
                    mappings['value_column'] = classification['numeric_columns']
                
                if classification.get('location_columns'):
                    mappings['location_column'] = classification['location_columns']
                
                self._logger.info(f"LLM classified {len(column_info)} columns into {len(mappings)} parameter types")
                return mappings
            else:
                self._logger.warning("LLM column classification returned invalid format, falling back to keyword matching")
        
        except Exception as e:
            self._logger.warning(f"LLM column classification failed: {e}, falling back to keyword matching")
        
        # FALLBACK: Use keyword-based classification if LLM fails
        return self._build_column_mappings_fallback(dataset_snapshot)
    
    def _build_column_mappings_fallback(self, dataset_snapshot: Dict[str, Any]) -> Dict[str, List[str]]:
        """Fallback keyword-based column mapping (used if LLM fails).
        
        This is the original implementation - fast but less intelligent.
        """
        mappings = {}
        
        columns = dataset_snapshot.get('columns', [])
        if not columns:
            return mappings
        
        # Temporal/datetime column mapping
        datetime_keywords = ['date', 'time', 'datetime', 'timestamp', 'created', 'updated', 'crash', 'incident']
        datetime_candidates = [
            col for col in columns 
            if any(kw in col.lower() for kw in datetime_keywords)
        ]
        if datetime_candidates:
            mappings['datetime_column'] = datetime_candidates
            mappings['date_column'] = datetime_candidates
            mappings['temporal_column'] = datetime_candidates
        
        # Text column mapping
        text_keywords = ['desc', 'description', 'note', 'notes', 'comment', 'text', 'message', 'summary', 'narrative']
        text_candidates = [
            col for col in columns 
            if any(kw in col.lower() for kw in text_keywords)
        ]
        if text_candidates:
            mappings['text_column'] = text_candidates
        
        # Categorical column mapping
        categorical_cols = dataset_snapshot.get('categorical_columns', [])
        if categorical_cols:
            mappings['category_column'] = categorical_cols
            mappings['categorical_column'] = categorical_cols
            mappings['group_column'] = categorical_cols
        
        # Numeric column mapping
        numeric_cols = dataset_snapshot.get('numeric_columns', [])
        if numeric_cols:
            mappings['numeric_column'] = numeric_cols
            mappings['value_column'] = numeric_cols
        
        # Location columns
        location_keywords = ['lat', 'latitude', 'long', 'longitude', 'location', 'address', 'place']
        location_candidates = [
            col for col in columns 
            if any(kw in col.lower() for kw in location_keywords)
        ]
        if location_candidates:
            mappings['location_column'] = location_candidates
        
        return mappings

    async def planner(self, state: AnalysisPipelineState) -> Command:
        """LLM-based planner producing structured plan_steps; no execution here."""
        self._log_node("planner", state)
        # SINGLE RUN GUARD: If a synthesis already finalized this run, do not re-plan.
        if getattr(self, "_single_run_guard_enabled", False) and getattr(self, "_single_run_completed", False):
            self._logger.info("planner: single-run guard active; skipping planning and routing to END")
            return Command(update={}, goto=END)
        
        # Ground planning in the actual dataset by ensuring we have a schema/profile
        # Prefer a cached/persisted profile; otherwise build it now and persist for reuse
        if isinstance(getattr(state, 'preprocess_profile', None), dict) and state.preprocess_profile.get('columns'):
            dataset_snapshot: Dict[str, Any] = dict(state.preprocess_profile)
            self._logger.info("planner: Reusing cached dataset profile from state")
        else:
            dataset_snapshot: Dict[str, Any] = await self._build_dataset_inspection(state)
            try:
                if isinstance(dataset_snapshot, dict) and dataset_snapshot.get('columns') and state.dataset_path:
                    self._save_persisted_profile(state.dataset_path, dataset_snapshot)
                    self._logger.info("planner: Persisted dataset profile for reuse")
            except Exception:
                pass
        
        # Build intelligent column mappings using LLM to analyze actual data
        # This can detect 'xyzabc' as a datetime column if it contains date values
        # OPTIMIZATION: Only run LLM classification once per dataset, reuse cached mappings
        column_mappings = getattr(state, 'column_mappings', None)
        if not column_mappings:
            column_mappings = await self._build_column_mappings_with_llm(dataset_snapshot)
            self._logger.info("planner: Performed LLM column classification (first time)")
        else:
            self._logger.debug("planner: Reusing cached column mappings from previous planning iteration")
        
        # Log discovered mappings for debugging
        if column_mappings:
            self._logger.info(f"planner: Discovered column mappings: {json.dumps(column_mappings, indent=2)}")
        else:
            self._logger.warning("planner: No column mappings discovered from dataset")
        
        # Build available tools description with INTELLIGENT COLUMN HINTS
        tool_lines: List[str] = []
        for name, tool in self._tool_map.items():
            try:
                # Extract args schema keys if tool has args schema attribute
                arg_schema = []
                if hasattr(tool, 'args_schema') and hasattr(tool.args_schema, 'model_fields'):
                    arg_schema = list(tool.args_schema.model_fields.keys())
                elif hasattr(tool, 'args') and isinstance(tool.args, dict):
                    arg_schema = list(tool.args.keys())
                
                # ENHANCED: Add column mapping hints for parameters
                schema_parts = []
                for param in arg_schema:
                    literal_options = []
                    # Try to include Literal enum choices if present
                    try:
                        if hasattr(tool, 'args_schema') and hasattr(tool.args_schema, 'model_fields'):
                            field_info = tool.args_schema.model_fields.get(param)
                            if field_info and hasattr(field_info, 'annotation'):
                                from typing import get_origin, get_args
                                origin = get_origin(field_info.annotation)
                                if origin is Literal:
                                    literal_options = [str(o) for o in get_args(field_info.annotation)]
                    except Exception:
                        literal_options = []
                    annotation_part = ''
                    if literal_options:
                        # Truncate very long lists for readability
                        opts_display = ','.join(literal_options[:8])
                        if len(literal_options) > 8:
                            opts_display += ',…'
                        annotation_part = f"<Literal:{opts_display}>"
                    if param in column_mappings and column_mappings[param]:
                        suggestions = column_mappings[param][:3]
                        schema_parts.append(f"{param}{(' ' + annotation_part) if annotation_part else ''}=[suggested: {', '.join(suggestions)}]")
                    else:
                        schema_parts.append(f"{param}{(' ' + annotation_part) if annotation_part else ''}")
                
                schema_str = ", ".join(schema_parts) if schema_parts else "(no documented args)"
                tool_lines.append(f"{name}: {schema_str}")
            except Exception:
                tool_lines.append(f"{name}: (schema unavailable)")
        available_tools_str = "\n".join(tool_lines)

        # ------------------------------------------------------------------
        # Avoid re-running already successful tool calls within the same run.
        # We dedupe by (tool_name, stable_json(args)). Errors are NOT cached.
        # ------------------------------------------------------------------
        executed_ok_signatures = set()
        try:
            for ev in (state.tool_transcript or []):
                if not isinstance(ev, dict):
                    continue
                if str(ev.get("status") or "") != "ok":
                    continue
                tname = str(ev.get("tool") or "").strip()
                if not tname:
                    continue
                try:
                    args_sig = json.dumps(ev.get("args") or {}, sort_keys=True, default=str)
                except Exception:
                    args_sig = str(ev.get("args") or {})
                executed_ok_signatures.add((tname, args_sig))
        except Exception:
            executed_ok_signatures = set()

        def _sig(tool_name: str, args_obj: Dict[str, Any]) -> tuple[str, str]:
            try:
                return (tool_name, json.dumps(args_obj or {}, sort_keys=True, default=str))
            except Exception:
                return (tool_name, str(args_obj or {}))

        # Persist snapshot for downstream nodes (e.g., tool arg inference) via delta
        updates: Dict[str, Any] = {}
        try:
            if isinstance(dataset_snapshot, dict):
                updates["preprocess_profile"] = dataset_snapshot
                # Also persist column mappings for execute_step to use
                updates["column_mappings"] = column_mappings
        except Exception:
            pass

        # Compact JSON snapshot for prompt (limit size by token budget, not a magic number)
        # Rationale: keep planner grounded without blowing context window. Default ~600 tokens.
        try:
            token_budget = int(os.getenv("ANALYSIS_PLANNER_SNAPSHOT_TOKENS", "600"))
        except Exception:
            token_budget = 600
        try:
            chars_per_tok = float(os.getenv("ANALYSIS_PLANNER_CHARS_PER_TOKEN", "4.0"))
        except Exception:
            chars_per_tok = 4.0
        char_cap = max(200, int(token_budget * chars_per_tok))
        snapshot_json = json.dumps(dataset_snapshot, ensure_ascii=False)
        if len(snapshot_json) > char_cap:
            snapshot_json = snapshot_json[:char_cap]

        prompt_dict = build_planner_prompt(state, available_tools_str)
        system_msg = SystemMessage(content=prompt_dict["system"])
        # UPDATED ENUM: 'analysis' deprecated; use insights_only for text-only analytical workflows.
        schema_hint = "{\"plan_meta\": {\"output_mode\": \"visualization_only|insights_only|mixed\", \"confidence\": <float>, \"rationale\": \"<brief justification>\"}, \"steps\": [{\"tool\": \"<tool_name>\", \"args\": {{}}, \"why\": \"<very short reason>\"}]}"
        
        # Extract explicit column list for CRITICAL constraint
        explicit_columns = []
        if isinstance(dataset_snapshot, dict) and 'columns' in dataset_snapshot:
            explicit_columns = dataset_snapshot.get('columns', [])
        
        # Build column mapping guidance for the LLM (ASCII-only for Windows safety)
        column_mapping_guide = ""
        if column_mappings:
            column_mapping_guide = "\n\n[INTELLIGENT COLUMN MAPPING GUIDE]\n"
            column_mapping_guide += "When planning tool calls, use these column suggestions:\n"
            for param_type, suggestions in column_mappings.items():
                if suggestions:
                    column_mapping_guide += f"- For '{param_type}' parameters: use {suggestions[0]}"
                    if len(suggestions) > 1:
                        column_mapping_guide += f" (alternatives: {', '.join(suggestions[1:3])})"
                    column_mapping_guide += "\n"
            column_mapping_guide += "\nEXAMPLE: If a tool needs 'datetime_column', use the suggested column name from the list above.\n"
        
        column_constraint = ""
        if explicit_columns:
            column_constraint = (
                f"\n\n[CRITICAL COLUMN CONSTRAINT]\n"
                f"The dataset has EXACTLY these columns: {explicit_columns}\n"
                f"DO NOT reference any column not in this list.\n"
                f"If the instruction mentions a column not in this list, skip that part or use an alternative column.\n"
                f"NEVER invent column names like 'DateTime', 'Timestamp', etc. unless they appear above.\n"
                f"{column_mapping_guide}"
            )
        
        refinement_block = ""
        try:
            refinement_text = str(prompt_dict.get("refinement") or "").strip()
            if refinement_text:
                refinement_block = f"\nRefinement targets (from reflection): {refinement_text}\n"
        except Exception:
            refinement_block = ""

        user_block = (
            f"Instruction: {prompt_dict['instruction']}\n"
            f"Dataset Path: {state.dataset_path or 'unspecified'}\n"
            f"Refinement round: {int(getattr(state, 'refinement_round', 0) or 0)}\n"
            f"{column_constraint}\n"
            f"Dataset Snapshot (for planning):\n{snapshot_json}\n"
            f"Available Tools:\n{available_tools_str}\n"
            f"Context:\n{prompt_dict['context']}\n"
            f"{refinement_block}\n"
            "Planning requirements:\n"
            "- Ground every step in the dataset snapshot above (columns/types/observed data).\n"
            "- Do NOT pick generic patterns; plan specifically for this dataset and question.\n"
            "- Use ONLY tool names shown above and ONLY columns from the dataset snapshot.\n"
            "- Provide minimal required args (e.g., file_path, columns matching actual dataset).\n"
            "- Prefer the shortest workflow that answers the instruction; skip ornamental plots.\n"
            "- If refinement targets exist, address them directly and avoid repeating already-successful tool calls.\n"
            "- If the instruction asks for analysis using columns that don't exist, adapt to use available columns.\n\n"
            "- Use data-quality assessment tools (e.g., assess_data_quality) at most once unless the user explicitly asks for data quality analysis. Prioritize domain analyses over repeated quality checks.\n\n"
            f"Return ONLY JSON matching: {schema_hint} (max {self._step_cap} steps)."
        )
        user_msg = HumanMessage(content=user_block)

        # Optional: Print full planner system + user blocks for debugging
        try:
            dbg_planner = str(os.getenv("ANALYSIS_DEBUG_PRINT_PLANNER", "0")).lower() in ("1", "true", "yes")
        except Exception:
            dbg_planner = False
        # Print only when explicit planner debug flag is set (avoid coupling to generic debug)
        if dbg_planner:
            self._debug_ascii("\n=== PLANNER SYSTEM PROMPT ===\n" + str(prompt_dict.get("system") or "") + "\n=== END SYSTEM PROMPT ===\n")
            self._debug_ascii("\n=== PLANNER USER PROMPT ===\n" + user_block + "\n=== END USER PROMPT ===\n")

        llm = self._llm
        raw = "{}"
        try:
            resp = await llm.ainvoke([system_msg, user_msg])
            raw = resp.content if isinstance(resp, AIMessage) else str(resp)
        except Exception as e:
            self._logger.error(f"planner: LLM invocation failed ({e}) - proceeding with empty plan")

        plan_payload = self._extract_json(raw)
        steps_raw = []
        output_mode = None
        classification_confidence = None
        classification_rationale = ""
        if isinstance(plan_payload, dict):
            # Extract steps
            steps_raw = plan_payload.get("steps") or []
            # Extract plan_meta classification
            meta = plan_payload.get("plan_meta") or {}
            try:
                output_mode = str(meta.get("output_mode") or "").strip().lower() or None
            except Exception:
                output_mode = None
            try:
                classification_confidence = meta.get("confidence")
                if isinstance(classification_confidence, str):
                    classification_confidence = float(classification_confidence)
            except Exception:
                classification_confidence = None
            try:
                classification_rationale = str(meta.get("rationale") or "")[:180]
            except Exception:
                classification_rationale = ""
        # Backward compatibility: map deprecated 'analysis' to 'insights_only'
        if output_mode == "analysis":
            output_mode = "insights_only"
        # Validation and proactive correction summary counters
        raw_considered = len(steps_raw[: self._step_cap])
        dropped_invalid_tool = 0
        validated: List[Dict[str, Any]] = []
        for s in steps_raw[: self._step_cap]:
            try:
                tname = str(s.get("tool") or "").strip()
                if not tname or tname not in self._tool_map:
                    dropped_invalid_tool += 1
                    continue
                args_obj = s.get("args") if isinstance(s.get("args"), dict) else {}
                if hasattr(self._tool_map[tname], 'args_schema') and hasattr(self._tool_map[tname].args_schema, 'model_fields'):
                    allowed = set(self._tool_map[tname].args_schema.model_fields.keys())
                    args_obj = {k: v for k, v in args_obj.items() if k in allowed}
                    
                    # Auto-inject file_path if missing but required
                    if 'file_path' in allowed and 'file_path' not in args_obj:
                        if state.dataset_path:
                            args_obj['file_path'] = state.dataset_path
                            self._logger.debug(f"planner: Auto-injected file_path for {tname}")
                # Proactively auto-correct invalid column arguments during planning
                try:
                    if isinstance(dataset_snapshot, dict) and dataset_snapshot.get('columns'):
                        args_obj = self._auto_correct_column_args(tname, args_obj, dataset_snapshot, column_mappings)
                except Exception as _e:
                    self._logger.debug(f"planner: arg auto-correction skipped for {tname}: {_e}")

                # Skip if this exact successful tool call already ran in this run.
                try:
                    if _sig(tname, args_obj) in executed_ok_signatures:
                        continue
                except Exception:
                    pass
                
                why = str(s.get("why") or s.get("reason") or "").strip()[:160] or "unspecified"
                validated.append({"tool": tname, "args": args_obj, "why": why})
            except Exception:
                continue

        # Determine route intents from explicit classification first; fallback heuristic only if missing
        viz_only_intent = False
        insights_only_intent = False
        if output_mode == "visualization_only":
            viz_only_intent = True
        elif output_mode == "insights_only":
            insights_only_intent = True
        elif output_mode == "mixed":
            # Mixed: neither exclusive flag set (both viz + insights allowed)
            pass
        else:
            # Fallback heuristic when classification absent: derive exclusive modes
            try:
                _inst_low = (state.instruction or "").lower()
                _wants_viz = any(tok in _inst_low for tok in ["plot", "chart", "graph", "visualization", "visualise", "visualize", "heatmap", "scatter", "histogram", "pairplot"])
                _wants_insights = any(tok in _inst_low for tok in ["insight", "insights", "recommendation", "report", "summary", "explain", "root cause", "factor", "relationship"])
                viz_only_intent = _wants_viz and not _wants_insights
                insights_only_intent = _wants_insights and not _wants_viz
            except Exception:
                viz_only_intent = False
                insights_only_intent = False

        # Helper to add a step if tool can be resolved and we have capacity (available to boosters/backfill)
        def _try_add_step(intent: str, preferred_name: Optional[str], build_args_fn) -> None:
            nonlocal validated
            if len(validated) >= self._step_cap:
                return
            tool_name = None
            # Prefer exact name if available, else resolve by intent
            if preferred_name and preferred_name in self._tool_map:
                tool_name = preferred_name
            else:
                tool_name = self._resolve_tool(preferred_name, intent)
            if not tool_name:
                return
            # Build args
            args_local = build_args_fn(tool_name)
            # Filter args by tool schema and inject file_path if needed
            try:
                if hasattr(self._tool_map[tool_name], 'args_schema') and hasattr(self._tool_map[tool_name].args_schema, 'model_fields'):
                    allowed = set(self._tool_map[tool_name].args_schema.model_fields.keys())
                    args_local = {k: v for k, v in (args_local or {}).items() if k in allowed}
                    if 'file_path' in allowed and 'file_path' not in args_local and state.dataset_path:
                        args_local['file_path'] = state.dataset_path
                else:
                    # Best effort: always include file_path when available
                    if state.dataset_path and 'file_path' not in args_local:
                        args_local['file_path'] = state.dataset_path
            except Exception:
                pass
            # Proactive column correction
            try:
                args_local = self._auto_correct_column_args(tool_name, args_local, dataset_snapshot, column_mappings)
            except Exception:
                pass

            # Skip if already executed successfully in this run.
            try:
                if _sig(tool_name, args_local) in executed_ok_signatures:
                    return
            except Exception:
                pass
            validated.append({
                'tool': tool_name,
                'args': args_local,
                'why': f"Add coverage: {intent[:120]}"
            })

        # Coverage booster: ensure major column groups get at least one analysis step
        # Skip when visualization-only (we only want the plot + brief insights) or insights-only (LLM will derive insights without extra tools)
        try:
            if viz_only_intent or insights_only_intent:
                raise RuntimeError("viz_only_intent: skipping coverage booster")
            # Determine available column groups
            numeric_cols = list((dataset_snapshot or {}).get('numeric_columns') or [])
            categorical_cols = list((dataset_snapshot or {}).get('categorical_columns') or [])
            datetime_cols = []
            if column_mappings:
                for k in ('datetime_column', 'date_column', 'temporal_column'):
                    if k in column_mappings and column_mappings[k]:
                        datetime_cols = column_mappings[k]
                        break

            # Identify which groups are already targeted by plan args
            def _args_use_any(args: Dict[str, Any], candidates: List[str]) -> bool:
                try:
                    vals = []
                    for v in (args or {}).values():
                        if isinstance(v, str):
                            vals.append(v)
                        elif isinstance(v, list):
                            vals.extend([x for x in v if isinstance(x, str)])
                    lower = {x.lower() for x in vals}
                    for c in candidates:
                        if c and c.lower() in lower:
                            return True
                except Exception:
                    pass
                return False

            has_temporal = any(_args_use_any(step.get('args', {}), datetime_cols) for step in validated) if datetime_cols else True
            has_numeric = any(_args_use_any(step.get('args', {}), numeric_cols) for step in validated) if numeric_cols else True
            has_categorical = any(_args_use_any(step.get('args', {}), categorical_cols) for step in validated) if categorical_cols else True

            # Add temporal analysis if dataset has datetime and plan lacks it
            if datetime_cols and not has_temporal:
                def _temporal_args(tool_name: str) -> Dict[str, Any]:
                    args = {
                        'datetime_column': datetime_cols[0],
                    }
                    # Try include a target/category if present
                    candidate_target = None
                    for cand in ('crash_type', 'most_severe_injury'):
                        if cand in categorical_cols:
                            candidate_target = cand
                            break
                    if candidate_target:
                        args['category_column'] = candidate_target
                    return args
                _try_add_step(
                    intent="analyze temporal patterns by hour, day and month; time-based incident trends",
                    preferred_name="analyze_temporal_patterns",
                    build_args_fn=_temporal_args,
                )

            # Add numeric correlation analysis if numeric columns present and not covered
            if numeric_cols and not has_numeric:
                def _numeric_args(tool_name: str) -> Dict[str, Any]:
                    cols = numeric_cols[:8]  # cap to avoid bloat
                    return {
                        'columns': cols,
                    }
                _try_add_step(
                    intent="correlation analysis across numeric columns; identify strongest relationships",
                    preferred_name="analyze_numeric_correlations",
                    build_args_fn=_numeric_args,
                )

            # Add categorical pattern analysis if categorical present and not covered
            if categorical_cols and not has_categorical:
                def _categorical_args(tool_name: str) -> Dict[str, Any]:
                    cols = [c for c in categorical_cols if c != 'crash_date'][:10]
                    return {
                        'columns': cols,
                    }
                _try_add_step(
                    intent="categorical distribution and cross-tab analysis; identify dominant categories",
                    preferred_name="analyze_csv_patterns",
                    build_args_fn=_categorical_args,
                )

        except Exception as _cov_e:
            self._logger.debug(f"planner: coverage booster skipped due to error: {_cov_e}")

        # Visualization backfill: only when not insights-only (user wants no plots in that mode)
        # If instruction clearly asks for visualization and no viz tool step present, append one plotting step.
        try:
            intent_text = (state.instruction or "").lower()
            wants_viz = any(tok in intent_text for tok in ["plot", "chart", "graph", "visualization", "visualise", "visualize"])  # en/alt spellings

            def _has_viz_step(steps: List[Dict[str, Any]]) -> bool:
                try:
                    for st in steps:
                        name = (st.get("tool") or "").lower()
                        if name in ("generate_plot", "data_visualization") or "visualization" in name:
                            return True
                except Exception:
                    return False
                return False

            if wants_viz and not insights_only_intent and not _has_viz_step(validated):
                # Prefer the concrete MCP plotting tool if available
                preferred = "generate_plot" if "generate_plot" in self._tool_map else None

                def _viz_args(tool_name: str) -> Dict[str, Any]:
                    args: Dict[str, Any] = {}
                    if tool_name == "generate_plot":
                        # Prefer schema-driven choices; avoid brittle hardcoding
                        cols = list((dataset_snapshot or {}).get('columns') or [])
                        dtypes = dict((dataset_snapshot or {}).get('dtypes') or {})
                        # Heuristic: choose bar_chart on an existing day-of-week column if one exists; else auto_temporal
                        day_like = None
                        try:
                            for c in cols:
                                low = str(c).lower()
                                if any(tok in low for tok in ("day_of_week", "dayofweek", "weekday", "dow")):
                                    day_like = c
                                    break
                        except Exception:
                            day_like = None
                        if day_like:
                            args = {
                                "plot_type": "bar_chart",
                                "x_column": day_like,
                                # y omitted → plotting tool will count occurrences of x and label categories
                                "y_column": None,
                                "title": f"Counts by {day_like}",
                            }
                        else:
                            # Let the tool auto-detect the best temporal view based on schema
                            args = {
                                "plot_type": "auto_temporal",
                            }
                    return args

                _try_add_step(
                    intent="visualize daily temporal pattern by day",
                    preferred_name=preferred,
                    build_args_fn=_viz_args,
                )
        except Exception as _viz_e:
            self._logger.debug(f"planner: visualization backfill skipped due to error: {_viz_e}")

        # Deduplicate steps by (tool, args) signature
        seen_signatures = set()
        deduplicated = []
        for step in validated:
            try:
                sig = (step['tool'], json.dumps(step['args'], sort_keys=True))
                if sig not in seen_signatures:
                    seen_signatures.add(sig)
                    deduplicated.append(step)
            except Exception:
                # If json serialization fails, keep the step anyway
                deduplicated.append(step)
        
        duplicates_removed = (len(validated) - len(deduplicated)) if len(deduplicated) < len(validated) else 0
        if duplicates_removed > 0:
            self._logger.info(f"planner: Removed {duplicates_removed} duplicate steps")
        # Log concise validation summary for observability
        try:
            self._logger.info(
                f"planner: validation_summary raw={raw_considered} kept={len(deduplicated)} dropped_invalid_tool={dropped_invalid_tool} duplicates_removed={duplicates_removed}"
            )
        except Exception:
            pass
        
        # Enforce per-tool repetition caps to avoid flooding with generic tools
        # Default: limit assess_data_quality to once per plan (env ANALYSIS_MAX_QA_STEPS overrides)
        try:
            max_qa = int(os.getenv("ANALYSIS_MAX_QA_STEPS", "1"))
        except Exception:
            max_qa = 1
        per_tool_caps: Dict[str, Optional[int]] = {
            "assess_data_quality": max_qa,
        }
        counts: Dict[str, int] = {}
        filtered_steps: List[Dict[str, Any]] = []
        dropped_by_cap: Dict[str, int] = {}
        for st in deduplicated:
            name = st.get("tool") or ""
            cap = per_tool_caps.get(name)
            if cap is None:
                filtered_steps.append(st)
                continue
            cnt = counts.get(name, 0)
            if cnt < cap:
                filtered_steps.append(st)
                counts[name] = cnt + 1
            else:
                dropped_by_cap[name] = dropped_by_cap.get(name, 0) + 1
        if dropped_by_cap:
            try:
                details = ", ".join([f"{k}:{v}" for k, v in dropped_by_cap.items()])
                self._logger.info(f"planner: per-tool caps applied, dropped by cap -> {details}")
            except Exception:
                pass

        validated = filtered_steps
        # ------------------------------------------------------------
        # POST-PLAN VALIDATION & PER-STEP MICRO-REVISION
        # ------------------------------------------------------------
        try:
            if validated:
                dataset_columns = set(dataset_snapshot.get('columns', [])) if isinstance(dataset_snapshot, dict) else set()
                revised_any = False
                final_steps: List[Dict[str, Any]] = []
                from typing import get_origin, get_args

                async def _revise_step(original_step: Dict[str, Any], issues: List[str]) -> Dict[str, Any]:
                    """Call LLM to repair a single invalid step while preserving intent."""
                    try:
                        tname = original_step.get('tool')
                        args_in = original_step.get('args', {})
                        why_in = original_step.get('why', '')
                        # Build mini schema description
                        field_lines = []
                        if tname and tname in self._tool_map and hasattr(self._tool_map[tname], 'args_schema') and hasattr(self._tool_map[tname].args_schema, 'model_fields'):
                            for fname, finfo in self._tool_map[tname].args_schema.model_fields.items():
                                lit_vals = []
                                try:
                                    origin = get_origin(finfo.annotation)
                                    if origin is Literal:
                                        lit_vals = [str(o) for o in get_args(finfo.annotation)]
                                except Exception:
                                    lit_vals = []
                                req_flag = 'required' if finfo.is_required() else 'optional'
                                lit_part = f" | choices: {', '.join(lit_vals)}" if lit_vals else ''
                                field_lines.append(f"- {fname} ({req_flag}){lit_part}")
                        mini_schema = '\n'.join(field_lines) or '(schema unavailable)'
                        issue_block = '\n'.join(f"* {i}" for i in issues[:6])
                        col_list = ', '.join(list(dataset_columns)[:30])
                        repair_prompt = (
                            "You must output ONLY JSON for a single corrected step.\n"
                            f"Tool: {tname}\nOriginal args: {json.dumps(args_in, ensure_ascii=False)}\n"
                            f"Original reason: {why_in[:160]}\n"
                            f"Issues detected:\n{issue_block}\n\n"
                            f"Dataset columns (authoritative): {col_list}\n"
                            f"Tool argument schema:\n{mini_schema}\n\n"
                            "Return JSON: {\"tool\": \"<same_tool>\", \"args\": {...}, \"why\": \"short reason\"}.\n"
                            "Rules: Use ONLY columns listed. Fix literals to valid choices. Include required args. Do not invent columns."
                        )
                        sys_msg = SystemMessage(content="You correct invalid tool plan steps strictly to schema.")
                        user_msg = HumanMessage(content=repair_prompt)
                        resp = await self._llm.ainvoke([sys_msg, user_msg])
                        fixed = self._extract_json(resp.content if hasattr(resp, 'content') else str(resp))
                        if isinstance(fixed, dict) and fixed.get('tool') == tname and isinstance(fixed.get('args'), dict):
                            fixed['why'] = fixed.get('why', why_in)[:160]
                            return fixed
                    except Exception as _rev_e:
                        self._logger.debug(f"planner: step revision failed ({_rev_e}); keeping original")
                    return original_step

                for step_obj in validated:
                    tname = step_obj.get('tool')
                    args_obj = step_obj.get('args', {}) or {}
                    issues: List[str] = []
                    # Skip unknown tools (already filtered earlier)
                    if not tname or tname not in self._tool_map:
                        issues.append('unknown_tool')
                    else:
                        # Required args check
                        if hasattr(self._tool_map[tname], 'args_schema') and hasattr(self._tool_map[tname].args_schema, 'model_fields'):
                            for fname, finfo in self._tool_map[tname].args_schema.model_fields.items():
                                try:
                                    if finfo.is_required() and fname not in args_obj:
                                        issues.append(f"missing_required:{fname}")
                                except Exception:
                                    continue
                            # Literal enum validation
                            for fname, finfo in self._tool_map[tname].args_schema.model_fields.items():
                                try:
                                    if fname in args_obj and hasattr(finfo, 'annotation'):
                                        origin = get_origin(finfo.annotation)
                                        if origin is Literal:
                                            allowed = set(str(v) for v in get_args(finfo.annotation))
                                            val = str(args_obj[fname])
                                            if val not in allowed:
                                                issues.append(f"invalid_literal:{fname}:{val}")
                                except Exception:
                                    continue
                        # Column existence validation (only for args that look like column refs)
                        for k, v in list(args_obj.items()):
                            try:
                                if isinstance(v, str) and k.lower().endswith('column'):
                                    if v not in dataset_columns:
                                        issues.append(f"invalid_column:{k}:{v}")
                                elif isinstance(v, list) and k.lower() in ('columns', 'numeric_columns', 'categorical_columns'):
                                    for colv in v:
                                        if isinstance(colv, str) and colv not in dataset_columns:
                                            issues.append(f"invalid_column_list_item:{k}:{colv}")
                            except Exception:
                                continue
                    if issues:
                        revised_any = True
                        step_obj = await _revise_step(step_obj, issues)
                    final_steps.append(step_obj)
                if revised_any:
                    self._logger.info("planner: one or more steps revised post-validation")
                validated = final_steps
        except Exception as _post_val_e:
            self._logger.debug(f"planner: post-plan validation error: {_post_val_e}")
        # Propagate route intents for downstream synthesis behavior
        viz_only_flag = viz_only_intent
        insights_only_flag = insights_only_intent
        if output_mode:
            updates["output_mode"] = output_mode
        if classification_confidence is not None:
            updates["classification_confidence"] = classification_confidence
        if classification_rationale:
            updates["classification_rationale"] = classification_rationale

        # Planner delta update with plan hashing and conditional index reset
        import hashlib
        try:
            plan_serial = json.dumps(validated, sort_keys=True, ensure_ascii=False)
            plan_hash = hashlib.sha256(plan_serial.encode("utf-8")).hexdigest()
        except Exception:
            plan_hash = None

        updates["plan_steps"] = validated
        updates["plan_hash"] = plan_hash
        if viz_only_flag:
            updates["visualization_only"] = True

        # If no valid steps were produced, finalize to avoid empty loops
        if len(validated) == 0:
            updates["next_step_index"] = 0
            updates["finalize"] = True
            self._logger.warning("planner: No valid steps; finalizing to synthesis")
            # Route through execute_step so a single planner edge controls flow.
            # execute_step will detect finalize and route to synthesis.
            return Command(update=updates, goto="execute_step")

        # Reset next_step_index only if the plan actually changed.
        # If the plan is unchanged *and* already exhausted, finalize immediately to avoid
        # reflect->planner loops that cannot make progress.
        prev_hash = getattr(state, "plan_hash", None)
        if prev_hash and plan_hash and prev_hash == plan_hash:
            current_index = int(getattr(state, "next_step_index", 0) or 0)
            updates["next_step_index"] = current_index
            updates["replan_no_change"] = True
            if current_index >= len(validated):
                updates["finalize"] = True
                self._logger.info("planner: Plan unchanged and exhausted; finalizing to synthesis")
            else:
                updates["finalize"] = False
                self._logger.info("planner: Plan unchanged; preserving next_step_index")
        else:
            updates["next_step_index"] = 0
            updates["replan_no_change"] = False
            updates["finalize"] = False
            self._logger.info("planner: New plan detected; resetting next_step_index to 0")

        # Telemetry: count planner invocations
        updates["telemetry"] = {"planner_calls": 1}
        # Note: __finalize__ may have been set above for unchanged+exhausted plans.

        # ------------------------------------------------------------------
        # Phase 4: HITL plan review (interrupt -> requires_input -> resume)
        # ------------------------------------------------------------------
        try:
            hitl_enabled = str(os.getenv("ANALYSIS_HITL_PLAN_REVIEW", "0")).lower() in ("1", "true", "yes")
            if not hitl_enabled:
                hitl_enabled = str(os.getenv("ANALYSIS_HITL_ENABLED", "0")).lower() in ("1", "true", "yes")
        except Exception:
            hitl_enabled = False

        if hitl_enabled:
            try:
                already_reviewed = bool(getattr(state, "hitl_plan_reviewed", False))
            except Exception:
                already_reviewed = False

            # Default: review only the first plan in a run; do not re-interrupt on replans.
            try:
                review_each_replan = str(os.getenv("ANALYSIS_HITL_REVIEW_EACH_REPLAN", "0")).lower() in ("1", "true", "yes")
            except Exception:
                review_each_replan = False

            if review_each_replan:
                # If the plan changed, require another review.
                try:
                    prev_hash = getattr(state, "plan_hash", None)
                    if (prev_hash is None) or (plan_hash and prev_hash != plan_hash):
                        already_reviewed = False
                except Exception:
                    pass

            if (not already_reviewed) and (not bool(updates.get("finalize"))):
                payload = {
                    "type": "plan_review",
                    "plan_steps": list(validated),
                    "plan_meta": {
                        "output_mode": output_mode,
                        "confidence": classification_confidence,
                        "rationale": classification_rationale,
                    },
                    "instruction": (state.instruction or "")[:500],
                    "dataset_path": state.dataset_path or "",
                }
                decision = interrupt(payload)

                decision_obj: Dict[str, Any] = {}
                if isinstance(decision, dict):
                    decision_obj = decision
                elif isinstance(decision, str) and decision.strip():
                    decision_obj = {"decision": decision.strip()}

                raw_decision = str(decision_obj.get("decision") or decision_obj.get("action") or "approve").strip().lower()
                feedback = decision_obj.get("feedback") or decision_obj.get("notes") or decision_obj.get("comment")
                if feedback is not None:
                    try:
                        feedback = str(feedback).strip()
                    except Exception:
                        feedback = None

                if raw_decision in ("approve", "approved", "yes", "y", "ok", "continue"):
                    updates["hitl_plan_reviewed"] = True
                    updates["hitl_plan_decision"] = "approve"
                    if feedback:
                        updates["hitl_plan_feedback"] = feedback
                    return Command(update=updates, goto="execute_step")

                if raw_decision in ("revise", "edit", "change", "update", "replan"):
                    updates["hitl_plan_reviewed"] = False
                    updates["hitl_plan_decision"] = "revise"
                    if feedback:
                        updates["hitl_plan_feedback"] = feedback
                        updates["messages"] = [HumanMessage(content=f"Plan feedback: {feedback}")]
                    # Re-run planner to incorporate feedback.
                    return Command(update=updates, goto="planner")

                if raw_decision in ("abort", "cancel", "stop"):
                    updates["hitl_plan_reviewed"] = True
                    updates["hitl_plan_decision"] = "abort"
                    if feedback:
                        updates["hitl_plan_feedback"] = feedback
                    updates["finalize"] = True
                    # Route via execute_step so existing routing handles synth/cleanup.
                    return Command(update=updates, goto="execute_step")

                # Unknown decision: default to approve to avoid deadlock.
                updates["hitl_plan_reviewed"] = True
                updates["hitl_plan_decision"] = "approve"
                return Command(update=updates, goto="execute_step")
        
        return Command(update=updates, goto="execute_step")

    def _auto_correct_column_args(self, tool_name: str, args: Dict[str, Any], dataset_snapshot: Dict[str, Any], 
                                   column_mappings: Optional[Dict[str, List[str]]] = None) -> Dict[str, Any]:
        """Automatically correct invalid column references in tool arguments.
        
        If a tool parameter expects a column name but receives an invalid one,
        try to intelligently map it to an actual column using pre-built mappings.
        """
        if not dataset_snapshot or not args:
            return args
        
        available_columns = dataset_snapshot.get('columns', [])
        if not available_columns:
            return args
        
        # Use provided mappings or build fallback
        if not column_mappings:
            column_mappings = self._build_column_mappings_fallback(dataset_snapshot)
        
        corrected_args = args.copy()
        # Relevance-aware preferences (final guard)
        relevance_scores: Dict[str, int] = dataset_snapshot.get('relevance_scores', {}) or {}
        numeric_cols = dataset_snapshot.get('numeric_columns', []) or []
        categorical_cols = dataset_snapshot.get('categorical_columns', []) or []
        datetime_cols = dataset_snapshot.get('datetime_columns', []) or []

        def _pref_cols_for_param(param_lower: str) -> List[str]:
            if 'date' in param_lower or 'time' in param_lower or 'datetime' in param_lower:
                base = [c for c in datetime_cols if c in available_columns]
            elif 'numeric' in param_lower or 'value' in param_lower or 'target' in param_lower:
                base = [c for c in numeric_cols if c in available_columns]
            elif 'category' in param_lower or 'categorical' in param_lower or 'group' in param_lower:
                base = [c for c in categorical_cols if c in available_columns]
            elif param_lower == 'columns':
                # Generic multi-column arg: prefer numeric for analytical operations like correlations
                base = [c for c in numeric_cols if c in available_columns] or [c for c in available_columns]
            else:
                base = [c for c in available_columns]
            # Sort by relevance (desc), keep only score>=4 first; if empty, fallback to all sorted by relevance
            high = [c for c in base if relevance_scores.get(c, 0) >= 4]
            if high:
                return sorted(high, key=lambda x: relevance_scores.get(x, 0), reverse=True)
            return sorted(base, key=lambda x: relevance_scores.get(x, 0), reverse=True)
        
        # Common parameter names that expect column names
        column_param_patterns = [
            'datetime_column', 'date_column', 'temporal_column', 'time_column',
            'text_column', 'category_column', 'categorical_column', 'group_column',
            'numeric_column', 'value_column', 'target_column', 'location_column',
            'incident_type_column', 'columns'
        ]
        
        for param_name, param_value in args.items():
            # Check if this looks like a column parameter
            param_lower = param_name.lower()
            is_column_param = any(pattern in param_lower for pattern in column_param_patterns)
            
            if not is_column_param:
                continue
            
            # Handle list of columns (e.g., columns parameter)
            if isinstance(param_value, list):
                corrected_list: List[str] = []
                for col in param_value:
                    if isinstance(col, str) and col not in available_columns:
                        # Try to find best match
                        best_match = self._find_best_column_match(col, available_columns, column_mappings, param_name)
                        if best_match:
                            self._logger.warning(f"Auto-correcting column '{col}' → '{best_match}' for {tool_name}.{param_name}")
                            corrected_list.append(best_match)
                        else:
                            corrected_list.append(col)  # Keep original if no match
                    else:
                        corrected_list.append(col)
                # Final guard: keep only available, unique; prefer high-relevance ordering
                uniq = []
                seen = set()
                for c in corrected_list:
                    if isinstance(c, str) and c in available_columns and c not in seen:
                        uniq.append(c)
                        seen.add(c)
                if not uniq:
                    # Fill with preferred columns for this param
                    preferred = _pref_cols_for_param(param_lower)
                    # Default caps: correlation/columns-like lists up to 12; else up to 5
                    cap = 12 if param_lower == 'columns' else 5
                    uniq = preferred[:cap]
                else:
                    # Reorder by relevance (desc), keep size reasonable
                    uniq.sort(key=lambda x: relevance_scores.get(x, 0), reverse=True)
                    if param_lower == 'columns' and len(uniq) > 12:
                        uniq = uniq[:12]
                corrected_args[param_name] = uniq
            
            # Handle single column string
            elif isinstance(param_value, str) and param_value not in available_columns:
                best_match = self._find_best_column_match(param_value, available_columns, column_mappings, param_name)
                if best_match:
                    self._logger.warning(f"Auto-correcting column '{param_value}' → '{best_match}' for {tool_name}.{param_name}")
                    corrected_args[param_name] = best_match
                else:
                    # Choose top preferred by relevance for this param as a conservative fallback
                    preferred = _pref_cols_for_param(param_lower)
                    if preferred:
                        self._logger.warning(f"Auto-selecting high-relevance column '{preferred[0]}' for {tool_name}.{param_name}")
                        corrected_args[param_name] = preferred[0]
        
        return corrected_args
    
    def _find_best_column_match(self, invalid_col: str, available_columns: List[str], 
                                 column_mappings: Dict[str, List[str]], param_name: str) -> Optional[str]:
        """Find the best matching column for an invalid column reference.
        
        Strategy:
        1. Check if param_name has suggested mappings (e.g., datetime_column → crash_date)
        2. Fuzzy match based on similarity
        3. Return None if no good match
        """
        # Strategy 1: Use pre-built mappings
        if param_name in column_mappings and column_mappings[param_name]:
            return column_mappings[param_name][0]  # Return top suggestion
        
        # Strategy 2: Fuzzy matching - find similar column names
        invalid_lower = invalid_col.lower()
        
        # Exact case-insensitive match
        for col in available_columns:
            if col.lower() == invalid_lower:
                return col
        
        # Partial match - the invalid column is contained in an available column
        for col in available_columns:
            if invalid_lower in col.lower() or col.lower() in invalid_lower:
                return col
        
        # Keyword-based matching for common patterns
        datetime_keywords = ['date', 'time', 'datetime', 'timestamp', 'crash', 'incident', 'created']
        text_keywords = ['desc', 'description', 'note', 'comment', 'text', 'narrative']
        
        if any(kw in invalid_lower for kw in datetime_keywords):
            for col in available_columns:
                if any(kw in col.lower() for kw in datetime_keywords):
                    return col
        
        if any(kw in invalid_lower for kw in text_keywords):
            for col in available_columns:
                if any(kw in col.lower() for kw in text_keywords):
                    return col
        
        return None  # No good match found

    async def execute_step(self, state: AnalysisPipelineState) -> Command:
        """Execute exactly one planned step and record the result."""
        # Early exit: if synthesis already finalized this run, stop cleanly.
        try:
            if getattr(self, "_single_run_guard_enabled", False) and getattr(self, "_single_run_completed", False):
                return Command(update={}, goto=END)
        except Exception:
            pass
        self._log_node("execute_step", state)
        # Global cap: finalize if too many steps have executed across the run
        try:
            self._steps_executed += 1
            if self._steps_executed > getattr(self, "_max_total_steps", 60):
                self._logger.warning("Global step cap reached - finalizing run to avoid runaway execution")
                return Command(update={"finalize": True}, goto="synthesis")
        except Exception:
            pass
        # SINGLE RUN GUARD: If synthesis already ran, hard-stop this run.
        if getattr(self, "_single_run_guard_enabled", False) and getattr(self, "_single_run_completed", False):
            # Keep log concise only for non-insights_only to preserve observability
            if str(getattr(state, "output_mode", "")) != "insights_only":
                self._logger.info("execute_step: single-run guard active; routing to END")
            return Command(update={}, goto=END)
        # If finalization requested, go straight to synthesis
        try:
            if getattr(state, "finalize", False):
                self._logger.info("execute_step: finalize flag detected, routing to synthesis to end")
                return Command(update={}, goto="synthesis")
        except Exception:
            pass
        # If no plan exists, short-circuit straight to synthesis to avoid empty loops
        if len(state.plan_steps or []) == 0:
            return Command(update={}, goto="synthesis")
        if state.next_step_index >= len(state.plan_steps):
            return Command(update={}, goto="interpret_results")

        step = state.plan_steps[state.next_step_index]
        tool_name = step.get("tool")
        args = step.get("args") or {}
        why = step.get("why") or ""

        guard_warnings: List[str] = []

        # Guardrail Option 1/2: tool-call firewall + path/output-dir enforcement (centralized)
        policy = None
        try:
            cfg = self._get_config()
            analysis_dir = cfg.analysis_agent_dir
            candidates = []
            try:
                candidates.append(analysis_dir.parents[3])
            except Exception:
                pass
            try:
                candidates.append(analysis_dir.parents[2])
            except Exception:
                pass
            repo_root = next((c for c in candidates if isinstance(c, Path) and c.exists()), analysis_dir)
            safe_output_dir = cfg.reports_path
            safe_output_dir.mkdir(parents=True, exist_ok=True)
            policy = build_default_policy(
                repo_root=repo_root,
                allowed_roots=[repo_root, repo_root / "agent_system", cfg.analysis_agent_dir, cfg.reports_path, cfg.temp_datasets_path],
                forced_output_dir=safe_output_dir,
            )
        except Exception:
            policy = None

        # TRACE: log selected step and inputs
        try:
            if self._trace_enabled:
                self._logger.info("TRACE_PLAN_STEP::" + json.dumps({
                    "index": state.next_step_index,
                    "tool": tool_name,
                    "args": args,
                    "why": why,
                }, ensure_ascii=True))
        except Exception:
            pass
        
        # AUTO-CORRECT COLUMN ARGUMENTS: Fix invalid column references before execution
        if hasattr(state, 'preprocess_profile') and state.preprocess_profile:
            try:
                # Reuse column mappings from planner (stored in state) for consistency
                stored_mappings = getattr(state, 'column_mappings', None)
                corrected_args = self._auto_correct_column_args(
                    tool_name, args, state.preprocess_profile, stored_mappings
                )
                if corrected_args != args:
                    self._logger.info(f"execute_step: Auto-corrected args for {tool_name}")
                    args = corrected_args
            except Exception as e:
                self._logger.warning(f"execute_step: Auto-correction failed for {tool_name}: {e}")
        
        if not tool_name or tool_name not in self._tool_map:
            skip_event = {
                "tool": tool_name or "unknown",
                "args": args,
                "output": f"Skipped: unknown tool in step {state.next_step_index}",
                "result": f"Skipped: unknown tool in step {state.next_step_index}",
                "artifact": None,
                "timestamp": datetime.now().isoformat(),
                "status": "skip",
                "why": why,
            }
            next_index = state.next_step_index + 1
            updates = {
                "tool_transcript": [skip_event],
                "next_step_index": next_index,
                "telemetry": {"tool_calls": 1, "tool_errors": 0},
            }
            if next_index < len(state.plan_steps):
                return Command(update=updates, goto="execute_step")
            return Command(update=updates, goto="interpret_results")

        tool = self._tool_map[tool_name]

        if policy is not None:
            try:
                args, guard_warnings = policy.sanitize_args(str(tool_name), dict(args), tool_obj=tool)
            except GuardrailViolation as gv:
                err_msg = f"Security policy blocked tool call '{tool_name}': {gv}"
                block_event = {
                    "tool": tool_name or "unknown",
                    "args": args,
                    "output": err_msg,
                    "result": err_msg,
                    "artifact": None,
                    "timestamp": datetime.now().isoformat(),
                    "status": "error",
                    "why": why,
                }
                updates = {
                    "tool_transcript": [block_event],
                    "telemetry": {"tool_calls": 1, "tool_errors": 1},
                    "warnings": [err_msg],
                }
                # Ask planner to replan around the blocked call.
                try:
                    updates["messages"] = [HumanMessage(content=err_msg + " Please replan using safe tools/paths.")]
                except Exception:
                    pass
                return Command(update=updates, goto="planner")
        result_text = ""
        artifacts_found: List[str] = []
        # Snapshot plots directory before tool runs to detect newly created files
        try:
            import time as _time
        except Exception:
            _time = None
        start_ts = _time.time() if _time else None
        pre_plots: set[str] = set()
        plots_dir = None
        try:
            cfg_snap = self._get_config()
            plots_dir = cfg_snap.plots_path
            if plots_dir and plots_dir.exists():
                pre_plots = {str(p) for p in plots_dir.glob("*.png")}
        except Exception:
            plots_dir = None
        try:
            # Phase 3 streaming: emit tool start as a custom event
            try:
                self._emit_custom_progress(
                    {
                        "type": "progress",
                        "event": "tool_start",
                        "run_id": self._current_run_id or "-",
                        "node": "execute_step",
                        "tool": str(tool_name),
                        "step_index": int(getattr(state, "next_step_index", 0) or 0),
                        "ts": datetime.now().isoformat(),
                    }
                )
            except Exception:
                pass
            call_result = None
            # Prefer LangChain's async API when available
            if hasattr(tool, 'ainvoke') and callable(getattr(tool, 'ainvoke')):
                call_result = await tool.ainvoke(args)
            elif hasattr(tool, 'invoke') and callable(getattr(tool, 'invoke')):
                call_result = tool.invoke(args)
            elif hasattr(tool, 'arun') and callable(getattr(tool, 'arun')):
                call_result = await tool.arun(args)
            elif hasattr(tool, 'run') and callable(getattr(tool, 'run')):
                call_result = tool.run(args)
            elif hasattr(tool, 'call') and callable(getattr(tool, 'call')):
                maybe = tool.call(args)
                call_result = await maybe if inspect.isawaitable(maybe) else maybe
            else:
                raise TypeError(f"Tool '{tool_name}' has no compatible invoke/call interface")

            if hasattr(call_result, "content"):
                # Concatenate textual parts
                texts = []
                for c in getattr(call_result, "content", []) or []:
                    if hasattr(c, "text") and isinstance(c.text, str):
                        texts.append(c.text)
                result_text = "\n".join(texts).strip()
            else:
                result_text = str(call_result)
            artifacts_found = self._parse_artifact_paths(result_text)
            # If no artifact string was returned, detect new plot files created by this tool
            try:
                if (not artifacts_found) and plots_dir and plots_dir.exists():
                    # Consider files created after the tool started or not present in snapshot
                    candidates = []
                    for f in plots_dir.glob("*.png"):
                        try:
                            mtime_ok = True
                            if start_ts and _time:
                                mtime_ok = (f.stat().st_mtime >= (start_ts - 1))
                            if (str(f) not in pre_plots) or mtime_ok:
                                candidates.append(f)
                        except Exception:
                            continue
                    if candidates:
                        # Sort newest first and take a few
                        try:
                            candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                        except Exception:
                            candidates = candidates
                        # Use top 3 to be safe; first will become primary
                        artifacts_found = [str(p) for p in candidates[:3]]
            except Exception:
                pass
        except Exception as e:
            result_text = f"Tool execution error: {e}"

        # Phase 3 streaming: emit tool end as a custom event (best-effort)
        try:
            tool_status = "ok"
            try:
                if isinstance(result_text, str) and result_text.strip().lower().startswith("tool execution error"):
                    tool_status = "error"
            except Exception:
                tool_status = "ok"
            self._emit_custom_progress(
                {
                    "type": "progress",
                    "event": "tool_end",
                    "run_id": self._current_run_id or "-",
                    "node": "execute_step",
                    "tool": str(tool_name),
                    "step_index": int(getattr(state, "next_step_index", 0) or 0),
                    "status": tool_status,
                    "ts": datetime.now().isoformat(),
                }
            )
        except Exception:
            pass

        # Choose a primary artifact if any (prefer absolute/existing paths over bare filenames)
        primary_artifact = None
        try:
            if artifacts_found:
                # Prefer entries that look like absolute paths or actually exist
                candidates = list(artifacts_found)
                abs_first = []
                others = []
                for p in candidates:
                    try:
                        pp = Path(p)
                        if pp.is_absolute() or pp.exists():
                            abs_first.append(str(pp))
                        else:
                            others.append(str(pp))
                    except Exception:
                        others.append(p)
                ordered = abs_first + others
                primary_artifact = ordered[0] if ordered else None
                # Normalize artifacts_found to prefer absolute path first
                artifacts_found = ordered
        except Exception:
            primary_artifact = artifacts_found[0] if artifacts_found else None

        # Ensure artifacts are placed under the configured reports directory for clean embedding
        try:
            if artifacts_found:
                cfg = self._get_config()
                report_dir: Path = cfg.reports_path
                report_dir.mkdir(parents=True, exist_ok=True)

                def _place_in_report_dir(pth: str) -> str:
                    try:
                        if not pth:
                            return pth
                        p = Path(pth)
                        # If it's just a filename and exists in report_dir, use that
                        if not p.is_absolute():
                            candidate = report_dir / p.name
                            if candidate.exists():
                                return str(candidate)
                            # If not existing elsewhere, leave as-is (may still be saved by tool under report_dir later)
                        # If file exists and is outside report_dir, copy it in
                        if p.exists():
                            try:
                                # Already under report_dir
                                p_resolved = p.resolve()
                                if str(report_dir.resolve()) in str(p_resolved.parent):
                                    return str(p_resolved)
                            except Exception:
                                pass
                            target = report_dir / p.name
                            # Avoid collisions by prefixing run id
                            if target.exists():
                                run_id = self._current_run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
                                target = report_dir / f"{run_id}_{p.name}"
                            try:
                                shutil.copy2(str(p), str(target))
                                return str(target)
                            except Exception:
                                # If copy fails, fall back to original
                                return str(p)
                        return pth
                    except Exception:
                        return pth

                # Map all artifacts into report_dir when possible
                artifacts_mapped = [_place_in_report_dir(a) for a in artifacts_found]
                artifacts_found = artifacts_mapped
                # Update primary artifact accordingly
                if primary_artifact:
                    primary_artifact = _place_in_report_dir(primary_artifact)
        except Exception:
            # Non-fatal: continue without relocating artifacts
            pass

        # Adaptive response analysis: persist raw output and summarize if very large
        summarized_text = result_text
        raw_output_path: Optional[str] = None
        try:
            threshold = int(os.getenv("ANALYSIS_RESPONSE_ANALYSIS_THRESHOLD", "5000"))
        except Exception:
            threshold = 5000
        try:
            max_summary = int(os.getenv("ANALYSIS_RESPONSE_ANALYSIS_MAX_CHARS", "1200"))
        except Exception:
            max_summary = 1200

        if isinstance(result_text, str) and len(result_text) > threshold and not result_text.startswith("Tool execution error"):
            # Persist raw text to an artifact file for full-fidelity traceability
            try:
                raw_output_path = self._save_raw_tool_output(tool_name, result_text)
            except Exception:
                raw_output_path = None
            # Summarize to relevant content using auxiliary LLM
            try:
                summarized_text = await self._analyze_tool_response(
                    tool_name=tool_name,
                    tool_args=args,
                    raw_response=result_text,
                    instruction=state.instruction,
                    context_summary=state.context_summary,
                    max_chars=max_summary,
                )
            except Exception as e:
                # If summarization fails, fall back to trimmed raw text
                self._logger.warning(f"response_analysis failed for {tool_name}: {e}")
                summarized_text = result_text[:max_summary]
        
        # If raw output was saved, treat it as an artifact
        if raw_output_path:
            artifacts_found = artifacts_found or []
            artifacts_found.append(raw_output_path)
            if not primary_artifact:
                primary_artifact = raw_output_path

        # Attach basenames for downstream synthesis mapping
        try:
            artifacts_basenames = [Path(a).name for a in artifacts_found] if artifacts_found else []
        except Exception:
            artifacts_basenames = []

        # Determine event status: treat textual error returns as errors too
        status_val = "ok"
        try:
            if isinstance(result_text, str) and result_text.strip().lower().startswith("error"):
                status_val = "error"
            if isinstance(result_text, str) and result_text.strip().lower().startswith("tool execution error"):
                status_val = "error"
        except Exception:
            pass

        # Debuggability: when tool_errors increments, emit which tool/args failed.
        try:
            if status_val == "error":
                step_i = int(getattr(state, "next_step_index", -1) or -1)
                err_preview = ""
                try:
                    err_preview = str(result_text).replace("\r", " ").replace("\n", " ")[:300]
                except Exception:
                    err_preview = ""
                # Keep args short to avoid log spam.
                try:
                    args_preview = json.dumps(args, ensure_ascii=True)[:300]
                except Exception:
                    args_preview = str(args)[:300]
                self._logger.warning(
                    f"tool_error: tool={tool_name} step_index={step_i} args={args_preview} error={err_preview}"
                )
        except Exception:
            pass

        event = {
            "tool": tool_name,
            "args": args,
            "output": summarized_text[:4000],  # synthesis expects 'output'
            "result": summarized_text[:4000],
            "artifact": primary_artifact,
            "timestamp": datetime.now().isoformat(),
            "status": status_val,
            "why": why,
        }
        if raw_output_path:
            event["raw_output_artifact"] = raw_output_path
        if artifacts_basenames:
            # Lightweight linkage for viz-only synthesis and traceability
            event["artifacts"] = artifacts_basenames
        artifacts_delta = []
        if primary_artifact:
            artifacts_delta.append(primary_artifact)
        if artifacts_found:
            artifacts_delta.extend(artifacts_found[1:])

        next_index = state.next_step_index + 1
        next_index = state.next_step_index + 1
        # Defensive safety: ensure monotonic progress of next_step_index
        try:
            old_index = int(getattr(state, "next_step_index", 0) or 0)
            if next_index <= old_index:
                next_index = old_index + 1
        except Exception:
            pass
        # Also register artifacts on state for immediate availability
        try:
            for a in artifacts_delta:
                try:
                    state.register_artifact(a)
                except Exception:
                    pass
        except Exception:
            pass

        updates = {
            "tool_transcript": [event],
            "artifact_log": artifacts_delta,
            "next_step_index": next_index,
            "telemetry": {
                "tool_calls": 1,
                "tool_errors": 1 if event["status"] == "error" else 0,
            },
        }
        if guard_warnings:
            updates["warnings"] = list(guard_warnings)
        if next_index < len(state.plan_steps):
            return Command(update=updates, goto="execute_step")
        return Command(update=updates, goto="interpret_results")

    def _save_raw_tool_output(self, tool_name: str, content: str) -> str:
        """Save large raw tool output to a log artifact and return its path.

        Writes under logs/tool_outputs with timestamped filename. Returns a string path.
        """
        try:
            base = Path("logs") / "tool_outputs"
            base.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_tool = re.sub(r"[^a-zA-Z0-9_\-]", "_", tool_name or "tool")
            fname = f"{ts}_{safe_tool}.txt"
            fpath = base / fname
            with fpath.open("w", encoding="utf-8") as fh:
                fh.write(content)
            try:
                if self._trace_enabled:
                    self._logger.info("TRACE_TOOL_OUTPUT_SAVED::" + json.dumps({
                        "tool": tool_name,
                        "path": str(fpath),
                    }, ensure_ascii=True))
            except Exception:
                pass
            return str(fpath)
        except Exception as e:
            self._logger.warning(f"failed to save raw tool output: {e}")
            raise

    async def _analyze_tool_response(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        raw_response: str,
        instruction: str,
        context_summary: str,
        max_chars: int = 12000,
    ) -> str:
        """Use auxiliary LLM to extract task-relevant information from a long tool output.

        Returns a concise summary string (<= max_chars) focused on data directly relevant
        to the current instruction and context.
        """
        # TRACE: log tool input before summarization
        try:
            if self._trace_enabled:
                self._logger.info("TRACE_TOOL_INPUT::" + json.dumps({
                    "tool": tool_name,
                    "args": tool_args,
                }, ensure_ascii=True))
        except Exception:
            pass
        llm = self._aux_llm or self._llm
        sys = SystemMessage(content=(
            "You are a precise assistant. Extract only the information directly relevant to the user's task. "
            "Keep exact numbers and key facts. Remove metadata, IDs, and unrelated sections. Return plain text only."
        ))
        prompt = (
            f"Task: {instruction}\n\n"
            f"Context: {context_summary}\n\n"
            f"Tool: {tool_name}\n"
            f"Args: {json.dumps(tool_args, ensure_ascii=False)}\n\n"
            "Full Tool Response (may be very long):\n"
            f"{raw_response}\n\n"
            "Instructions:\n"
            "1) Keep only content that answers or advances the task.\n"
            "2) Include concrete numbers, distributions, and named entities relevant to the task.\n"
            "3) Keep error messages and warnings if present.\n"
            f"4) Max output length: {max_chars} characters."
        )
        user = HumanMessage(content=prompt)
        try:
            resp = await llm.ainvoke([sys, user])
            text = resp.content if isinstance(resp, AIMessage) else str(resp)
        except Exception as e:
            # Bubble up to caller; they'll fallback to trimming
            raise RuntimeError(f"aux LLM summarization failed: {e}")
        return (text or "").strip()[:max_chars]

    # ----------------------------
    # Profile persistence helpers
    # ----------------------------
    def _dataset_key(self, dataset_path: str) -> str:
        try:
            return Path(dataset_path).stem.replace(" ", "_")
        except Exception:
            return "dataset"

    def _profile_file(self, dataset_path: str) -> Path:
        # Store dataset profiles alongside reports under the agent's report directory
        try:
            base = self._get_config().reports_path / "metadata"
        except Exception:
            # Fallback to a path relative to this module if config is unavailable
            base = Path(__file__).resolve().parent.parent / "reportDemo" / "metadata"
        try:
            base.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        key = self._dataset_key(dataset_path or "dataset")
        return base / f"{key}_profile.json"

    def _load_persisted_profile(self, dataset_path: Optional[str]) -> Optional[Dict[str, Any]]:
        try:
            if not dataset_path:
                return None
            f = self._profile_file(dataset_path)
            if not f.exists():
                # Migrate legacy location (repository-root relative) if present
                try:
                    key = self._dataset_key(dataset_path)
                    legacy = Path("src/agents/analysis/reportDemo/metadata") / f"{key}_profile.json"
                    if legacy.exists():
                        try:
                            f.parent.mkdir(parents=True, exist_ok=True)
                        except Exception:
                            pass
                        try:
                            shutil.copy2(str(legacy), str(f))
                        except Exception:
                            # If copy fails, fall back to reading from legacy path
                            f = legacy
                    else:
                        return None
                except Exception:
                    return None
            with f.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            return data if isinstance(data, dict) else None
        except Exception:
            return None

    def _save_persisted_profile(self, dataset_path: Optional[str], profile: Dict[str, Any]) -> None:
        try:
            if not dataset_path or not isinstance(profile, dict) or not profile.get("columns"):
                return
            f = self._profile_file(dataset_path)
            with f.open("w", encoding="utf-8") as fh:
                json.dump(profile, fh, indent=2)
        except Exception:
            pass

    async def interpret_results(self, state: AnalysisPipelineState) -> Command:
        """Interpret recent tool outputs into grounded insights and coverage.

        - Uses only the last 5 tool_transcript entries to avoid token bloat.
        - **FILTERS OUT ERROR ENTRIES** - only interprets successful tool outputs.
        - Prompts the LLM for strict JSON with verbatim evidence quotes.
        - Retries once if JSON parsing fails.
        - Appends valid insights and replaces coverage in state.
        """
        # Early exit: if synthesis already finalized this run, stop cleanly.
        try:
            if getattr(self, "_single_run_guard_enabled", False) and getattr(self, "_single_run_completed", False):
                return Command(update={}, goto=END)
        except Exception:
            pass
        self._log_node("interpret_results", state)
        # Enforce single-pass flow: if steps remain, do not synthesize yet.
        # Route back to execute_step until we've executed all planned steps.
        try:
            if state.next_step_index < len(state.plan_steps or []):
                return Command(update={}, goto="execute_step")
        except Exception:
            pass
        # SINGLE RUN GUARD: Avoid additional interpretation after final synthesis.
        if getattr(self, "_single_run_guard_enabled", False) and getattr(self, "_single_run_completed", False):
            self._logger.info("interpret_results: single-run guard active; routing to END")
            return Command(update={}, goto=END)
        
        # CRITICAL FIX: Filter out errors BEFORE interpretation
        # Simple mode can opt to use the full transcript instead of just the last N items
        try:
            interpret_use_all = str(os.getenv("ANALYSIS_INTERPRET_USE_ALL", "0")).lower() in ("1", "true", "yes")
        except Exception:
            interpret_use_all = False
        try:
            simple_mode = str(os.getenv("ANALYSIS_SIMPLE_MODE", "0")).lower() in ("1", "true", "yes")
        except Exception:
            simple_mode = False
        if simple_mode or interpret_use_all:
            full_tt = list(state.tool_transcript or [])
            recent_all = full_tt
            start_index = 0
        else:
            full_tt = list(state.tool_transcript or [])
            recent_all = full_tt[-5:] if full_tt else []
            start_index = max(0, len(full_tt) - len(recent_all))
        # Keep track of original transcript indices to avoid off-by-N errors after filtering
        recent_pairs = [(start_index + i, ev) for i, ev in enumerate(recent_all)]
        filtered_pairs = [(idx, ev) for (idx, ev) in recent_pairs if ev.get('status') != 'error']
        recent_indices = [idx for (idx, _) in filtered_pairs]
        recent = [ev for (_, ev) in filtered_pairs]
        
        # Build structured, compact JSON view of recent successful tool events
        events_compact: list[dict] = []
        for i, ev in enumerate(recent):
            try:
                events_compact.append({
                    "line_index": int(recent_indices[i]) if i < len(recent_indices) else i,
                    "tool": ev.get("tool"),
                    "args": ev.get("args"),
                    "output": (ev.get("output") or ev.get("result") or "")[:1200],
                    "artifact": ev.get("raw_output_artifact") or ev.get("artifact"),
                    "status": ev.get("status"),
                    "timestamp": ev.get("timestamp"),
                    "why": ev.get("why"),
                })
            except Exception:
                continue
        # Also retain a plain-text concatenation as auxiliary context (not authoritative)
        text_blocks: list[str] = []
        for ev in recent:
            try:
                block = (ev.get("output") or ev.get("result") or "").strip()
            except Exception:
                block = ""
            if block:
                text_blocks.append(block)
        tool_outputs_text = "\n---\n".join(text_blocks)

        # If nothing to interpret, proceed based on finalize flag
        if not tool_outputs_text and not events_compact:
            try:
                if getattr(state, "finalize", False):
                    return Command(update={}, goto="synthesis")
            except Exception:
                pass
            # If steps remain, keep executing; else hand off to synthesis
            try:
                steps_remain = state.next_step_index < len(state.plan_steps or [])
            except Exception:
                steps_remain = False
            return Command(update={
                "coverage": {"answers_query": False, "missing_info": []},
                "tool_events": [],
                "synthesis_input": {"insights": [], "coverage": {"answers_query": False, "missing_info": []}, "tool_events": []}
            }, goto=("execute_step" if steps_remain else "synthesis"))

        def _call_llm_once() -> Dict[str, Any]:
            prompt = build_interpretation_prompt(state, tool_outputs_text)
            system_msg = SystemMessage(content=prompt["system"])  # type: ignore[index]
            # Provide explicit guidance on coverage and missing_info to reduce loops
            try:
                available_tool_names = ", ".join(sorted(list(self._tool_map.keys())))
            except Exception:
                available_tool_names = ""
            coverage_rules = (
                "\n\n[Coverage rules]\n"
                "- answers_query must be true if the provided tool results already contain a direct answer or sufficient evidence to conclude.\n"
                "- missing_info must be a SHORT list (0-3) of concrete, actionable data needs that can be satisfied by the available tools.\n"
                "  Examples: 'run analyze_temporal_patterns with datetime_column=<col> to get hour-of-day distribution'.\n"
                "- Do NOT include vague requests like 'do more analysis', 'visualize', or 'dig deeper'.\n"
                "- Do NOT request steps that simply repeat tools already executed with the same inputs.\n"
                "- If nothing actionable is missing, set answers_query=true and missing_info=[].\n"
                f"- Only request items that our system can do using these tools: {available_tool_names}"
            )
            try:
                tools_json = json.dumps(events_compact, ensure_ascii=False)
            except Exception:
                tools_json = "[]"
            user_msg = HumanMessage(
                content=(
                    f"Instruction: {prompt['instruction']}\n\n"  # type: ignore[index]
                    "TOOLS_JSON (authoritative structured list; use this to extract evidence):\n"
                    f"{tools_json}\n\n"
                    "OPTIONAL_TEXT (auxiliary, may be truncated):\n"
                    f"{prompt['tool_results']}\n\n"  # type: ignore[index]
                    "Return JSON only." + coverage_rules
                )
            )
            return {"system": system_msg, "user": user_msg}

        # First attempt
        llm = self._llm
        msgs = _call_llm_once()
        raw1 = "{}"
        try:
            resp1 = await llm.ainvoke([msgs["system"], msgs["user"]])
            raw1 = resp1.content if isinstance(resp1, AIMessage) else str(resp1)
        except Exception as e:
            self._logger.warning(f"interpret_results: LLM call failed on first attempt: {e}")
        interpret_tokens = len(raw1) // 4
        payload = self._extract_json(raw1)

        # Validate structure; if invalid, do a single retry
        def _valid(p: Dict[str, Any]) -> bool:
            return isinstance(p, dict) and isinstance(p.get("insights"), list) and isinstance(p.get("coverage"), dict)

        if not _valid(payload):
            # Retry once with a stricter reminder
            try:
                reminder_user = HumanMessage(
                    content=(
                        "Reminder: Return ONLY JSON with keys insights (list) and coverage (object). "
                        "Evidence MUST be a literal substring from the provided tool results."
                    )
                )
                resp2 = await llm.ainvoke([msgs["system"], msgs["user"], reminder_user])
                raw2 = resp2.content if isinstance(resp2, AIMessage) else str(resp2)
                interpret_tokens += len(raw2) // 4
                payload = self._extract_json(raw2)
            except Exception as e:
                self._logger.warning(f"interpret_results: second attempt failed: {e}")

        # Final validation and state updates
        insights = []
        coverage = {"answers_query": False, "missing_info": []}
        # Default strict evidence flag (may be overridden inside insights logic)
        strict_env = str(os.getenv("ANALYSIS_STRICT_EVIDENCE", "0")).lower() in ("1", "true", "yes")
        if isinstance(payload, dict):
            # Coverage normalization
            cov = payload.get("coverage") or {}
            ans = bool(cov.get("answers_query")) if isinstance(cov, dict) else False
            missing = []
            if isinstance(cov, dict) and isinstance(cov.get("missing_info"), list):
                raw_missing_items: list[str] = []
                for item in cov.get("missing_info", []):
                    if isinstance(item, str) and item.strip():
                        raw_item = item.strip()
                        # Debug: capture raw (pre-truncation) missing_info for diagnosis.
                        if self._debug or self._trace_enabled:
                            raw_missing_items.append(raw_item)
                        missing.append(raw_item[:300])

                # Emit previews before any downstream invariants potentially clear the list.
                if (self._debug or self._trace_enabled) and raw_missing_items:
                    try:
                        max_items = _int_env("ANALYSIS_MISSING_INFO_DEBUG_MAX_ITEMS", 10)
                        max_chars = _int_env("ANALYSIS_MISSING_INFO_DEBUG_MAX_CHARS", 1200)
                        total = len(raw_missing_items)
                        shown = min(total, max_items)
                        self._logger.info(
                            f"interpret_results: missing_info raw pre-truncation items={total} showing={shown} (set ANALYSIS_MISSING_INFO_DEBUG_MAX_ITEMS/CHARS to tune)"
                        )
                        for idx, raw_item in enumerate(raw_missing_items[:shown]):
                            raw_len = len(raw_item)
                            snippet = raw_item[:max_chars].replace("\r", "").replace("\n", "\\n")
                            self._logger.info(f"interpret_results: missing_info_raw[{idx}] len={raw_len} {self._ascii_sanitize(snippet, keep_newlines=False)}")
                    except Exception:
                        pass
            # Invariant: if the model says we answered the query, nothing should be "missing".
            # Keeping both leads to ambiguous reflection decisions and can trigger runaway replans.
            if ans:
                missing = []
            coverage = {"answers_query": ans, "missing_info": missing}

            # Insights normalization + grounding check
            raw_insights = payload.get("insights") or []
            if isinstance(raw_insights, list):
                # Environment flags controlling inferred acceptance (strict_env already initialized)
                try:
                    max_inferred = int(os.getenv("ANALYSIS_MAX_INFERRED_INSIGHTS", "30"))
                except Exception:
                    max_inferred = 30
                inferred_count = 0
                # Columns list for heuristic matching (accept if statement references real column)
                dataset_columns = []
                try:
                    if isinstance(state.preprocess_profile, dict):
                        dataset_columns = list(state.preprocess_profile.get("columns") or [])
                except Exception:
                    dataset_columns = []
                columns_lower = {c.lower(): c for c in dataset_columns}
                numeric_pattern = re.compile(r"\b\d+(?:\.\d+)?%?\b")
                def _references_column(stmt: str) -> bool:
                    low = stmt.lower()
                    return any(col in low for col in columns_lower.keys())
                def _heuristic_inferred_ok(stmt: str, evidence: str) -> bool:
                    # Accept if: references a dataset column OR contains at least one number and is non-trivial length
                    if not stmt:
                        return False
                    if _references_column(stmt):
                        return True
                    if numeric_pattern.search(stmt):
                        return True
                    # fallback: evidence contains numbers
                    if numeric_pattern.search(evidence):
                        return True
                    return False
                for it in raw_insights:
                    try:
                        stmt = str(it.get("statement") or "").strip()
                        ev = str(it.get("evidence") or "").strip()
                        refs = it.get("refs") if isinstance(it.get("refs"), list) else []
                        
                        # CRITICAL FIX: Reject only ACTUAL error messages, not insights about missing data
                        if not stmt or not ev:
                            continue
                        
                        # Only reject if evidence contains actual error messages (not data insights)
                        error_patterns = ["Error:", "Exception:", "Traceback", "failed to", "could not"]
                        is_error_evidence = any(pattern in ev for pattern in error_patterns)
                        
                        # Reject insights where the statement is about tool errors (not about data)
                        # Allow: "No missing values", "0% missing", etc. (data insights)
                        # Reject: "Error executing tool", "Tool not found", etc. (error messages)
                        is_error_statement = any(pattern in stmt for pattern in ["Error:", "Exception:", "failed to execute", "tool not found"])
                        
                        if is_error_evidence or is_error_statement:
                            self._logger.debug(f"Dropping error-based insight: {stmt[:50]}")
                            continue
                        
                        # Evidence must be a literal substring of the tool outputs
                        # Build structured reference for synthesis verification
                        evidence_ref = None
                        if ev in tool_outputs_text:
                            # Find which tool transcript entry contains this evidence
                            for idx, tool_event in enumerate(recent):
                                tool_output = tool_event.get("output") or tool_event.get("result") or ""
                                if ev in tool_output:
                                    evidence_ref = {
                                        "source": "transcript",
                                        "line_index": int(recent_indices[idx]) if idx < len(recent_indices) else idx,
                                        "tool": tool_event.get("tool")
                                    }
                                    break
                            
                            if evidence_ref:
                                # Store with structured reference expected by synthesis verification
                                insight_obj = {
                                    "statement": stmt[:INSIGHT_TEXT_MAX_CHARS], 
                                    "evidence": ev[:INSIGHT_TEXT_MAX_CHARS],
                                    "evidence_snippet": ev[:INSIGHT_TEXT_MAX_CHARS],
                                    "evidence_ref": evidence_ref,
                                    "evidence_source": "transcript"
                                }
                                # Add refs if provided
                                if refs:
                                    insight_obj["refs"] = [str(r)[:100] for r in refs if isinstance(r, str)]
                                insights.append(insight_obj)
                                self._logger.debug(f"interpret_results: Added insight with ref (tool={evidence_ref['tool']}): {stmt[:60]}")
                            else:
                                self._debug_ascii(f"interpret_results: Evidence in text but couldn't build ref: {ev[:50]}")
                        else:
                            # Relaxed acceptance path when strict mode disabled
                            if (not strict_env) and stmt and ev and inferred_count < max_inferred and _heuristic_inferred_ok(stmt, ev):
                                inferred_insight = {
                                    "statement": stmt[:INSIGHT_TEXT_MAX_CHARS],
                                    "evidence": ev[:INSIGHT_TEXT_MAX_CHARS],
                                    "evidence_snippet": ev[:INSIGHT_TEXT_MAX_CHARS],
                                    "evidence_ref": None,
                                    "evidence_source": "inferred",
                                    "inferred": True,
                                    "verification": "heuristic"
                                }
                                if refs:
                                    inferred_insight["refs"] = [str(r)[:100] for r in refs if isinstance(r, str)]
                                insights.append(inferred_insight)
                                inferred_count += 1
                                self._debug_ascii(f"interpret_results: Accepted inferred insight (no literal evidence match): {stmt[:60]}")
                            else:
                                self._debug_ascii(f"interpret_results: Rejected insight - evidence not in tool outputs: {ev[:50]}")
                    except Exception as e:
                        self._debug_ascii(f"interpret_results: Error processing insight: {e}")
                        continue

        # Apply updates via delta merge
        updates = {
            "insights": insights,
            "coverage": coverage,
            "token_metrics": {"interpret": interpret_tokens},
            "telemetry": {"interpret_calls": 1},
        }
        self._logger.info(f"interpret_results: Generated {len(insights)} verified insights (strict_evidence={strict_env})")
        # Always route through reflect; it decides whether to execute, replan, or synthesize.
        return Command(update=updates, goto="reflect")

    async def delegate_code_interpreter(self, state: AnalysisPipelineState) -> Command:
        """Delegate a capability-gap task to an external Code Interpreter (CI) graph.

        PLACEHOLDER FOR INTEGRATION WITH CI
        This node is intentionally implemented as a *parent-graph placeholder* so we can:
        1) Route here from reflect() only when our current toolset cannot satisfy missing_info.
        2) Build a stable payload contract for the child CI graph.
        3) Append the child output as a tool-like event to tool_transcript so interpret_results()
           can run a second grounded pass (substring evidence checks).

        Child graph interface (TBD):
        - Input payload (fixed/stable):
          {instruction, missing_info, capability_gap, dataset_path, recent_tool_transcript_excerpt}
        - Output payload (black box for now):
          Must include at least `evidence_text: str`.
          We may later extend this with `summary_text`, `artifacts`, `structured_results`, etc.
        """

        self._log_node("delegate_code_interpreter", state)

        # PLACEHOLDER FOR INTEGRATION WITH CI
        # Feature flag: keep behavior inert unless explicitly enabled.
        try:
            enabled = str(os.getenv("ANALYSIS_ENABLE_CODE_INTERPRETER_DELEGATION", "0")).lower() in (
                "1",
                "true",
                "yes",
            )
        except Exception:
            enabled = False

        if not enabled:
            evidence_text = (
                "PLACEHOLDER FOR INTEGRATION WITH CI\n"
                "Delegation feature flag ANALYSIS_ENABLE_CODE_INTERPRETER_DELEGATION is disabled.\n"
                "No child CI graph was invoked."
            )
            event = {
                "tool": "delegate_code_interpreter",
                "args": {"enabled": False},
                "output": evidence_text[:4000],
                "result": evidence_text[:4000],
                "timestamp": datetime.now().isoformat(),
                "status": "ok",
                "why": "PLACEHOLDER FOR INTEGRATION WITH CI: feature disabled",
                "placeholder": True,
            }
            return Command(
                update={
                    "tool_transcript": [event],
                    "telemetry": {"delegation_calls": 1},
                },
                goto="interpret_results",
            )

        # --- Build child input payload (stable contract) ---
        try:
            missing_info = state.coverage.get("missing_info") if isinstance(state.coverage, dict) else []
        except Exception:
            missing_info = []
        missing_info = [m for m in (missing_info or []) if isinstance(m, str) and m.strip()]

        capability_gap = None
        try:
            capability_gap = getattr(state, "capability_gap", None)
        except Exception:
            capability_gap = None

        # Compact transcript excerpt to prevent token bloat.
        recent_excerpt: list[dict] = []
        try:
            transcript = list(state.tool_transcript or [])
            # Prefer last successful events as context for CI.
            ok_events = [(i, ev) for i, ev in enumerate(transcript) if isinstance(ev, dict) and ev.get("status") == "ok"]
            # PLACEHOLDER FOR INTEGRATION WITH CI
            # Keep this small and structured so it's safe to log/ship cross-graph.
            for idx, ev in ok_events[-5:]:
                recent_excerpt.append(
                    {
                        "line_index": idx,
                        "tool": ev.get("tool"),
                        "args": ev.get("args"),
                        "output": (ev.get("output") or ev.get("result") or "")[:1200],
                        "artifact": ev.get("raw_output_artifact") or ev.get("artifact"),
                        "timestamp": ev.get("timestamp"),
                    }
                )
        except Exception:
            recent_excerpt = []

        payload = {
            "instruction": getattr(state, "instruction", "") or "",
            "missing_info": missing_info,
            "capability_gap": capability_gap,
            "dataset_path": getattr(state, "dataset_path", "") or "",
            "recent_tool_transcript_excerpt": recent_excerpt,
        }

        # --- Invoke child graph via subprocess (avoids `src` import collisions) ---
        # PLACEHOLDER FOR INTEGRATION WITH CI (subprocess implementation)
        # The child repo is executed in a separate Python process so it can safely import its own
        # top-level `src` package without shadowing the parent's `src`.
        import subprocess
        import tempfile

        def _find_child_root() -> Optional[Path]:
            # 1) Explicit override
            override = os.getenv("ANALYSIS_CODE_INTERPRETER_CHILD_ROOT")
            if override:
                p = Path(override).expanduser()
                if p.exists():
                    return p

            # 2) Heuristic: locate DAA repo root from this file, then look for the sibling codeGen folder.
            try:
                here = Path(__file__).resolve()
                daa_root = None
                for parent in here.parents:
                    if (parent / "pyproject.toml").exists() and (parent / "src").exists():
                        daa_root = parent
                        break
                if daa_root is not None:
                    candidate = daa_root.parent / "codeGen" / "MCP_Tool_Code_Interpreter_Generator"
                    if candidate.exists():
                        return candidate
            except Exception:
                pass

            # 3) Fallback: common relative path from CWD
            try:
                candidate = Path.cwd().parent / "codeGen" / "MCP_Tool_Code_Interpreter_Generator"
                if candidate.exists():
                    return candidate
            except Exception:
                pass

            return None

        child_root = _find_child_root()
        if child_root is None:
            evidence_text = (
                "PLACEHOLDER FOR INTEGRATION WITH CI\n"
                "Subprocess mode enabled, but could not locate child repo root.\n"
                "Set env ANALYSIS_CODE_INTERPRETER_CHILD_ROOT to: <path>/MCP_Tool_Code_Interpreter_Generator"
            )
            event = {
                "tool": "delegate_code_interpreter",
                "args": payload,
                "output": evidence_text[:4000],
                "result": evidence_text[:4000],
                "timestamp": datetime.now().isoformat(),
                "status": "error",
                "why": "PLACEHOLDER FOR INTEGRATION WITH CI: child repo path missing",
                "placeholder": True,
            }
            return Command(
                update={
                    "tool_transcript": [event],
                    "errors": ["code_interpreter_child_root_not_found"],
                    "telemetry": {"delegation_calls": 1},
                },
                goto="interpret_results",
            )

        runner = child_root / "integration" / "subprocess_runner.py"
        if not runner.exists():
            evidence_text = (
                "PLACEHOLDER FOR INTEGRATION WITH CI\n"
                "Child subprocess runner not found at integration/subprocess_runner.py.\n"
                f"Expected: {str(runner)}"
            )
            event = {
                "tool": "delegate_code_interpreter",
                "args": payload,
                "output": evidence_text[:4000],
                "result": evidence_text[:4000],
                "timestamp": datetime.now().isoformat(),
                "status": "error",
                "why": "PLACEHOLDER FOR INTEGRATION WITH CI: runner missing",
                "placeholder": True,
            }
            return Command(
                update={
                    "tool_transcript": [event],
                    "errors": ["code_interpreter_subprocess_runner_missing"],
                    "telemetry": {"delegation_calls": 1},
                },
                goto="interpret_results",
            )

        # Build child initial state (mirrors codeGen/integration/mapper.py build_child_input)
        # Always include the full parent instruction so the child can act end-to-end.
        # Append missing_info as supplemental context when present.
        base_instruction = payload.get("instruction") or ""
        if missing_info:
            child_query = base_instruction
            if child_query:
                child_query += "\n\n"
            child_query += "Capability gaps / missing info:\n- " + "\n- ".join(missing_info)
        else:
            child_query = base_instruction
        data_path = payload.get("dataset_path") or ""
        try:
            data_path = str(Path(data_path).expanduser().resolve()) if data_path else ""
        except Exception:
            data_path = str(data_path or "")

        child_init: Dict[str, Any] = {
            "user_query": child_query,
            "data_path": data_path,
            "extracted_intent": None,
            "has_gap": False,
            "matched_tool": None,
            "tool_spec": None,
            "generated_code": None,
            "draft_path": None,
            "validation_result": None,
            "repair_attempts": 0,
            "execution_output": None,
            "draft_output_path": None,
            "promoted_tool": None,
            "errors": None,
            "task_id": None,
            "projected_tool_transcript": None,
            "projected_artifact_log": None,
            "projected_capability_gap": None,
            "projected_errors": None,
            "projected_warnings": None,
            "projected_final_artifacts": None,
            "messages": [],
        }

        # Timeout (seconds) for the child subprocess.
        try:
            timeout_sec = int(os.getenv("ANALYSIS_CI_SUBPROCESS_TIMEOUT_SEC", "300"))
        except Exception:
            timeout_sec = 300

        # Run child process with isolated import path.
        with tempfile.TemporaryDirectory(prefix="daa_ci_") as td:
            tmpdir = Path(td)
            in_path = tmpdir / "child_input.json"
            out_path = tmpdir / "child_output.json"
            in_path.write_text(json.dumps(child_init, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

            env = os.environ.copy()
            # Ensure child repo root is on PYTHONPATH so it can import its own `src`.
            env["PYTHONPATH"] = str(child_root)

            # Default child sandbox mode to subprocess unless explicitly overridden.
            # This avoids Docker assumptions during end-to-end runs.
            env.setdefault("TOOLGEN_SANDBOX_MODE", "subprocess")

            # Allow running the child in a dedicated venv to avoid dependency churn in the parent env.
            child_python = os.getenv("ANALYSIS_CODE_INTERPRETER_PYTHON")
            if not child_python:
                try:
                    # Prefer a local venv inside the child repo if present.
                    candidates = [
                        child_root / ".venv" / "Scripts" / "python.exe",  # Windows
                        child_root / ".venv" / "bin" / "python",  # POSIX
                        child_root / "venv" / "Scripts" / "python.exe",
                        child_root / "venv" / "bin" / "python",
                    ]
                    for c in candidates:
                        if c.exists():
                            child_python = str(c)
                            break
                except Exception:
                    child_python = None
            child_python = child_python or sys.executable

            cmd = [
                str(child_python),
                str(runner),
                "--input",
                str(in_path),
                "--output",
                str(out_path),
            ]

            try:
                proc = subprocess.run(
                    cmd,
                    cwd=str(child_root),
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=timeout_sec,
                )
            except subprocess.TimeoutExpired:
                evidence_text = (
                    "PLACEHOLDER FOR INTEGRATION WITH CI\n"
                    f"Child tool-generator subprocess timed out after {timeout_sec}s."
                )
                event = {
                    "tool": "delegate_code_interpreter",
                    "args": payload,
                    "output": evidence_text[:4000],
                    "result": evidence_text[:4000],
                    "timestamp": datetime.now().isoformat(),
                    "status": "error",
                    "why": "PLACEHOLDER FOR INTEGRATION WITH CI: subprocess timeout",
                    "placeholder": True,
                }
                return Command(
                    update={
                        "tool_transcript": [event],
                        "errors": ["code_interpreter_subprocess_timeout"],
                        "telemetry": {"delegation_calls": 1},
                    },
                    goto="interpret_results",
                )

            # Parse child output (prefer output file; include stderr in errors if missing)
            child_payload: Dict[str, Any] = {}
            try:
                if out_path.exists():
                    child_payload = json.loads(out_path.read_text(encoding="utf-8"))
            except Exception as e:
                child_payload = {"ok": False, "error": f"child_output_parse_failed: {e}"}

            child_ok = bool(child_payload.get("ok")) and proc.returncode == 0
            projected_tt = child_payload.get("projected_tool_transcript") or []
            projected_art = child_payload.get("projected_artifact_log") or []
            projected_gap = child_payload.get("projected_capability_gap")
            projected_errs = child_payload.get("projected_errors") or []
            projected_warns = child_payload.get("projected_warnings") or []
            projected_fa = child_payload.get("projected_final_artifacts") or {}

            # Ensure projected transcript outputs are strings so interpret_results can consume them.
            def _stringify(v: Any) -> str:
                if v is None:
                    return ""
                if isinstance(v, str):
                    return v
                try:
                    return json.dumps(v, ensure_ascii=False, default=str)
                except Exception:
                    return str(v)

            stamped_tt: List[Dict[str, Any]] = []
            now_iso = datetime.now().isoformat()
            for ev in projected_tt:
                if not isinstance(ev, dict):
                    continue
                out_val = ev.get("output")
                out_text = _stringify(out_val)
                stamped_tt.append(
                    {
                        "tool": str(ev.get("tool") or "ci_child"),
                        "args": ev.get("args") or {},
                        "output": out_text,
                        "result": out_text,
                        "timestamp": ev.get("timestamp") or now_iso,
                        "status": ev.get("status") or "ok",
                        "why": "CI subprocess: projected transcript",
                        "source": "ci_subprocess",
                    }
                )

            # Build an evidence_text summary from promoted output JSON (if present) so the next
            # interpret_results pass has quotable substrings.
            evidence_lines: List[str] = [
                "PLACEHOLDER FOR INTEGRATION WITH CI (subprocess)",
                f"child_ok={child_ok}",
            ]
            promoted_tool = None
            try:
                if isinstance(projected_fa, dict):
                    promoted_tool = projected_fa.get("promoted_tool")
            except Exception:
                promoted_tool = None
            if isinstance(promoted_tool, dict):
                evidence_lines.append(f"promoted_tool_name={promoted_tool.get('name')}")
                evidence_lines.append(f"promoted_tool_path={promoted_tool.get('path')}")
                evidence_lines.append(f"promoted_output_path={promoted_tool.get('output_path')}")

                op = promoted_tool.get("output_path")
                if isinstance(op, str) and op:
                    try:
                        op_path = Path(op)
                        if not op_path.is_absolute():
                            op_path = child_root / op_path
                        if op_path.exists():
                            out_obj = json.loads(op_path.read_text(encoding="utf-8"))
                            evidence_lines.append("output_json_preview=")
                            if isinstance(out_obj, dict):
                                # Emit up to 20 scalar-ish key/value pairs
                                shown = 0
                                for k, v in out_obj.items():
                                    if shown >= 20:
                                        break
                                    # keep concise but quotable
                                    evidence_lines.append(f"- {k}: {v}")
                                    shown += 1
                            else:
                                evidence_lines.append(str(out_obj)[:1200])
                    except Exception as _e:
                        evidence_lines.append(f"output_json_read_failed: {_e}")

            if proc.stdout:
                evidence_lines.append("child_stdout_tail=")
                evidence_lines.append(proc.stdout[-800:])
            if proc.stderr:
                evidence_lines.append("child_stderr_tail=")
                evidence_lines.append(proc.stderr[-800:])

            evidence_text = "\n".join([ln for ln in evidence_lines if ln is not None])

            delegate_event = {
                "tool": "delegate_code_interpreter",
                "args": {
                    **payload,
                    "child_root": str(child_root),
                    "timeout_sec": timeout_sec,
                    "returncode": proc.returncode,
                },
                "output": evidence_text[:4000],
                "result": evidence_text[:4000],
                "timestamp": datetime.now().isoformat(),
                "status": "ok" if child_ok else "error",
                "why": "PLACEHOLDER FOR INTEGRATION WITH CI: ran tool-generator as subprocess",
                "placeholder": True,
            }

            # Build LangGraph-safe updates.
            updates: Dict[str, Any] = {
                "telemetry": {"delegation_calls": 1},
            }
            if stamped_tt:
                updates["tool_transcript"] = stamped_tt
            # Always append the final delegate event last so it lands in the last-5 window.
            updates.setdefault("tool_transcript", [])
            updates["tool_transcript"].append(delegate_event)

            if projected_art:
                updates["artifact_log"] = [str(p) for p in projected_art if p]
            if projected_warns:
                updates["warnings"] = [str(w) for w in projected_warns if w]

            # Include stderr/returncode as an error when the child failed.
            if not child_ok:
                err_lines = []
                if child_payload.get("error"):
                    err_lines.append(str(child_payload.get("error")))
                if projected_errs:
                    err_lines.extend([str(e) for e in projected_errs if e])
                if proc.stderr:
                    err_lines.append(f"child_stderr_tail: {proc.stderr[-400:]}")
                if err_lines:
                    updates["errors"] = err_lines[:50]
            else:
                # Even on success, propagate any warnings/errors emitted by the child.
                if projected_errs:
                    updates["errors"] = [str(e) for e in projected_errs if e]

            # Replace-reducer fields
            updates["capability_gap"] = projected_gap

            if isinstance(projected_fa, dict) and projected_fa:
                # Parent final_artifacts uses replace-latest dict; merge with existing first.
                merged_fa = dict(state.final_artifacts or {})
                merged_fa.update(projected_fa)
                updates["final_artifacts"] = merged_fa

            return Command(update=updates, goto="interpret_results")

    async def reflect(self, state: AnalysisPipelineState) -> Command:
        """Decide whether to synthesize or re-plan with refinement context.

        Rules:
        - If answers_query is True OR missing_info is empty: go to synthesis.
        - Else: store missing_info into state.refinement_request and re-route to planner.
        """
        self._log_node("reflect", state)
        # SINGLE RUN GUARD: Do not attempt further reflection if synthesis finalized.
        if getattr(self, "_single_run_guard_enabled", False) and getattr(self, "_single_run_completed", False):
            # Route back through synthesis so callers still receive a PipelineOutputState
            # (synthesis will return cached output and persist_cleanup will skip duplicates).
            self._logger.info("reflect: single-run guard active; routing to synthesis")
            return Command(update={}, goto="synthesis")
        # Feature flag to disable reflection (single-pass runs for speed)
        disable_reflect = str(os.getenv("ANALYSIS_DISABLE_REFLECT", "0")).lower() in ("1", "true", "yes")
        disable_reason = "ANALYSIS_DISABLE_REFLECT" if disable_reflect else None
        # Simple mode forces reflection off to ensure single-pass behavior
        try:
            if str(os.getenv("ANALYSIS_SIMPLE_MODE", "0")).lower() in ("1", "true", "yes"):
                disable_reflect = True
                disable_reason = disable_reason or "ANALYSIS_SIMPLE_MODE"
        except Exception:
            pass
        if disable_reflect:
            self._logger.info(f"reflect: Reflection disabled by {disable_reason or 'feature flag'}; routing directly")
            # Even when reflection is disabled, perform a cheap deterministic recovery:
            # if tools errored (and nothing succeeded yet), replan using valid columns.
            try:
                transcript = state.tool_transcript or []
                has_ok = any((e.get("status") == "ok") for e in transcript if isinstance(e, dict))
                has_errors = any((e.get("status") == "error") for e in transcript if isinstance(e, dict))
                current_round = getattr(state, "refinement_round", 0) or 0
                max_ref = 1
                try:
                    max_ref = int(os.getenv("ANALYSIS_MAX_REFINEMENTS", "1"))
                except Exception:
                    max_ref = 1

                if (not has_ok) and has_errors and current_round < max_ref:
                    valid_columns: list[str] = []
                    if state.preprocess_profile and isinstance(state.preprocess_profile, dict):
                        valid_columns = list(state.preprocess_profile.get("columns") or [])

                    refinement_msg = (
                        "Previous plan failed due to tool errors (likely invalid column references). "
                        "Replan using ONLY real dataset columns. Do NOT invent column names."
                    )
                    if valid_columns:
                        col_list = ", ".join(valid_columns[:15])
                        if len(valid_columns) > 15:
                            col_list += f", ... ({len(valid_columns)} total)"
                        refinement_msg += f" Available columns: {col_list}."

                    updates = {
                        "refinement_request": [refinement_msg],
                        "refinement_round": current_round + 1,
                        "telemetry": {"refinement_rounds": 1},
                    }
                    self._logger.info(
                        f"reflect: disabled-mode recovery replanning requested (round {current_round + 1}/{max_ref})"
                    )
                    return Command(update=updates, goto="planner")
            except Exception as e:
                self._logger.warning(f"reflect: disabled-mode recovery check failed: {e}")

            # If steps remain execute them, else go straight to synthesis
            remaining = len(state.plan_steps or []) - state.next_step_index
            if remaining > 0:
                return Command(update={}, goto="execute_step")
            return Command(update={}, goto="synthesis")
        # Honor finalize flag immediately
        try:
            if getattr(state, "finalize", False):
                return Command(update={}, goto="synthesis")
        except Exception:
            return Command(update={}, goto="synthesis")

        # Early-exit guards to prevent loops when planner/tools cannot progress
        try:
            transcript = state.tool_transcript or []
            has_ok = any((e.get("status") == "ok") for e in transcript if isinstance(e, dict))
            has_errors = any((e.get("status") == "error") for e in transcript if isinstance(e, dict))
            plan_len = len(state.plan_steps or [])
            current_round = getattr(state, "refinement_round", 0) or 0
            
            # NEW RECOVERY LOGIC: If we have errors but no successes, try replanning with valid columns
            if not has_ok and has_errors and current_round < 2:
                # Extract valid columns from preprocess_profile for refinement
                valid_columns = []
                if state.preprocess_profile and isinstance(state.preprocess_profile, dict):
                    valid_columns = state.preprocess_profile.get('columns', [])
                
                if valid_columns:
                    # Build column-aware refinement message
                    col_list = ', '.join(valid_columns[:15])  # First 15 columns
                    if len(valid_columns) > 15:
                        col_list += f", ... ({len(valid_columns)} total)"
                    
                    refinement_msg = (
                        f"Previous plan failed due to tool errors (likely invalid column references). "
                        f"The dataset has these columns: {col_list}. "
                        f"Replan using ONLY these columns. Do NOT invent column names."
                    )
                    
                    updates = {
                        "refinement_request": [refinement_msg],
                        "refinement_round": current_round + 1,
                        # Do not reset next_step_index here; planner will decide based on plan_hash
                        "telemetry": {"refinement_rounds": 1}
                    }
                    self._logger.info(f"Requesting column-aware replanning (round {current_round + 1}): {len(valid_columns)} columns available")
                    return Command(update=updates, goto="planner")
            
            # Original early-exit: only after 2 refinement attempts
            if (not has_ok) and (plan_len == 0 or current_round >= 2):
                self._logger.warning(f"No successful outputs after {current_round} refinement rounds, proceeding to synthesis")
                return Command(update={}, goto="synthesis")
                
        except Exception as e:
            self._logger.error(f"reflect: error in recovery logic: {e}")
            # On any error evaluating guards, fall back to synthesis rather than looping
            return Command(update={}, goto="synthesis")
        try:
            answered = bool(state.coverage.get("answers_query")) if isinstance(state.coverage, dict) else False
            missing = state.coverage.get("missing_info") if isinstance(state.coverage, dict) else []
            missing = [m for m in (missing or []) if isinstance(m, str) and m.strip()]
        except Exception:
            answered = False
            missing = []

        self._logger.info(f"reflect: answered={answered}, missing_count={len(missing)}, refine_round={getattr(state, 'refinement_round', 0)}")

        # Preserve the raw missing list for potential capability-gap diagnosis
        missing_raw: list[str] = list(missing) if isinstance(missing, list) else []

        # Optional fallback: if the user explicitly requested generating/promoting a new tool,
        # and we have not produced one, force a single delegation attempt even if the LLM
        # coverage claims the query is answered.
        try:
            force_delegate = str(os.getenv("ANALYSIS_FORCE_DELEGATION_FOR_TOOL_REQUEST", "0")).lower() in (
                "1",
                "true",
                "yes",
            )
        except Exception:
            force_delegate = False

        if force_delegate:
            try:
                delegation_enabled = str(os.getenv("ANALYSIS_ENABLE_CODE_INTERPRETER_DELEGATION", "0")).lower() in (
                    "1",
                    "true",
                    "yes",
                )
            except Exception:
                delegation_enabled = False
            try:
                max_delegations = int(os.getenv("ANALYSIS_MAX_CI_DELEGATIONS", "1"))
            except Exception:
                max_delegations = 1
            try:
                attempts = int(getattr(state, "delegation_attempts", 0) or 0)
            except Exception:
                attempts = 0

            # The pipeline may rewrite/condense state.instruction during ingest. To preserve the
            # original intent, also scan the most recent user message content.
            parts: list[str] = []
            try:
                parts.append(str(getattr(state, "instruction", "") or ""))
            except Exception:
                pass
            try:
                msgs = list(getattr(state, "messages", []) or [])
                if msgs:
                    parts.append(str(getattr(msgs[-1], "content", "") or ""))
            except Exception:
                pass
            instr = "\n".join([p for p in parts if p]).lower()
            wants_tool = ("promote" in instr and "tool" in instr) or ("create" in instr and "tool" in instr)

            has_promoted = False
            try:
                fa = getattr(state, "final_artifacts", None)
                if isinstance(fa, dict) and isinstance(fa.get("promoted_tool"), dict):
                    has_promoted = True
            except Exception:
                has_promoted = False

            if delegation_enabled and wants_tool and (not has_promoted) and attempts < max_delegations:
                forced_missing = [
                    "Need code interpreter tool generation; current tools cannot create/promote new tool files"
                ]
                updates = {
                    "finalize": False,
                    "delegation_attempts": attempts + 1,
                    "coverage": {"answers_query": False, "missing_info": forced_missing},
                    "telemetry": {"delegation_rounds": 1, "delegation_forced": 1},
                }
                self._logger.info(
                    f"reflect: forced delegation for tool request (attempt {attempts + 1}/{max_delegations})"
                )
                return Command(update=updates, goto="delegate_code_interpreter")

        # Enforce actionable missing_info policy to avoid vague replanning loops
        try:
            require_actionable = str(os.getenv("ANALYSIS_REQUIRE_ACTIONABLE_MISSING", "1")).lower() in ("1", "true", "yes")
            max_missing = int(os.getenv("ANALYSIS_MAX_MISSING_ITEMS", "3"))
        except Exception:
            require_actionable = True
            max_missing = 3
        if require_actionable and isinstance(missing, list):
            filtered: list[str] = []
            seen_lower = set()
            available_tools = set((self._tool_map or {}).keys()) if hasattr(self, "_tool_map") and self._tool_map else set()
            dataset_columns = []
            try:
                if isinstance(state.preprocess_profile, dict):
                    dataset_columns = list(state.preprocess_profile.get("columns") or [])
            except Exception:
                dataset_columns = []
            cols_lower = {c.lower() for c in dataset_columns}

            # Build signatures of already executed tool calls to avoid exact repeats
            executed_signatures = set()
            try:
                for ev in (state.tool_transcript or []):
                    if not isinstance(ev, dict):
                        continue
                    tname = str(ev.get("tool") or "")
                    args_sig = ""
                    try:
                        args_sig = json.dumps(ev.get("args") or {}, sort_keys=True)
                    except Exception:
                        args_sig = str(ev.get("args") or {})
                    executed_signatures.add((tname, args_sig))
            except Exception:
                pass

            def looks_actionable(item: str) -> bool:
                if not item or len(item.strip()) < 10:
                    return False
                s = item.lower()
                # Vague phrases that shouldn't trigger replanning
                vague = ["more analysis", "dig deeper", "visualize", "explore further", "additional insights", "deeper analysis"]
                if any(v in s for v in vague):
                    return False
                # Prefer if it references a known tool
                if any(tool.lower() in s for tool in available_tools):
                    return True
                # Or if it references a real dataset column
                if any(col in s for col in cols_lower):
                    return True
                # Otherwise consider non-actionable
                return False

            # Simple duplicate/elaboration checks and cap
            for m_item in missing:
                try:
                    m_text = str(m_item).strip()
                except Exception:
                    continue
                if not m_text:
                    continue
                if not looks_actionable(m_text):
                    continue
                low = m_text.lower()
                if low in seen_lower:
                    continue
                seen_lower.add(low)
                filtered.append(m_text)
                if len(filtered) >= max_missing:
                    break

            # If nothing actionable remains, finalize or proceed to synthesis
            if not filtered:
                # Nothing actionable means the missing targets cannot be mapped to current tools/columns.
                # Diagnose whether this is a tool capability gap (needs new tool) vs. other root cause.
                try:
                    explain_gaps = str(os.getenv("ANALYSIS_EXPLAIN_TOOL_GAPS", "1")).lower() in ("1", "true", "yes")
                except Exception:
                    explain_gaps = True

                # PLACEHOLDER FOR INTEGRATION WITH CI
                # If enabled, delegate once to a CI graph to compute missing evidence not satisfiable
                # via the current MCP toolset. The CI graph's output will be appended to tool_transcript
                # as `evidence_text`, and we will route back to interpret_results for a second grounded pass.
                try:
                    delegation_enabled = str(os.getenv("ANALYSIS_ENABLE_CODE_INTERPRETER_DELEGATION", "0")).lower() in (
                        "1",
                        "true",
                        "yes",
                    )
                except Exception:
                    delegation_enabled = False
                try:
                    max_delegations = int(os.getenv("ANALYSIS_MAX_CI_DELEGATIONS", "1"))
                except Exception:
                    max_delegations = 1
                try:
                    attempts = int(getattr(state, "delegation_attempts", 0) or 0)
                except Exception:
                    attempts = 0

                # Compute capability-gap diagnosis if needed for delegation or for final reporting.
                diag = None
                if missing_raw and (explain_gaps or delegation_enabled):
                    try:
                        diag = await self._diagnose_tool_gap(state, missing_raw)
                    except Exception as _ge:
                        self._logger.warning(f"reflect: capability-gap diagnosis failed ({_ge})")

                if delegation_enabled and missing_raw and attempts < max_delegations:
                    updates: dict = {
                        "finalize": False,
                        "delegation_attempts": attempts + 1,
                        # Preserve missing_info for debug sidecars and for CI payload.
                        "coverage": {"answers_query": False, "missing_info": missing_raw[:10]},
                        "telemetry": {"delegation_rounds": 1},
                    }
                    if isinstance(diag, dict) and diag:
                        updates["capability_gap"] = diag
                    self._logger.info(
                        f"reflect: no actionable missing_info; delegating to CI (attempt {attempts + 1}/{max_delegations})"
                    )
                    return Command(update=updates, goto="delegate_code_interpreter")

                # Fallback: original behavior (finalize to synthesis to avoid loops)
                updates: dict = {"finalize": True}
                if isinstance(diag, dict) and diag:
                    updates["capability_gap"] = diag
                    updates["coverage"] = {"answers_query": False, "missing_info": missing_raw[:10]}
                self._logger.info("reflect: no actionable missing_info; finalizing to synthesis to avoid loops")
                return Command(update=updates, goto="synthesis")
            else:
                # Replace missing with filtered actionable items
                missing = filtered

        # Optional behavior: End immediately when the query is answered or no missing info remains,
        # even if there are still planned steps. This prevents long loops and multiple partial reports.
        try:
            end_on_answer = str(os.getenv("ANALYSIS_END_ON_ANSWER", "1")).lower() in ("1", "true", "yes")
        except Exception:
            end_on_answer = True
        if end_on_answer and (answered or not missing):
            self._logger.info("reflect: Ending on answer (or no missing info) per ANALYSIS_END_ON_ANSWER; routing to synthesis")
            return Command(update={"finalize": True}, goto="synthesis")

        # Check if there are remaining planned steps to execute
        remaining_steps = len(state.plan_steps or []) - state.next_step_index
        if remaining_steps > 0:
            self._logger.info(f"reflect: {remaining_steps} planned steps remaining, continuing execution")
            # Don't route to synthesis yet - let execute_step continue
            return Command(update={}, goto="execute_step")

        # If no steps remain and it's still not answered, fall back to synthesis
        if answered or not missing:
            self._logger.info("reflect: No missing info or query answered, routing to synthesis")
            return Command(update={}, goto="synthesis")

        # INTELLIGENT REASONING: Before blindly replanning, check if existing tool outputs can answer
        # the missing information or if we can reason through what we already have
        self._logger.info(f"reflect: Found missing info, checking if existing data can answer...")
        try:
            reasoning_result = await self._reason_about_existing_data(state, missing)
            self._logger.info(f"reflect: Reasoning complete - can_answer={reasoning_result['can_answer']}")
            if reasoning_result["can_answer"]:
                self._logger.info(f"reflect: Existing data sufficient to answer. Reasoning: {reasoning_result['reasoning']}")
                # If interpretation may have under-extracted insights, allow ONE re-interpretation pass
                # using the existing transcript (no new tool calls). This tries to recover the missing 1/4.
                try:
                    attempts = int(getattr(state, "reinterpret_attempts", 0) or 0)
                except Exception:
                    attempts = 0
                if attempts < 1:
                    self._logger.info("reflect: triggering single reinterpretation pass to recover missed insights")
                    return Command(update={"reinterpret_attempts": attempts + 1}, goto="interpret_results")

                # Update coverage to mark as answered and proceed to synthesis.
                # Note: synthesis is formatting-only and relies on state.insights.
                updates = {
                    "coverage": {
                        "answers_query": True,
                        "missing_info": [],
                        "reasoning_override": reasoning_result['reasoning']
                    }
                }
                return Command(update=updates, goto="synthesis")
            else:
                self._logger.info(f"reflect: Reasoning determined replanning needed: {reasoning_result['reasoning']}")
        except Exception as e:
            self._logger.warning(f"reflect: Reasoning step failed: {e}, proceeding with normal flow")
            import traceback
            self._logger.debug(f"reflect: Reasoning error traceback:\n{traceback.format_exc()}")

        # Attach refinement request for planner consumption (deduplicated, trimmed)
        cleaned: list[str] = []
        seen = set()
        for m in missing:
            cm = m.strip()[:200]
            if cm and cm not in seen:
                seen.add(cm)
                cleaned.append(cm)
        # Refinement round capping
        try:
            max_refinements = int(os.getenv("ANALYSIS_MAX_REFINEMENTS", "2"))
        except Exception:
            max_refinements = 2
        current_round = getattr(state, "refinement_round", 0) or 0
        
        self._logger.info(f"reflect: current_round={current_round}, max={max_refinements}")
        
        if current_round >= max_refinements:
            # Stop looping; proceed to synthesis with note captured in warnings at synthesis stage
            self._logger.warning(f"reflect: Max refinements ({max_refinements}) reached, forcing synthesis to END")
            updates: dict = {"refinement_round": current_round, "finalize": True}
            # Optional: provide explicit tool-gap rationale on forced termination.
            try:
                explain_gaps = str(os.getenv("ANALYSIS_EXPLAIN_TOOL_GAPS", "1")).lower() in ("1", "true", "yes")
            except Exception:
                explain_gaps = True
            if explain_gaps:
                try:
                    already = getattr(state, "capability_gap", None)
                except Exception:
                    already = None
                if not isinstance(already, dict) and missing_raw:
                    try:
                        updates["capability_gap"] = await self._diagnose_tool_gap(state, missing_raw)
                    except Exception as _ge2:
                        self._logger.warning(f"reflect: forced-termination capability-gap diagnosis failed ({_ge2})")
            return Command(update=updates, goto="synthesis")

        # Replan guard: if last plan didn't change repeatedly, cap attempts
        try:
            max_replans = int(os.getenv("ANALYSIS_MAX_REPLAN_ATTEMPTS", "2"))
        except Exception:
            max_replans = 2
        replan_no_change = bool(getattr(state, "replan_no_change", False))
        prev_retries = int(getattr(state, "plan_retries", 0) or 0)

        if replan_no_change:
            retries_now = prev_retries + 1
            if retries_now > max_replans:
                self._logger.warning("reflect: Replan-without-change exceeded max; finalizing to synthesis")
                return Command(update={"plan_retries": retries_now, "finalize": True}, goto="synthesis")
            # Proceed to planner but persist retry count
            base_updates = {"plan_retries": retries_now}
        else:
            # Reset retry counter on observed plan change
            base_updates = {"plan_retries": 0} if prev_retries else {}

        # Increment refinement round and attach refinement without resetting next_step_index
        self._logger.info(f"reflect: Requesting replanning (round {current_round + 1}), missing: {cleaned[:3]}")
        updates = {**base_updates, "refinement_request": cleaned, "refinement_round": current_round + 1, "telemetry": {"refinement_rounds": 1}}
        return Command(update=updates, goto="planner")


    async def synthesis(self, state: AnalysisPipelineState) -> PipelineOutputState:
        """Generate the final analysis report and populate a structured output envelope.

        Strategy:
        - Ask the LLM for STRICT JSON with evidence-bound insights (see prompts.py).
        - If JSON parses, verify each insight's evidence is present in transcript/artifacts; drop unsupported ones.
        - Render Markdown deterministically from the verified JSON to avoid hallucinations.
        - If parsing fails, fallback to the legacy freeform Markdown synthesis for robustness.
        """
        self._log_node("synthesis", state)
        # If upstream provided a structured synthesis_input, log its presence for traceability
        try:
            synth_in = getattr(state, "synthesis_input", None)
            if isinstance(synth_in, dict):
                self._logger.info(f"synthesis: received synthesis_input with keys={list(synth_in.keys())}")
        except Exception:
            pass
        # SINGLE RUN GUARD EARLY RETURN: If synthesis already completed and cached, return it.
        if getattr(self, "_single_run_guard_enabled", False) and getattr(self, "_single_run_completed", False) and self._cached_final_output is not None:
            self._logger.info("synthesis: single-run guard active; returning cached final output")
            return self._cached_final_output
        
        # CRITICAL: We've reached synthesis - this MUST be the final node
        # Setting this in state doesn't work because synthesis returns PipelineOutputState
        # which doesn't update the AnalysisPipelineState for subsequent nodes.
        # The graph should END here via the edge to END, but if somehow execution continues,
        # other nodes check this flag.
        
        # If there is nothing to synthesize (no valid plan and no tool transcript),
        # return a diagnostic error instead of a generic placeholder.
        if not state.plan_steps and not state.tool_transcript:
            allowed = sorted(self._tool_map.keys())
            diag_artifacts: Dict[str, Any] = {
                "artifact_log": state.artifact_log,
                "plan_steps": [],
                "token_metrics": state.token_metrics,
                "user_instruction": state.instruction,
                "planning_allowed_tools": allowed,
            }
            if state.dataset_path:
                diag_artifacts["dataset_path"] = state.dataset_path

            msg = (
                "# Planning error\n\n"
                "The planner did not produce any executable steps using known MCP tools.\n\n"
                "What you can do next:\n"
                "- Ensure the instruction specifies concrete actions that match available tools.\n"
                "- Use tool names exactly from the allowed list below.\n\n"
                "Allowed tools: " + ", ".join(allowed)
            )

            return PipelineOutputState(
                status="error",
                title=None,
                report_text=msg,
                summary="Planner returned no executable steps",
                artifacts=diag_artifacts,
                dataset_path=state.dataset_path,
                steps=[],
                tool_summaries=[],
                token_metrics=state.token_metrics,
                warnings=["no_valid_plan_steps"],
                errors=["Planner produced no valid steps"],
            )

        # Route flags (visualization-only, insights-only)
        # visualization_only propagated by planner; insights_only via output_mode
        viz_only = bool(getattr(state, "visualization_only", False))
        insights_only = str(getattr(state, "output_mode", "")) == "insights_only"

        # Treat viz_only as sufficient for minimal formatting (remove prior strict gating)
        # Retain env toggle only to force legacy behavior if explicitly disabled.
        try:
            legacy_full_on_viz_only = str(os.getenv("ANALYSIS_VIZ_ONLY_LEGACY_FULL", "0")).lower() in ("1", "true", "yes")
        except Exception:
            legacy_full_on_viz_only = False
        # If legacy_full_on_viz_only is true, we will run standard synthesis even for viz_only.
        viz_minimal = viz_only and not legacy_full_on_viz_only


        report_text = None
        structured_payload_json = None

        # Attempt late artifact discovery if visualization-only and artifact log empty
        if viz_minimal and not state.artifact_log:
            try:
                # Re-parse tool transcript outputs for artifact paths
                discovered: list[str] = []
                for evt in (state.tool_transcript or [])[-10:]:
                    out = evt.get("output") or evt.get("result") or ""
                    for p in self._parse_artifact_paths(out):
                        if p not in discovered:
                            discovered.append(p)
                # Best-effort scan plots directory: prefer most recent plot files
                plots_dir = None
                try:
                    cfg = self._get_config()
                    plots_dir = cfg.plots_path
                except Exception:
                    plots_dir = None
                if plots_dir and plots_dir.exists():
                    try:
                        pngs = sorted(list(plots_dir.glob("*.png")), key=lambda p: p.stat().st_mtime, reverse=True)
                    except Exception:
                        pngs = list(plots_dir.glob("*.png"))
                    count = 0
                    for f in pngs:
                        try:
                            if str(f) not in discovered:
                                discovered.append(str(f))
                                count += 1
                            if count >= 10:
                                break
                        except Exception:
                            pass
                for d in discovered:
                    try:
                        state.register_artifact(d)
                    except Exception:
                        pass
            except Exception:
                pass

        if viz_minimal:
            try:
                # Strengthen artifact discovery: parse filenames from recent tool outputs and register immediately
                try:
                    for evt in (state.tool_transcript or [])[-10:]:
                        out_txt = str(evt.get("output") or evt.get("result") or "")
                        for p in self._parse_artifact_paths(out_txt):
                            try:
                                state.register_artifact(p)
                                # Attach lightweight event-level artifacts for downstream traceability
                                try:
                                    evt.setdefault("artifacts", []).append(Path(p).name)
                                except Exception:
                                    pass
                            except Exception:
                                pass
                except Exception:
                    pass
                art_names = []
                try:
                    art_names = [Path(a).name for a in (state.artifact_log or []) if a]
                except Exception:
                    art_names = list(state.artifact_log or [])
                lines = ["# Visualization Results", ""]
                ds = state.dataset_path or "unspecified"
                lines.append(f"Dataset: `{ds}`")
                lines.append("")
                if art_names:
                    first = next((a for a in art_names if a.lower().endswith((".png",".jpg",".jpeg",".svg"))), None)
                    if first:
                        lines.append(f"![{first}]({first})")
                        lines.append("")
                    lines.append("## Artifacts") 
                    for a in sorted(set(art_names)):
                        lines.append(f"- {a}")
                    lines.append("")
                    # NEW: Display tool inputs that produced visuals for traceability
                    try:
                        # Build a mapping from artifact basename -> list of (tool, key_args)
                        visual_inputs = []
                        recent_events = (state.tool_transcript or [])[-10:]
                        for evt in recent_events:
                            out_txt = str(evt.get("output") or evt.get("result") or "")
                            # Heuristically associate event to any listed artifact names appearing in output
                            for art in art_names:
                                try:
                                    evt_artifacts = set(evt.get("artifacts", []) or [])
                                    if art and (art in out_txt or art in evt_artifacts):
                                        tool = str(evt.get("tool") or "unknown")
                                        args = evt.get("args") or {}
                                        # Show only common plotting/analysis args to keep it concise
                                        key_arg_names = [
                                            "plot_type", "x_column", "y_column", "hue_column",
                                            "columns", "method", "datetime_column", "incident_type_column",
                                            "title"
                                        ]
                                        key_args = {k: args.get(k) for k in key_arg_names if k in args}
                                        visual_inputs.append({"artifact": art, "tool": tool, "args": key_args})
                                except Exception:
                                    continue
                        # If no direct association found but we have artifacts, fallback to last plotting event
                        if not visual_inputs and art_names:
                            try:
                                fallback_evt = None
                                for evt in reversed(recent_events):
                                    tn = str(evt.get("tool") or "")
                                    if "plot" in tn.lower():
                                        fallback_evt = evt
                                        break
                                if fallback_evt:
                                    args_fb = fallback_evt.get("args") or {}
                                    visual_inputs = [{"artifact": art_names[0], "tool": str(fallback_evt.get("tool") or "unknown"), "args": args_fb}]
                            except Exception:
                                pass

                        if visual_inputs:
                            lines.append("## Visual Inputs")
                            for vi in visual_inputs:
                                try:
                                    art = vi.get("artifact")
                                    tool = vi.get("tool")
                                    args = vi.get("args") or {}
                                    # Render compact arg string
                                    arg_pairs = [f"{k}={args[k]}" for k in args if args.get(k) is not None]
                                    arg_str = ", ".join(arg_pairs) if arg_pairs else "(no key args captured)"
                                    lines.append(f"- {art}: generated by `{tool}` with {arg_str}")
                                except Exception:
                                    continue
                            lines.append("")

                        # NEW: Generate plot-linked insights tied to each artifact using captured args
                        try:
                            def _mk_plot_insight(vi: dict) -> str:
                                art = vi.get("artifact")
                                args = vi.get("args") or {}
                                plot_type = str(args.get("plot_type") or "plot").lower()
                                title = str(args.get("title") or "").strip()
                                x = args.get("x_column") or args.get("datetime_column")
                                y = args.get("y_column")
                                hue = args.get("hue_column") or args.get("incident_type_column")
                                cols = args.get("columns")
                                method = str(args.get("method") or "").lower()
                                # Build a concise, artifact-referenced insight sentence
                                parts = []
                                if title:
                                    parts.append(title)
                                elif plot_type:
                                    parts.append(plot_type.capitalize())
                                base = " ".join(parts) if parts else "Visualization"
                                details = []
                                if x and y:
                                    details.append(f"shows {x} vs {y}")
                                elif cols and isinstance(cols, (list, tuple)) and len(cols) >= 2:
                                    details.append(f"relates {cols[0]} to {cols[1]}")
                                if hue:
                                    details.append(f"grouped by {hue}")
                                if method:
                                    details.append(f"using {method}")
                                detail_str = ", ".join(details) if details else "high-level patterns"
                                return f"{base} ({art}) {detail_str}."

                            plot_linked_insights = []
                            for vi in visual_inputs:
                                try:
                                    ins = _mk_plot_insight(vi)
                                    if ins:
                                        plot_linked_insights.append(ins[:180])
                                except Exception:
                                    continue
                            if plot_linked_insights:
                                lines.append("## Plot-Linked Insights")
                                for ins in plot_linked_insights:
                                    lines.append(f"- {ins}")
                                lines.append("")
                        except Exception:
                            pass
                    except Exception:
                        # Non-fatal: skip inputs section on error
                        pass
                else:
                    lines.append("...")

                # Brief insights & recommendations (no evidence) for viz-only route
                # Strategy: lightweight LLM prompt using recent tool transcript; fallback heuristics if failure.
                brief_insights: list[str] = []
                brief_recs: list[str] = []
                try:
                    llm_ctx = []
                    for evt in (state.tool_transcript or [])[-5:]:
                        out_txt = str(evt.get("output") or evt.get("result") or "")
                        if out_txt:
                            llm_ctx.append(out_txt[:LLM_CONTEXT_SNIPPET_MAX_CHARS])
                    ctx_block = "\n".join(llm_ctx)[:1500]
                    if ctx_block.strip() or art_names:
                        sys_msg = SystemMessage(content="You generate 1-2 concise insights (<=120 chars each) and 1-2 terse engineering recommendations from visualization context. No evidence, no extra plots.")
                        user_prompt = (
                            "VISUALIZATION CONTEXT:\n" + ctx_block + "\n\n" +
                            f"ARTIFACTS: {art_names}\n" +
                            "Return ONLY JSON: {\"insights\": [..], \"recommendations\": [..]} "
                            "Rules: Keep each item short; reference visible patterns (peaks, clusters, anomalies)."
                        )
                        resp = await self._llm.ainvoke([sys_msg, HumanMessage(content=user_prompt)])
                        parsed = self._extract_json(resp.content if hasattr(resp, 'content') else str(resp))
                        if isinstance(parsed, dict):
                            brief_insights = [str(x)[:120] for x in (parsed.get("insights") or []) if isinstance(x, str)][:2]
                            brief_recs = [str(x)[:140] for x in (parsed.get("recommendations") or []) if isinstance(x, str)][:2]
                except Exception:
                    pass
                # Revised behavior: avoid generic hardcoded insights.
                # If LLM didn't return JSON and we lack artifacts, emit a diagnostic note instead of placeholders.
                if not brief_insights:
                    if art_names:
                        # Keep minimal recommendation only when a visual exists
                        brief_insights = []
                        if not brief_recs:
                            brief_recs = ["Review generated visualization(s) and consider targeted follow-up analysis."]
                    else:
                        lines.append("> Note: No visualization artifacts were detected and the brief-insights generator returned no structured results.")
                        lines.append(
                            "> To get evidence-backed insights, run non-viz-only mode or ensure plotting tools produce artifacts."
                        )
                        lines.append("")
                        brief_insights = []
                        # Provide a clear next step instead of generic recommendations
                        if not brief_recs:
                            brief_recs = ["Its a work in progress. Sometimes there is relevant insights but sometimes not. I dont know why"]
                # If we have artifacts but only one insight, avoid adding a generic second insight.
                # Keep recommendations minimal and action-oriented.
                if len(brief_recs) < 1 and art_names:
                    brief_recs.append("Validate plot inputs (x/y/hue) and regenerate visuals for clarity.")

                # Only show LLM brief insights if present and distinct from plot-linked insights
                if brief_insights:
                    lines.append("## Brief Insights")
                    for ins in brief_insights:
                        lines.append(f"- {ins}")
                    lines.append("")
                lines.append("## Recommendations")
                for rec in brief_recs:
                    lines.append(f"- {rec}")
                lines.append("")
                report_text = "\n".join(lines)
            except Exception:
                art_names = []
                report_text = "# Visualization Results\n\n(Artifacts unavailable)"

            # Minimal summary to satisfy AnalysisPipelineState validation
            summary_text = f"Visualization-only run: {len(art_names)} artifact(s)."
            # Reflect in state as well for consistency
            try:
                state.set_report(report_text, summary_text, {"artifact_log": state.artifact_log})
            except Exception:
                state.summary = summary_text
                state.report_text = report_text

            try:
                steps_out = list(state.plan_steps or [])
                tool_summaries = list(state.tool_transcript or [])
            except Exception:
                steps_out, tool_summaries = [], []
            output = PipelineOutputState(
                status="ok",
                title="Visualization Results",
                report_text=report_text,
                summary=summary_text,
                artifacts={"artifact_log": state.artifact_log},
                dataset_path=state.dataset_path,
                steps=steps_out,
                tool_summaries=tool_summaries,
                token_metrics=state.token_metrics,
            )
            try:
                self._single_run_completed = True
                self._cached_final_output = output
            except Exception:
                pass
            return output

        # Non-visualization-only: build full synthesis prompt
        prompt_dict = build_synthesis_prompt(state)

        system_msg = SystemMessage(content=prompt_dict["system"])
        context_block = f"""
Context: {prompt_dict['context_summary']}
STATE (authoritative JSON):
{prompt_dict.get('state', '')}
Plan summary: {prompt_dict['plan_summary']}
Tool transcript: {prompt_dict['tool_transcript']}
Artifacts: {prompt_dict['artifacts']}
Insights (preview): {prompt_dict.get('insights', '')}
User instruction: {prompt_dict['instruction']}

Return JSON ONLY as per the schema described in the system message. Do not include markdown fences or commentary.
"""
        user_msg = HumanMessage(content=context_block.strip())

        llm = self._llm
        try:
            if not viz_minimal:
                response = await llm.ainvoke([system_msg, user_msg])
                raw = response.content if isinstance(response, AIMessage) else str(response)
                state.record_token_usage("synthesis", len(raw) // 4)

            # Attempt to parse the STRICT JSON
            payload = self._extract_json(raw) if not viz_minimal else {}
            self._logger.info(f"synthesis: Parsed payload type={type(payload)}, is_dict={isinstance(payload, dict)}")
            if isinstance(payload, dict) and not viz_minimal:
                self._logger.info(f"synthesis: Payload keys={list(payload.keys())}")
            else:
                if not viz_minimal:
                    self._logger.warning(f"synthesis: Payload is not a dict! Type={type(payload)}, Content={str(payload)[:200]}")

            # Keep a copy of the structured payload (if present) for optional JSON sidecar write
            structured_payload_json: Optional[str] = None
            try:
                if isinstance(payload, dict):
                    structured_payload_json = json.dumps(payload, ensure_ascii=False)
            except Exception:
                structured_payload_json = None

            def _concat_transcript(evts: list[dict]) -> str:
                try:
                    return "\n".join([str(e.get("output") or "") for e in evts if isinstance(e, dict)])
                except Exception:
                    return ""

            transcript_full = _concat_transcript(state.tool_transcript)
            # Normalize artifacts by basename to deduplicate path variants
            try:
                artifact_basenames = {Path(a).name for a in (state.artifact_log or []) if a}
            except Exception:
                artifact_basenames = set()

            # Env toggle: enforce dropping insights without verifiable trace
            try:
                enforce_trace = str(os.getenv("ANALYSIS_ENFORCE_TRACEABILITY", "1")).lower() in ("1", "true", "yes")
            except Exception:
                enforce_trace = True

            def _verify_with_ref(snippet: str, ref: Optional[dict]) -> tuple[bool, Optional[dict], Optional[str]]:
                """Verify evidence using structured reference when provided.

                Returns (ok, trace, error_reason). trace is a normalized dict like
                {"source":"transcript","line_index":int,"tool":str} or {"source":"artifact","artifact":str}.
                """
                snippet = (snippet or "").strip()
                if not ref or not isinstance(ref, dict):
                    # No structured ref; treat as unverifiable here
                    return False, None, "no_ref"
                source = str(ref.get("source") or "").strip().lower()
                if source == "transcript":
                    li = ref.get("line_index")
                    if not isinstance(li, int):
                        return False, None, "missing_line_index"
                    if li < 0 or li >= len(state.tool_transcript):
                        return False, None, "invalid_line_index"
                    ev = state.tool_transcript[li]
                    tool_expected = (ref.get("tool") or "").strip().lower()
                    tool_ok = True
                    if tool_expected:
                        tool_ok = str(ev.get("tool") or "").strip().lower() == tool_expected
                    # Evidence must be literal substring of this specific event's output/result
                    ev_text = (ev.get("output") or ev.get("result") or "")
                    snippet_ok = bool(snippet) and (snippet in ev_text)
                    if tool_ok and snippet_ok:
                        return True, {"source": "transcript", "line_index": li, "tool": ev.get("tool")}, None
                    return False, None, "mismatch_tool_or_snippet"
                elif source == "artifact":
                    art = ref.get("artifact")
                    try:
                        art_name = Path(art).name if art else None
                    except Exception:
                        art_name = art
                    if not art_name:
                        return False, None, "missing_artifact"
                    if art_name not in artifact_basenames:
                        return False, None, "artifact_not_found"
                    # For artifacts we require a non-empty snippet but don't enforce substring in transcript
                    if not snippet:
                        return False, None, "empty_snippet"
                    return True, {"source": "artifact", "artifact": art_name}, None
                else:
                    return False, None, "unknown_source"

            def _fallback_verify(snippet: str, source: str, artifact: Optional[str]) -> bool:
                """Legacy verification without structured ref.

                - artifact: snippet must be non-empty and either appear in transcript or equal artifact basename
                - transcript: snippet must appear in the full transcript
                """
                snippet = (snippet or "").strip()
                if source == "artifact":
                    # Tighten: require the artifact to actually exist in the artifact log; do not accept phantom plots
                    if not snippet:
                        return False
                    try:
                        art_name = Path(artifact).name if artifact else None
                    except Exception:
                        art_name = artifact
                    if not art_name:
                        return False
                    # Only accept if the artifact name is present in discovered artifacts
                    if art_name not in artifact_basenames:
                        return False
                    # Snippet presence is optional once artifact exists; keep minimal requirement
                    return True
                return bool(snippet) and (snippet in transcript_full)

            # When structured payload is present, build deterministic Markdown
            # If strict viz-only, skip structured JSON rendering path
            report_text = report_text if viz_minimal else None
            if (not viz_minimal) and isinstance(payload, dict) and ("insights" in payload or "dataset_overview" in payload):
                self._logger.info("synthesis: Rendering Markdown from structured payload")
                title = payload.get("title") or "Analysis Report"
                dov = payload.get("dataset_overview") or {}
                path = dov.get("path") or (state.dataset_path or "unspecified")
                rows = dov.get("rows")
                cols = dov.get("columns")
                notes = dov.get("notes")
                insights = payload.get("insights") or []
                next_steps = payload.get("next_steps") or []
                recs = payload.get("recommendations") or []

                # Telemetry counters for traceability
                t_verified = 0
                t_dropped = 0
                t_invalid_index = 0
                t_artifact_missing = 0
                # Pre-cull repetitive/near-duplicate insights to speed verification and rendering
                try:
                    max_process = int(os.getenv("ANALYSIS_SYNTHESIS_MAX_INSIGHTS_PROCESSED", "60"))
                except Exception:
                    max_process = 60
                try:
                    dedup_ratio = float(os.getenv("ANALYSIS_SYNTHESIS_DEDUP_RATIO", "0.92"))
                except Exception:
                    dedup_ratio = 0.92
                try:
                    max_per_evidence = int(os.getenv("ANALYSIS_SYNTHESIS_MAX_PER_EVIDENCE", "2"))
                except Exception:
                    max_per_evidence = 2
                # Allow "show all" mode to bypass pre-cull caps and keep all insights
                try:
                    show_all = str(os.getenv("ANALYSIS_REPORT_SHOW_ALL", "0")).lower() in ("1", "true", "yes")
                except Exception:
                    show_all = False
                try:
                    simple_mode = str(os.getenv("ANALYSIS_SIMPLE_MODE", "0")).lower() in ("1", "true", "yes")
                except Exception:
                    simple_mode = False
                if not (show_all or simple_mode):
                    prefiltered: list[dict] = []
                    seen_norm: list[str] = []
                    evidence_counts: dict[str, int] = {}
                    def _norm_txt(t: str) -> str:
                        try:
                            return re.sub(r"\s+", " ", (t or "").strip().lower())
                        except Exception:
                            return (t or "").strip().lower()

                    original_total = len(insights or [])
                    for item in (insights or []):
                        try:
                            stmt = str(item.get("statement") or "").strip()
                            ev = str(item.get("evidence_snippet") or item.get("evidence") or "").strip()
                            if not stmt or not ev:
                                continue
                            nstmt = _norm_txt(stmt)
                            # Exact duplicate by normalized statement
                            if any(nstmt == s for s in seen_norm):
                                continue
                            # Near-duplicate by similarity
                            is_dup = False
                            for s in seen_norm:
                                try:
                                    if difflib.SequenceMatcher(None, nstmt, s).ratio() >= dedup_ratio:
                                        is_dup = True
                                        break
                                except Exception:
                                    continue
                            if is_dup:
                                continue
                            # Cap the number of items sharing the same evidence snippet
                            ev_key = _norm_txt(ev)
                            cnt = evidence_counts.get(ev_key, 0)
                            if cnt >= max_per_evidence:
                                continue
                            evidence_counts[ev_key] = cnt + 1
                            seen_norm.append(nstmt)
                            prefiltered.append(item)
                            if len(prefiltered) >= max_process:
                                break
                        except Exception:
                            continue

                    if prefiltered:
                        self._logger.info(f"synthesis: pre-cull reduced insights from {original_total} to {len(prefiltered)} (max_process={max_process})")
                        insights = prefiltered

                t_total = len(insights or [])
                t_prefilter_dropped = max(0, (original_total if 'original_total' in locals() else t_total) - t_total)

                # Verify insights and drop unsupported ones
                verified = []
                for item in insights:
                    try:
                        stmt = str(item.get("statement") or "").strip()
                        ev = str(item.get("evidence_snippet") or "").strip()
                        src = str(item.get("evidence_source") or "transcript").strip().lower()
                        art = item.get("artifact")
                        ref = item.get("evidence_ref") or item.get("ref")
                        ok, trace, err = _verify_with_ref(ev, ref)
                        if not ok:
                            # When enforcement is OFF, accept the insight without a strict trace
                            if not enforce_trace:
                                verified.append({"statement": stmt, "evidence": ev, "source": src, "artifact": art, "trace": None})
                                t_verified += 1
                                continue
                            # With enforcement ON, attempt a legacy/fallback verification
                            if _fallback_verify(ev, src, art):
                                verified.append({"statement": stmt, "evidence": ev, "source": src, "artifact": art, "trace": trace})
                                t_verified += 1
                                continue
                            # Count specific errors
                            if err == "invalid_line_index":
                                t_invalid_index += 1
                            if err == "artifact_not_found":
                                t_artifact_missing += 1
                            t_dropped += 1
                            continue
                        # ok via structured ref
                        verified.append({"statement": stmt, "evidence": ev, "source": trace.get("source") if trace else src, "artifact": (trace.get("artifact") if trace else art), "trace": trace})
                        t_verified += 1
                    except Exception:
                        t_dropped += 1
                        continue

                # Authoritative insights: if interpret_results already produced state.insights,
                # render those verbatim (same order/count) and do NOT reprocess or rewrite.
                # We still use the LLM payload for analysis_summary/recommendations/next_steps.
                use_state_insights_for_report = False
                try:
                    state_insights = list(getattr(state, "insights", []) or [])
                except Exception:
                    state_insights = []
                if state_insights:
                    normalized: list[dict] = []
                    for it in state_insights:
                        try:
                            if not isinstance(it, dict):
                                continue
                            stmt = str(it.get("statement") or it.get("title") or "").strip()
                            ev = str(it.get("evidence_snippet") or it.get("evidence") or "").strip()
                            src = str(it.get("evidence_source") or "transcript").strip().lower()
                            art = it.get("artifact")
                            ref = it.get("evidence_ref") or it.get("ref")
                            normalized.append({
                                "statement": stmt,
                                "evidence": ev,
                                "source": src,
                                "artifact": art,
                                "trace": ref,
                            })
                        except Exception:
                            # Keep going; do not fail synthesis due to a single malformed insight.
                            continue
                    if normalized:
                        verified = normalized
                        use_state_insights_for_report = True
                        # Telemetry: when using authoritative state insights, treat them as the displayed set.
                        try:
                            t_total = len(verified)
                            t_verified = len(verified)
                            t_dropped = 0
                            t_invalid_index = 0
                            t_artifact_missing = 0
                        except Exception:
                            pass

                # If visualization-only and nothing verified yet but we have artifacts, synthesize a minimal insight tied to the plot artifact
                try:
                    if viz_only and not verified and artifact_basenames:
                        first_art = sorted(list(artifact_basenames))[0]
                        verified.append({
                            "statement": "Visualization generated for temporal pattern by day.",
                            "evidence": first_art,
                            "source": "artifact",
                            "artifact": first_art,
                            "trace": {"source": "artifact", "artifact": first_art}
                        })
                        t_verified += 1
                except Exception:
                    pass

                # CRITICAL: Check quality threshold before rendering
                # BUT: Skip this check if we're at max refinements (forced termination)
                # Make the minimum verified insights threshold configurable via env.
                try:
                    min_verified_threshold = int(os.getenv("ANALYSIS_MIN_VERIFIED_INSIGHTS", "3"))
                except Exception:
                    min_verified_threshold = 3
                # Visualization-only runs require fewer insights; allow single-insight output
                if viz_only:
                    min_verified_threshold = min(min_verified_threshold, 1)
                current_refine_round = getattr(state, "refinement_round", 0) or 0
                max_refinements = 2
                try:
                    max_refinements = int(os.getenv("ANALYSIS_MAX_REFINEMENTS", "2"))
                except Exception:
                    pass
                
                forced_termination = current_refine_round >= max_refinements or getattr(state, "finalize", False)
                
                if forced_termination:
                    self._logger.info(f"synthesis: FORCED TERMINATION - bypassing quality threshold (refine_round={current_refine_round}, finalize={getattr(state, 'finalize', False)})")
                
                if len(verified) < min_verified_threshold and not forced_termination:
                    self._logger.warning(f"Only {len(verified)} verified insights (threshold: {min_verified_threshold}), checking if we should return diagnostic")
                    
                    # If we dropped insights due to errors OR have tool errors, return diagnostic
                    if t_dropped > 0 or (state.tool_transcript and any(e.get('status') == 'error' for e in state.tool_transcript)):
                        error_summary = [
                            e.get('output', '')[:200] 
                            for e in (state.tool_transcript or [])
                            if e.get('status') == 'error'
                        ]
                        
                        diag_msg = (
                            "# Analysis Quality Issue\n\n"
                            f"The pipeline completed but produced only **{len(verified)} verified insights** (minimum: {min_verified_threshold}).\n\n"
                        )
                        
                        if error_summary:
                            diag_msg += "**Tool Execution Errors:**\n"
                            for idx, err in enumerate(error_summary[:3], 1):
                                diag_msg += f"{idx}. `{err}`\n"
                            diag_msg += "\n"
                        
                        if t_dropped > 0:
                            diag_msg += f"**Insights Dropped:** {t_dropped} insights failed verification\n\n"
                        
                        diag_msg += (
                            "**Possible Causes:**\n"
                            "- Column names in plan don't match dataset schema\n"
                            "- Required data fields are missing\n"
                            "- Tool configuration issues\n"
                            "- Dataset format incompatibility\n\n"
                            "**Recommendation:** Review the dataset schema and ensure analysis instructions "
                            "reference actual columns in the data. Check the dataset snapshot for available columns."
                        )
                        
                        # Persist diagnostic report to disk EVEN in early quality failure so caller sees a file
                        try:
                            cfg = self._get_config()
                            report_dir: Path = cfg.reports_path
                            report_dir.mkdir(parents=True, exist_ok=True)
                            dataset_name = "no_dataset"
                            if state.dataset_path:
                                try:
                                    dataset_name = Path(state.dataset_path).stem
                                except Exception:
                                    dataset_name = "no_dataset"
                            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                            report_filename = f"analysis_report_{dataset_name}_diag_{ts}.md"
                            report_path = report_dir / report_filename
                            with report_path.open("w", encoding="utf-8") as f:
                                f.write(diag_msg)
                        except Exception:
                            report_path = None
                        artifacts_diag = {
                            "artifact_log": state.artifact_log,
                            "plan_steps": state.plan_steps,
                            "token_metrics": state.token_metrics,
                            "user_instruction": state.instruction,
                            "telemetry": {
                                "insights_total": t_total,
                                "insights_verified": t_verified,
                                "insights_dropped": t_dropped,
                            }
                        }
                        if report_path:
                            artifacts_diag["report_path"] = str(report_path)
                        return PipelineOutputState(
                            status="error",
                            title="Analysis Quality Issue",
                            report_text=diag_msg,
                            summary=f"Only {len(verified)} insights met quality threshold",
                            artifacts=artifacts_diag,
                            dataset_path=state.dataset_path,
                            steps=[],
                            tool_summaries=[],
                            token_metrics=state.token_metrics,
                            errors=[f"insufficient_insights_{len(verified)}_of_{min_verified_threshold}"],
                            warnings=["quality_threshold_not_met"],
                        )

                # Render deterministic Markdown with INSIGHTS FIRST, plots supporting each insight
                lines = [f"# {title}", ""]

                # Viz-only layout: place the first plot at the top of the report
                try:
                    if viz_only and artifact_basenames:
                        first_art = sorted(list(artifact_basenames))[0]
                        try:
                            rel_display_top = Path(first_art).name
                        except Exception:
                            rel_display_top = str(first_art)
                        rel_display_top = rel_display_top.replace('\\', '/')
                        lines.append(f"![{rel_display_top}]({rel_display_top})")
                        lines.append("")
                except Exception:
                    pass
                
                # Add quality warning if we have low insights but forced termination
                if len(verified) < min_verified_threshold and forced_termination:
                    lines.append("> **Note**: Analysis completed with limited insights due to max refinement attempts reached.")
                    lines.append("> Some requested aspects may be incomplete or missing.")
                    lines.append("")

                if verified:
                    # Adaptively choose how many insights to render based on instruction specificity
                    # rather than a fixed cap. All verified insights remain in artifacts/telemetry.
                    def _instruction_specificity(instr: str, columns: List[str]) -> float:
                        try:
                            text = (instr or "").lower()
                            if not text:
                                return 0.0  # treat empty as generic
                            score = 0.5
                            # shorter, targeted prompts tend to be specific
                            if len(text) < 60:
                                score += 0.2
                            # presence of dataset column names suggests specificity
                            try:
                                cols_low = {c.lower() for c in (columns or [])}
                                hits = sum(1 for c in cols_low if c and c in text)
                                if hits >= 3:
                                    score += 0.3
                                elif hits == 2:
                                    score += 0.2
                                elif hits == 1:
                                    score += 0.1
                            except Exception:
                                pass
                            # intent tokens
                            specific_tokens = [
                                "correlation", "regression", "impact", "effect", "between", "vs", "by ",
                                "predict", "classify", "anomaly", "cluster", "hypothesis", "test", "trend of"
                            ]
                            if any(tok in text for tok in specific_tokens):
                                score += 0.2
                            generic_tokens = ["comprehensive", "overview", "explore", "exploratory", "broad", "general", "all aspects"]
                            if any(tok in text for tok in generic_tokens):
                                score -= 0.3
                            # clamp
                            if score < 0.0:
                                score = 0.0
                            if score > 1.0:
                                score = 1.0
                            return score
                        except Exception:
                            return 0.5

                    # Show-all mode bypasses adaptive slicing and displays every verified insight
                    try:
                        show_all = str(os.getenv("ANALYSIS_REPORT_SHOW_ALL", "0")).lower() in ("1", "true", "yes")
                    except Exception:
                        show_all = False
                    try:
                        simple_mode = str(os.getenv("ANALYSIS_SIMPLE_MODE", "0")).lower() in ("1", "true", "yes")
                    except Exception:
                        simple_mode = False
                    if not (show_all or simple_mode):
                        # Ratios are env-tunable; using ratios avoids an artificial fixed number
                        try:
                            ratio_generic = float(os.getenv("ANALYSIS_REPORT_INSIGHTS_RATIO_GENERIC", "0.6"))
                        except Exception:
                            ratio_generic = 0.6
                        try:
                            ratio_specific = float(os.getenv("ANALYSIS_REPORT_INSIGHTS_RATIO_SPECIFIC", "0.3"))
                        except Exception:
                            ratio_specific = 0.3
                        try:
                            min_show = int(os.getenv("ANALYSIS_REPORT_INSIGHTS_MIN", "7"))
                        except Exception:
                            min_show = 4
                        try:
                            max_show_env = os.getenv("ANALYSIS_REPORT_INSIGHTS_MAX", "")
                            max_show = int(max_show_env) if max_show_env else None
                        except Exception:
                            max_show = None

                    dataset_columns = []
                    try:
                        if isinstance(state.preprocess_profile, dict):
                            dataset_columns = list(state.preprocess_profile.get("columns") or [])
                    except Exception:
                        dataset_columns = []

                    spec = _instruction_specificity(state.instruction, dataset_columns)
                    if use_state_insights_for_report:
                        # Preserve exact ordering and count from interpret_results.
                        displayed = list(verified)
                    else:
                        # Simple de-duplication by normalized statement text
                        seen_norm = set()
                        deduped: List[Dict[str, Any]] = []
                        for v in verified:
                            try:
                                stmt = str(v.get("statement", "")).strip()
                                norm = re.sub(r"\s+", " ", stmt.lower())
                                if not stmt or norm in seen_norm:
                                    continue
                                seen_norm.add(norm)
                                deduped.append(v)
                            except Exception:
                                continue

                        if show_all or simple_mode:
                            displayed = deduped
                        else:
                            # Visualization-friendly: if instruction is viz-only (non-strict), cap to small number (default 1)
                            if viz_only and not viz_minimal:
                                try:
                                    viz_cap = int(os.getenv("ANALYSIS_VIZ_ONLY_MAX_INSIGHTS", "1"))
                                except Exception:
                                    viz_cap = 1
                                viz_cap = max(0, viz_cap)
                                displayed = deduped[:viz_cap]
                            else:
                                # Interpolate display ratio between generic and specific only when needed
                                # spec=0 -> generic ratio; spec=1 -> specific ratio
                                display_ratio = ratio_specific * spec + ratio_generic * (1 - spec)
                                target_count = int(max(min_show, round(display_ratio * len(deduped))))
                                if max_show is not None:
                                    target_count = min(target_count, max_show)
                                displayed = deduped[:target_count]

                    # For insights-only mode, add an LLM-crafted analysis summary first (2–3 lines)
                    if insights_only:
                        lines.append("## Analysis Summary")
                        # Prefer LLM-provided analysis_summary from structured payload
                        summary_text = None
                        try:
                            if isinstance(structured_payload_json, str) and structured_payload_json.strip():
                                import json as _json
                                try:
                                    _payload = _json.loads(structured_payload_json)
                                except Exception:
                                    _payload = None
                                if isinstance(_payload, dict):
                                    cand = _payload.get("analysis_summary")
                                    if isinstance(cand, str) and cand.strip():
                                        summary_text = cand.strip()
                        except Exception:
                            summary_text = None

                        if summary_text:
                            # Last-resort sanitization: remove snapshot-size phrasing like '100 rows' or '100 records'
                            try:
                                summary_text = re.sub(r"\b~?\d+\s*(rows|records)\b", "the dataset", summary_text, flags=re.IGNORECASE)
                            except Exception:
                                pass
                            parts = re.split(r"(?<=[\.!?])\s+", summary_text)
                            parts = [p.strip() for p in parts if p.strip()]
                            if not parts:
                                parts = [summary_text]
                            for s in parts[:3]:
                                lines.append(s)
                            lines.append("")
                        else:
                            # Fallback: synthesize concise sentences from verified insights
                            try:
                                total_verified = len(verified)
                            except Exception:
                                total_verified = 0
                            try:
                                showing_count = len(displayed)
                            except Exception:
                                showing_count = 0
                            sentences: List[str] = []
                            try:
                                source_list = displayed if showing_count > 0 else verified
                            except Exception:
                                source_list = verified
                            for v in (source_list or [])[:3]:
                                try:
                                    stmt = str(v.get("statement") or "").strip()
                                    if not stmt:
                                        continue
                                    stmt = re.sub(r"\s+", " ", stmt)
                                    if len(stmt) > 220:
                                        stmt = stmt[:217] + "…"
                                    if not stmt.endswith(('.', '!', '?')):
                                        stmt = stmt + "."
                                    sentences.append(stmt)
                                except Exception:
                                    continue
                            if len(sentences) < 2:
                                sentences.insert(0, f"{total_verified} verified insights; showing {showing_count} key items.")
                            for s in sentences[:3]:
                                lines.append(s)
                            lines.append("")

                    lines.append("## Key Insights")
                    embedded_artifacts = set()
                    for idx, v in enumerate(displayed, start=1):
                        lines.append(f"- {v['statement']}")
                        if not viz_only:
                            lines.append(f"  - Evidence: \"{v['evidence']}\"")
                            art = v.get("artifact")
                            if art and not insights_only:
                                try:
                                    ext = Path(art).suffix.lower()
                                except Exception:
                                    ext = ""
                                if ext in (".png", ".jpg", ".jpeg", ".svg", ".gif", ".webp", ".pdf"):
                                    # Normalize to simple filename relative to report directory
                                    try:
                                        filename_only = Path(art).name if isinstance(art, (str, Path)) else str(art)
                                    except Exception:
                                        filename_only = str(art)
                                    rel_display = filename_only.replace('\\', '/')
                                    if art not in embedded_artifacts:
                                        lines.append(f"  - Supporting plot: `{rel_display}`")
                                        lines.append("")
                                        lines.append(f"  ![{rel_display}]({rel_display})")
                                        lines.append("")
                                        embedded_artifacts.add(art)
                                    else:
                                        lines.append(f"  - Supporting plot (shown above): `{rel_display}`")
                    remaining = max(0, len(verified) - len(displayed))
                    if remaining > 0:
                        lines.append(f"\n> Note: {remaining} additional verified insights are available in the run metadata but not shown here for readability.")
                    lines.append("")

                # Dataset summary after insights
                # lines.append("## Dataset")
                # lines.append(f"- Path: `{path}`")
                # if isinstance(rows, int):
                #     lines.append(f"- Rows: {rows}")
                # if isinstance(cols, int):
                #     lines.append(f"- Columns: {cols}")
                # if notes:
                #     lines.append(f"- Notes: {notes}")
                # # Clarify snapshot vs full-data execution to avoid confusion in viz-only runs
                # lines.append("- Note: Planning used a small snapshot (~100 rows) for schema grounding; tool execution analyzed the full dataset.")
                # lines.append("")

                # Verify and render traffic engineering recommendations (optional)
                if recs:
                    verified_recs = []
                    for r in recs:
                        try:
                            rec_text = str(r.get("recommendation") or "").strip()
                            ev = str(r.get("evidence_snippet") or "").strip()
                            src = str(r.get("evidence_source") or "transcript").strip().lower()
                            art = r.get("artifact")
                            cat = (r.get("category") or "operations").strip()
                            ref = r.get("evidence_ref") or r.get("ref")
                            ok, trace, err = _verify_with_ref(ev, ref)
                            if not ok:
                                # When enforcement is OFF, keep recommendation without strict trace
                                if not enforce_trace:
                                    verified_recs.append({"recommendation": rec_text, "category": cat, "evidence": ev, "artifact": art})
                                    continue
                                # With enforcement ON, attempt fallback verification
                                if _fallback_verify(ev, src, art):
                                    verified_recs.append({"recommendation": rec_text, "category": cat, "evidence": ev, "artifact": art})
                                    continue
                                if err == "invalid_line_index":
                                    t_invalid_index += 1
                                if err == "artifact_not_found":
                                    t_artifact_missing += 1
                                # drop unverified recommendation
                                continue
                            verified_recs.append({"recommendation": rec_text, "category": cat, "evidence": ev, "artifact": trace.get("artifact") if trace else art})
                        except Exception:
                            continue
                    if verified_recs:
                        lines.append("## Traffic Engineering Recommendations")
                        for r in verified_recs:
                            lines.append(f"- ({r['category']}) {r['recommendation']}")
                            if not (viz_only and not viz_minimal):
                                lines.append(f"  - Evidence: \"{r['evidence']}\"")
                                if r.get("artifact"):
                                    lines.append(f"  - Artifact: `{r['artifact']}`")
                        lines.append("")

                # Skip artifact section for insights-only route (no plots desired in report)
                if artifact_basenames and not insights_only:
                    lines.append("## Artifacts")
                    for a in sorted(list(artifact_basenames)):
                        lines.append(f"- {a}")
                    lines.append("")

                if next_steps:
                    # Filter out internal tool mentions for user-friendly next steps
                    filtered_steps = []
                    tool_names_lower = {str(n).lower() for n in self._tool_map.keys()}
                    for s in next_steps:
                        txt = str(s or "").strip()
                        low = txt.lower()
                        if not txt:
                            continue
                        if low.startswith("perform_"):
                            # Hide raw tool function names
                            continue
                        if any(tn in low for tn in tool_names_lower):
                            # Hide steps that reference tool identifiers directly
                            continue
                        filtered_steps.append(txt)
                    if filtered_steps:
                        lines.append("## Next Steps")
                        for s in filtered_steps[:6]:
                            lines.append(f"1. {s}")
                        lines.append("")

                report_text = "\n".join(lines)

            if report_text is None:
                # Fallback to raw text behavior (legacy)
                report_text = raw

            # Cache last good synthesis result for resilience on final pass
            try:
                if isinstance(report_text, str) and report_text.strip():
                    self._last_good_report_text = report_text
                if isinstance(structured_payload_json, str) and structured_payload_json:
                    self._last_good_structured_json = structured_payload_json
            except Exception:
                pass

            # Attach warning if insights were dropped
            warnings_local: List[str] = []
            try:
                if t_dropped > 0:
                    warnings_local.append("insights_dropped_traceability")
                if t_invalid_index > 0:
                    warnings_local.append("invalid_line_index")
                if t_artifact_missing > 0:
                    warnings_local.append("artifact_not_found")
            except Exception:
                warnings_local = []

            self._logger.info(f"synthesis: generated {len(report_text)} char report (grounded={report_text is not raw})")
            self._dprint(f"Synthesis grounded={report_text is not raw}, chars={len(report_text)}")
            
        except Exception as exc:
            self._logger.error(f"synthesis: LLM invocation failed ({exc})")
            # Use last good report if available; otherwise fallback message
            if isinstance(self._last_good_report_text, str) and self._last_good_report_text.strip():
                report_text = self._last_good_report_text
                try:
                    structured_payload_json = self._last_good_structured_json
                except Exception:
                    pass
                self._logger.warning("synthesis: Using cached last-good report due to failure this pass")
            else:
                report_text = "**Analysis Report**\n\nSynthesis failed. Please review tool outputs manually."
        
        # Build artifacts dict from state (back-compat + user-facing artifacts)
        artifacts: Dict[str, Any] = {
            "artifact_log": state.artifact_log,
            "plan_steps": [{"tool": s.get("tool"), "why": s.get("why")} for s in state.plan_steps],
            "token_metrics": state.token_metrics,
            "user_instruction": state.instruction,
        }
        # Attach structured payload JSON (if available) for downstream consumers and optional sidecar writes
        try:
            if 'structured_payload_json' in locals() and structured_payload_json:
                artifacts["structured_payload_json"] = structured_payload_json
        except Exception:
            pass
        # Include basic traceability telemetry when available
        try:
            telemetry_local = {}
            if 't_total' in locals():
                telemetry_local.update({
                    "insights_total": t_total,
                    "insights_verified": t_verified,
                    "insights_dropped_traceability": t_dropped,
                    "invalid_line_index": t_invalid_index,
                    "artifact_not_found": t_artifact_missing,
                })
                # Include adaptive display telemetry when available
                try:
                    if 'target_count' in locals():
                        telemetry_local.update({
                            "insights_displayed": int(target_count),
                        })
                    if 'display_ratio' in locals():
                        telemetry_local.update({
                            "insights_display_ratio": float(display_ratio),
                        })
                    if 'spec' in locals():
                        telemetry_local.update({
                            "instruction_specificity": float(spec),
                        })
                except Exception:
                    pass
            if telemetry_local:
                artifacts.setdefault("telemetry", {}).update(telemetry_local)
            # Include prefilter telemetry if available
            try:
                if 't_prefilter_dropped' in locals():
                    artifacts.setdefault("telemetry", {}).update({
                        "insights_dropped_prefilter": t_prefilter_dropped
                    })
            except Exception:
                pass
        except Exception:
            pass
        if state.dataset_path:
            artifacts["dataset_path"] = state.dataset_path

        # Compact steps provenance for envelope
        steps: list[dict] = []
        for idx, s in enumerate(state.plan_steps):
            steps.append({
                "tool": s.get("tool"),
                "completed": idx < state.next_step_index,
                "desc": s.get("why", "")[:240],
            })

        # Tool call summaries (no CoT, compact outputs)
        tool_summaries: list[dict] = []
        for ev in state.tool_transcript[-10:]:  # last 10 events
            tool_summaries.append({
                "tool": ev.get("tool"),
                "status": ev.get("status"),
                "note": (ev.get("output") or "")[:240],
            })

        # Optional short summary from report_text
        summary = None
        if isinstance(report_text, str) and report_text:
            first_para = report_text.split("\n\n", 1)[0].strip()
            summary = first_para[:240]

        errors_list = [ev.get("output") for ev in state.tool_transcript if ev.get("status") == "error"]
        status_val = "error" if errors_list else "ok"
        # If refinement capped without answering, escalate status
        if status_val == "ok" and getattr(state, "refinement_round", 0) > 0 and not state.coverage.get("answers_query"):
            status_val = "error"

        # Construct the output envelope first
        output = PipelineOutputState(
            status=status_val,
            title=None,
            report_text=report_text,
            summary=summary,
            artifacts=artifacts,
            dataset_path=state.dataset_path,
            steps=steps,
            tool_summaries=tool_summaries,
            token_metrics=state.token_metrics,
            warnings=(warnings_local if report_text is not None and 'warnings_local' in locals() else []) + ([] if status_val == "ok" else ["analysis_incomplete"]),
            errors=errors_list,
        )

        # Persist side-effects here (single terminal step):
        # - Append minimal run metadata to memory store
        # - Save Markdown report to disk (guarded against duplicate writes per run unless incremental enabled)
        try:
            # Prefer state.dataset_path; keep compatibility with artifacts field
            dataset_path = state.dataset_path or output.artifacts.get("dataset_path")

            # Determine if this is the final synthesis pass for this run. We allow
            # a final-pass write to overwrite any earlier report even when incremental
            # reports are disabled, so the saved file reflects the richest result.
            try:
                plan_len = len(state.plan_steps or [])
                step_idx = int(getattr(state, "next_step_index", 0) or 0)
                is_last_step = plan_len > 0 and step_idx >= (plan_len - 1)
            except Exception:
                is_last_step = False
            # Forced termination when reflection hit the cap or an explicit finalize flag is set
            try:
                current_refine_round = int(getattr(state, "refinement_round", 0) or 0)
            except Exception:
                current_refine_round = 0
            try:
                max_refinements_cfg = int(os.getenv("ANALYSIS_MAX_REFINEMENTS", "2"))
            except Exception:
                max_refinements_cfg = 2
            forced_termination_now = current_refine_round >= max_refinements_cfg or bool(getattr(state, "finalize", False))
            # Final pass should occur only after the last step or forced termination in single-pass mode
            # Do not end early on answers_query; we synthesize once at the end of execution.
            is_final_pass = is_last_step or forced_termination_now

            # Schedule memory append (fire-and-forget)
            if dataset_path:
                # Build metadata from the *real* state so HITL decisions and other
                # run-level fields (tool transcript, issues, etc.) are persisted.
                # Fall back to envelope values only when state is missing them.
                state_updates: Dict[str, Any] = {}
                try:
                    if dataset_path and not getattr(state, "dataset_path", None):
                        state_updates["dataset_path"] = dataset_path
                except Exception:
                    state_updates["dataset_path"] = dataset_path

                try:
                    if not getattr(state, "instruction", "") and output.artifacts.get("user_instruction"):
                        state_updates["instruction"] = output.artifacts.get("user_instruction", "")
                except Exception:
                    pass

                try:
                    if not (getattr(state, "artifact_log", None) or []) and output.artifacts.get("artifact_log"):
                        state_updates["artifact_log"] = output.artifacts.get("artifact_log", [])
                except Exception:
                    pass

                try:
                    if not (getattr(state, "plan_steps", None) or []) and getattr(output, "steps", None):
                        state_updates["plan_steps"] = [
                            {"tool": s.get("tool"), "args": {}, "why": s.get("desc", "")} for s in output.steps
                        ]
                except Exception:
                    pass

                if state_updates:
                    try:
                        state_for_metadata = state.model_copy(update=state_updates)
                    except Exception:
                        state_for_metadata = AnalysisPipelineState(**{**state.dict(), **state_updates})  # type: ignore[attr-defined]
                else:
                    state_for_metadata = state

                # Also attach HITL info to the output envelope for downstream stages.
                try:
                    if output and isinstance(output.artifacts, dict):
                        reviewed = bool(getattr(state, "hitl_plan_reviewed", False))
                        decision = getattr(state, "hitl_plan_decision", None)
                        feedback = getattr(state, "hitl_plan_feedback", None)
                        if reviewed or decision or feedback:
                            output.artifacts["hitl_plan_review"] = {
                                "reviewed": reviewed,
                                "decision": decision,
                                "feedback": feedback,
                                "plan_hash": getattr(state, "plan_hash", None),
                            }
                except Exception:
                    pass

                metadata = build_run_metadata(state_for_metadata, output.report_text)
                self._memory_store.schedule_append(metadata)
                self._logger.info("synthesis: memory append scheduled")

            # Duplicate write guards
            allow_multi = str(os.getenv("ANALYSIS_INCREMENTAL_REPORTS", "0")).lower() in ("1", "true", "yes")
            try:
                write_final_only = str(os.getenv("ANALYSIS_WRITE_FINAL_ONLY", "1")).lower() in ("1", "true", "yes")
            except Exception:
                write_final_only = True

            # Single-run latch: only when explicitly requested (stop_after_first_synthesis).
            try:
                if bool(getattr(state, "stop_after_first_synthesis", False)):
                    if getattr(self, "_single_run_guard_enabled", False):
                        self._single_run_completed = True
                        # cached output will be set after we construct it; guard latching prevents extra logs
            except Exception:
                pass

            # If configured, skip writing until the final pass to avoid multiple intermediate files/overwrites
            if write_final_only and not is_final_pass:
                self._logger.info("synthesis: ANALYSIS_WRITE_FINAL_ONLY enabled; skipping write until final pass")
                return output
            if not allow_multi and self._current_run_id and self._current_run_id in self._reports_written:
                if not is_final_pass:
                    self._logger.info("synthesis: report already saved for this run_id, skipping duplicate write")
                    return output
                else:
                    self._logger.info("synthesis: final-pass override enabled; overwriting previously saved report for this run_id")

            # Idempotency within the output envelope
            try:
                already_saved = bool(output.artifacts.get("__report_saved"))
            except Exception:
                already_saved = False
            if already_saved and not allow_multi and not is_final_pass:
                self._logger.info("synthesis: report already saved, skipping duplicate write")
                return output
            elif already_saved and not allow_multi and is_final_pass:
                self._logger.info("synthesis: final-pass override enabled; overwriting previously saved report (artifact flag)")

            # Write Markdown report to configured reports directory
            cfg = self._get_config()
            report_dir: Path = cfg.reports_path
            report_dir.mkdir(parents=True, exist_ok=True)

            single_report = str(os.getenv("ANALYSIS_SINGLE_REPORT", "0")).lower() in ("1", "true", "yes")
            try:
                dataset_name = Path(dataset_path).stem if dataset_path else "no_dataset"
            except Exception:
                dataset_name = "no_dataset"

            if single_report:
                instr = output.artifacts.get("user_instruction", "") or ""
                sig_basis = f"{dataset_name}|{instr}".lower().strip()
                sig = hashlib.sha256(sig_basis.encode()).hexdigest()[:12]
                report_filename = f"analysis_report_{dataset_name}_{sig}.md"
            else:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                report_filename = f"analysis_report_{dataset_name}_{ts}.md"

            report_path = report_dir / report_filename
            with report_path.open("w", encoding="utf-8") as f:
                f.write(output.report_text or "")

            # Optionally write a JSON sidecar containing the structured payload
            try:
                save_structured = str(os.getenv("ANALYSIS_SAVE_STRUCTURED_JSON", "1")).lower() in ("1", "true", "yes")
            except Exception:
                save_structured = True
            if save_structured:
                try:
                    payload_json = output.artifacts.get("structured_payload_json")
                except Exception:
                    payload_json = None
                if isinstance(payload_json, str) and payload_json:
                    try:
                        sidecar_path = report_path.with_suffix(".json")
                        with sidecar_path.open("w", encoding="utf-8") as jf:
                            jf.write(payload_json)
                        self._logger.info(f"synthesis: structured payload sidecar saved to {sidecar_path}")
                    except Exception as _se:
                        self._logger.warning(f"synthesis: failed to save structured payload sidecar ({_se})")

            # NEW: Optionally write the final contents of state.insights into a separate Markdown file
            # so users can compare interpret_results insights vs the rendered report.
            try:
                save_state_insights = str(os.getenv("ANALYSIS_SAVE_STATE_INSIGHTS", "1")).lower() in ("1", "true", "yes")
            except Exception:
                save_state_insights = True
            # Skip state insights sidecar for visualization-only runs
            if save_state_insights and (not viz_minimal):
                try:
                    insights_list = getattr(state, "insights", None)

                except Exception:
                    try:
                        total_insights = len(insights_list) if isinstance(insights_list, list) else 0
                        # Insert after header + blank line
                        lines.insert(2, f"Total insights: {total_insights}")
                        lines.insert(3, "")
                    except Exception:
                        pass

                    insights_list = None
                try:
                    coverage_obj = getattr(state, "coverage", None)
                except Exception:
                    coverage_obj = None

            # NEW: Optionally write the tool transcript into a separate Markdown file
            # so users can prove planner de-duplication and inspect exact tool/args/status history.
            try:
                save_tool_transcript = str(os.getenv("ANALYSIS_SAVE_TOOL_TRANSCRIPT", "1")).lower() in ("1", "true", "yes")
            except Exception:
                save_tool_transcript = True
            if save_tool_transcript and (not viz_minimal):
                try:
                    tt = getattr(state, "tool_transcript", None)
                except Exception:
                    tt = None
                if isinstance(tt, list) and tt:
                    try:
                        tt_path = report_path.with_suffix(".tool_transcript.md")
                        lines: list[str] = []
                        lines.append("# Tool Transcript")
                        lines.append("")
                        lines.append(f"Total events: {len(tt)}")
                        lines.append("")
                        lines.append("Notes:")
                        lines.append("- `output` may be summarized/truncated for very large tool results.")
                        lines.append("- If present, `raw_output_artifact` points to the full saved tool output.")
                        lines.append("")

                        for idx, ev in enumerate(tt, start=1):
                            if not isinstance(ev, dict):
                                continue
                            tool = str(ev.get("tool") or "")
                            status = str(ev.get("status") or "")
                            timestamp = str(ev.get("timestamp") or "")
                            why = str(ev.get("why") or "")
                            artifact = ev.get("artifact")
                            raw_art = ev.get("raw_output_artifact")
                            artifacts_list = ev.get("artifacts")
                            args_obj = ev.get("args") or {}
                            out_text = ev.get("output") or ev.get("result") or ""

                            lines.append(f"## Event {idx}")
                            if tool:
                                lines.append(f"- Tool: `{tool}`")
                            if status:
                                lines.append(f"- Status: {status}")
                            if timestamp:
                                lines.append(f"- Timestamp: {timestamp}")
                            if why:
                                lines.append(f"- Why: {why}")
                            if artifact:
                                lines.append(f"- Artifact: `{artifact}`")
                            if isinstance(artifacts_list, list) and artifacts_list:
                                try:
                                    safe_names = [str(a) for a in artifacts_list if a]
                                    if safe_names:
                                        lines.append(f"- Artifacts: {', '.join('`' + n + '`' for n in safe_names[:20])}")
                                except Exception:
                                    pass
                            if raw_art:
                                lines.append(f"- Raw output artifact: `{raw_art}`")
                            lines.append("")

                            # Args (stable json for easy diffing)
                            try:
                                args_json = json.dumps(args_obj, ensure_ascii=False, sort_keys=True, indent=2, default=str)
                            except Exception:
                                args_json = str(args_obj)
                            lines.append("Args:")
                            lines.append("```json")
                            lines.append(args_json)
                            lines.append("```")
                            lines.append("")

                            # Output
                            try:
                                out_str = str(out_text)
                            except Exception:
                                out_str = ""
                            lines.append("Output:")
                            lines.append("```text")
                            lines.append(out_str)
                            lines.append("```")
                            lines.append("")

                        with tt_path.open("w", encoding="utf-8") as tf:
                            tf.write("\n".join(lines).rstrip() + "\n")
                        self._logger.info(f"synthesis: tool transcript sidecar saved to {tt_path}")
                    except Exception as _tte:
                        self._logger.warning(f"synthesis: failed to save tool transcript sidecar ({_tte})")
                try:
                    si_path = report_path.with_suffix(".state_insights.md")
                    lines = ["# State Insights (pre-synthesis)", ""]
                    # Coverage summary if available
                    if isinstance(coverage_obj, dict):
                        ans = bool(coverage_obj.get("answers_query"))
                        missing = coverage_obj.get("missing_info") or []
                        lines.append(f"- answers_query: {ans}")
                        # Audit: record whether reflection triggered a reinterpretation retry
                        try:
                            ra = int(getattr(state, "reinterpret_attempts", 0) or 0)
                        except Exception:
                            ra = 0
                        lines.append(f"- reinterpret_attempts: {ra}")
                        if ra > 0:
                            lines.append("- note: reflect routed back to interpret_results to recover missed insights")
                        if missing:
                            lines.append("- missing_info:")
                            for m in missing[:10]:
                                try:
                                    lines.append(f"  - {str(m)[:300]}")
                                except Exception:
                                    continue
                        # Capability-gap details (debug only; not included in user-facing report)
                        try:
                            gap = getattr(state, "capability_gap", None)
                        except Exception:
                            gap = None
                        if isinstance(gap, dict) and gap:
                            lines.append("")
                            lines.append("## Capability Gap")
                            try:
                                nmt = gap.get("needs_more_tools")
                            except Exception:
                                nmt = None
                            if nmt is not None:
                                lines.append(f"- needs_more_tools: {bool(nmt)}")
                            try:
                                root = str(gap.get("root_cause") or "").strip()
                            except Exception:
                                root = ""
                            if root:
                                lines.append(f"- root_cause: {root}")
                            try:
                                why = str(gap.get("why_cannot_answer") or "").strip()
                            except Exception:
                                why = ""
                            if why:
                                lines.append(f"- why_cannot_answer: {why}")
                            proposed = gap.get("proposed_tools") if isinstance(gap.get("proposed_tools"), list) else []
                            if proposed:
                                lines.append("- proposed_tools:")
                                for t in proposed[:3]:
                                    if not isinstance(t, dict):
                                        continue
                                    tname = str(t.get("tool_name") or t.get("name") or "").strip() or "Proposed tool"
                                    purpose = str(t.get("purpose") or t.get("what_it_does") or "").strip()
                                    why_needed = str(t.get("why_needed") or "").strip()
                                    goal = str(t.get("goal") or "").strip()
                                    inputs = t.get("inputs") if isinstance(t.get("inputs"), list) else []
                                    outputs = t.get("outputs") if isinstance(t.get("outputs"), list) else []
                                    lines.append(f"  - {tname}")
                                    if goal:
                                        lines.append(f"    - goal: {goal}")
                                    if purpose:
                                        lines.append(f"    - purpose: {purpose}")
                                    if why_needed:
                                        lines.append(f"    - why_needed: {why_needed}")
                                    if inputs:
                                        lines.append(f"    - inputs: {', '.join([str(i) for i in inputs[:8]])}")
                                    if outputs:
                                        lines.append(f"    - outputs: {', '.join([str(o) for o in outputs[:8]])}")
                        lines.append("")
                    # Insights list from interpret_results
                    count = 0
                    if isinstance(insights_list, list):
                        for idx, it in enumerate(insights_list, start=1):
                            if not isinstance(it, dict):
                                continue
                            count += 1
                            stmt = str(it.get("statement") or "").strip()
                            ev = str(it.get("evidence_snippet") or it.get("evidence") or "").strip()
                            src = str(it.get("evidence_source") or "").strip()
                            ref = it.get("evidence_ref")
                            lines.append(f"## Insight {idx}")
                            lines.append(f"- Statement: {stmt}")
                            if ev:
                                lines.append(f"- Evidence: \"{ev}\"")
                            if src:
                                lines.append(f"- Source: {src}")
                            if ref is not None:
                                try:
                                    ref_json = json.dumps(ref, ensure_ascii=False)
                                except Exception:
                                    ref_json = str(ref)
                                lines.append(f"- Ref: {ref_json}")
                            lines.append("")
                    else:
                        lines.append("(No insights captured in state)")
                        lines.append("")
                    # Summary
                    lines.insert(1, f"Total insights: {count}")
                    with si_path.open("w", encoding="utf-8") as fsi:
                        fsi.write("\n".join(lines))
                    self._logger.info(f"synthesis: state insights markdown saved to {si_path}")
                except Exception as _ie:
                    self._logger.warning(f"synthesis: failed to save state insights markdown ({_ie})")

            # Attach path and mark saved
            try:
                output.artifacts["report_path"] = str(report_path)
                output.artifacts["__report_saved"] = True
            except Exception:
                pass

            # Mark run-level write complete
            try:
                if self._current_run_id:
                    # Keep marker to prevent non-final duplicate writes; final-pass override will still proceed
                    self._reports_written.add(self._current_run_id)
            except Exception:
                pass

            self._logger.info(f"synthesis: report saved to {report_path}")
        except Exception as e:
            self._logger.warning(f"synthesis: failed to persist report ({e})")

        # Latch single-run guard after first synthesis completion and cache output
        try:
            if getattr(self, "_single_run_guard_enabled", False):
                self._single_run_completed = True
                self._cached_final_output = output
        except Exception:
            pass
        return output
    # (UNREACHABLE) - keep consistent

    async def persist_cleanup(self, output: PipelineOutputState) -> PipelineOutputState:
        """Finalize artifacts and enqueue memory persistence."""
        try:
            rid = self._current_run_id or "-"
            self._logger.info(f"run={rid} node=persist_cleanup begin")
        except Exception:
            pass
        # Run-level duplicate write guard (allow final-pass override below)
        try:
            allow_multi = str(os.getenv("ANALYSIS_INCREMENTAL_REPORTS", "0")).lower() in ("1", "true", "yes")
            # Determine final-pass status similar to synthesis
            try:
                plan_len_pc = len(getattr(output, "steps", []) or [])
            except Exception:
                plan_len_pc = 0
            # We don't have state here; rely on artifact hints when available
            # Final-pass override in persist_cleanup will be handled after checking artifact flags
            if not allow_multi and self._current_run_id and self._current_run_id in self._reports_written:
                self._logger.info("persist_cleanup: report already saved for this run_id, skipping duplicate write")
                return output
        except Exception:
            pass
        # Prefer envelope dataset_path; fallback to artifact field
        dataset_path = output.dataset_path or output.artifacts.get("dataset_path")
        
        # Build and schedule memory append (async fire-and-forget)
        if dataset_path:
            # Reconstruct a minimal state snapshot for metadata
            hitl = None
            try:
                hitl = output.artifacts.get("hitl_plan_review") if isinstance(output.artifacts, dict) else None
            except Exception:
                hitl = None

            temp_state = AnalysisPipelineState(
                dataset_path=dataset_path,
                instruction=output.artifacts.get("user_instruction", ""),
                artifact_log=output.artifacts.get("artifact_log", []),
                plan_steps=[{"tool": s.get("tool"), "args": {}, "why": s.get("desc", "")} for s in output.steps],
                hitl_plan_reviewed=bool(hitl.get("reviewed")) if isinstance(hitl, dict) else False,
                hitl_plan_decision=str(hitl.get("decision")) if isinstance(hitl, dict) and hitl.get("decision") is not None else None,
                hitl_plan_feedback=str(hitl.get("feedback")) if isinstance(hitl, dict) and hitl.get("feedback") is not None else None,
                plan_hash=str(hitl.get("plan_hash")) if isinstance(hitl, dict) and hitl.get("plan_hash") is not None else None,
            )
            metadata = build_run_metadata(temp_state, output.report_text)
            self._memory_store.schedule_append(metadata)
            self._logger.info("persist_cleanup: memory append scheduled")

        # Save report to Markdown file following the original agent's template
        try:
            # Idempotency: avoid saving multiple reports per run unless explicitly allowed
            allow_multi = str(os.getenv("ANALYSIS_INCREMENTAL_REPORTS", "0")).lower() in ("1", "true", "yes")
            try:
                already_saved = bool(output.artifacts.get("__report_saved"))
            except Exception:
                already_saved = False
            if already_saved and not allow_multi:
                # In persist_cleanup we don't have fine-grained step context; respect the artifact flag
                self._logger.info("persist_cleanup: report already saved, skipping duplicate write")
                return output

            cfg = self._get_config()
            report_dir: Path = cfg.reports_path
            report_dir.mkdir(parents=True, exist_ok=True)

            # Single final report mode writes to a signature-based filename (overwrites)
            single_report = str(os.getenv("ANALYSIS_SINGLE_REPORT", "0")).lower() in ("1", "true", "yes")
            try:
                self._logger.info(f"run={self._current_run_id or '-'} persist_cleanup single_report={single_report} allow_multi={allow_multi}")
            except Exception:
                pass
            try:
                dataset_name = Path(dataset_path).stem if dataset_path else "no_dataset"
            except Exception:
                dataset_name = "no_dataset"
            if single_report:
                instr = output.artifacts.get("user_instruction", "") or ""
                sig_basis = f"{dataset_name}|{instr}".lower().strip()
                sig = hashlib.sha256(sig_basis.encode()).hexdigest()[:12]
                report_filename = f"analysis_report_{dataset_name}_{sig}.md"
            else:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                report_filename = f"analysis_report_{dataset_name}_{ts}.md"

            report_path = report_dir / report_filename
            # Content: use report_text directly to preserve Markdown
            with report_path.open("w", encoding="utf-8") as f:
                f.write(output.report_text or "")
            # Attach path to artifacts for downstream consumers
            try:
                output.artifacts["report_path"] = str(report_path)
                output.artifacts["__report_saved"] = True
            except Exception:
                pass
            # Mark run-level write complete
            try:
                if self._current_run_id:
                    self._reports_written.add(self._current_run_id)
            except Exception:
                pass
            self._logger.info(f"persist_cleanup: report saved to {report_path}")
        except Exception as e:
            self._logger.warning(f"persist_cleanup: failed to save report markdown ({e})")
        
        try:
            self._logger.info(f"run={self._current_run_id or '-'} node=persist_cleanup end")
        except Exception:
            pass
        return output

    # ------------------------------------------------------------------
    # A2A entrypoint wiring
    # ------------------------------------------------------------------
    
    async def _reason_about_existing_data(self, state: AnalysisPipelineState, missing_info: list[str]) -> dict:
        """Use LLM to intelligently reason if existing tool outputs can answer missing info.
        
        This prevents wasteful replanning when we already have sufficient data but the
        interpret_results node marked things as missing.
        
        Args:
            state: Current pipeline state with tool_transcript and instruction
            missing_info: List of supposedly missing information items
            
        Returns:
            dict with keys:
                - can_answer (bool): True if existing data is sufficient
                - reasoning (str): Explanation of why/why not
        """
        try:
            # Build a summary of what we have
            transcript = state.tool_transcript or []
            successful_outputs = [
                e for e in transcript 
                if isinstance(e, dict) and e.get("status") == "ok"
            ]
            
            if not successful_outputs:
                return {"can_answer": False, "reasoning": "No successful tool outputs to work with"}
            
            # Summarize tool outputs (limit to prevent token overflow)
            tool_summary = []
            for i, entry in enumerate(successful_outputs[:10]):  # Max 10 outputs
                tool_name = entry.get("tool") or entry.get("tool_name") or "unknown"
                output = str(entry.get("output", ""))[:TOOL_OUTPUT_MAX_CHARS]  # Truncate large outputs
                tool_summary.append(f"Tool {i+1} ({tool_name}): {output}")
            
            reasoning_prompt = f"""You are analyzing whether existing tool outputs can answer a data analysis question.

ORIGINAL INSTRUCTION:
{state.instruction or 'N/A'}

DATASET COLUMNS (available):
{', '.join(state.preprocess_profile.get('columns', [])[:20]) if state.preprocess_profile else 'N/A'}

TOOL OUTPUTS WE ALREADY HAVE:
{chr(10).join(tool_summary)}

SUPPOSEDLY MISSING INFORMATION:
{chr(10).join(f"- {m}" for m in missing_info[:5])}

QUESTION: Can you answer the original instruction using ONLY the existing tool outputs and your reasoning?
Consider:
1. Do the tool outputs contain the data needed to answer?
2. Can you reason/calculate from what's there?
3. Is replanning truly necessary or can synthesis work with this?

Respond in JSON:
{{
    "can_answer": true/false,
    "reasoning": "brief explanation (max 200 chars)"
}}"""

            # Call LLM for reasoning
            messages = [{"role": "user", "content": reasoning_prompt}]
            
            try:
                response = await self._llm.ainvoke(messages)
                content = response.content if hasattr(response, 'content') else str(response)
                
                # Try to parse JSON response
                import json
                import re
                
                # Extract JSON from response (handle markdown code blocks)
                json_match = re.search(r'\{[^{}]*"can_answer"[^{}]*\}', content, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group(0))
                    return {
                        "can_answer": bool(result.get("can_answer", False)),
                        "reasoning": str(result.get("reasoning", ""))[:200]
                    }
                
                # Fallback: look for keywords in response
                if any(word in content.lower() for word in ["yes", "can answer", "sufficient", "enough"]):
                    return {
                        "can_answer": True,
                        "reasoning": "LLM indicated existing data is sufficient"
                    }
                    
            except Exception as e:
                self._logger.warning(f"_reason_about_existing_data: LLM call failed: {e}")
            
            # Conservative fallback: don't claim we can answer if reasoning failed
            return {
                "can_answer": False,
                "reasoning": "Reasoning step failed, defaulting to replan for safety"
            }
            
        except Exception as e:
            self._logger.error(f"_reason_about_existing_data: unexpected error: {e}")
            return {
                "can_answer": False,
                "reasoning": f"Error in reasoning: {str(e)[:100]}"
            }

    async def _diagnose_tool_gap(self, state: AnalysisPipelineState, missing_info: list[str]) -> dict:
        """Explain *why* the query isn't answered and whether it requires new tools.

        This is used when reflection cannot construct actionable refinement targets
        for the current toolset, or when refinement is capped.

        Returns a structured dict suitable for storing in state.capability_gap.
        """
        # Avoid repeated diagnosis if already present
        try:
            existing = getattr(state, "capability_gap", None)
            if isinstance(existing, dict) and existing:
                return existing
        except Exception:
            pass

        # Build tool catalog (name + description + arg keys) for the LLM.
        tool_lines: list[str] = []
        try:
            for name, tool in (self._tool_map or {}).items():
                desc = ""
                try:
                    desc = str(getattr(tool, "description", "") or "")
                except Exception:
                    desc = ""
                arg_keys: list[str] = []
                try:
                    if hasattr(tool, "args_schema") and hasattr(tool.args_schema, "model_fields"):
                        arg_keys = list(tool.args_schema.model_fields.keys())
                except Exception:
                    arg_keys = []
                if desc:
                    tool_lines.append(f"- {name}: {desc} (args: {', '.join(arg_keys) if arg_keys else 'n/a'})")
                else:
                    tool_lines.append(f"- {name}: (args: {', '.join(arg_keys) if arg_keys else 'n/a'})")
        except Exception:
            tool_lines = []

        # Summarize recent successful tool outputs for grounding.
        transcript = state.tool_transcript or []
        successful = [e for e in transcript if isinstance(e, dict) and e.get("status") == "ok"]
        tool_summary: list[str] = []
        for i, entry in enumerate(successful[-6:]):
            try:
                tname = entry.get("tool") or entry.get("tool_name") or "unknown"
                out = str(entry.get("output") or entry.get("result") or "")
                tool_summary.append(f"{i+1}. {tname}: {out[:800]}")
            except Exception:
                continue

        cols_preview = "N/A"
        try:
            if isinstance(state.preprocess_profile, dict):
                cols = state.preprocess_profile.get("columns") or []
                if isinstance(cols, list):
                    cols_preview = ", ".join([str(c) for c in cols[:30]])
        except Exception:
            cols_preview = "N/A"

        missing_preview = "\n".join([f"- {str(m)[:200]}" for m in (missing_info or [])[:8]])
        tools_preview = "\n".join(tool_lines[:80])
        outputs_preview = "\n".join(tool_summary) if tool_summary else "(none)"

        prompt = (
            "You are diagnosing why an analysis agent cannot fully answer a user query.\n\n"
            "Your task: decide whether the CURRENT AVAILABLE TOOLS are capable of answering the instruction.\n"
            "If not capable, propose what NEW TOOL CAPABILITY is needed: describe it, why it's needed, and what goal it fulfills.\n\n"
            f"INSTRUCTION:\n{state.instruction or ''}\n\n"
            f"DATASET COLUMNS (preview):\n{cols_preview}\n\n"
            "AVAILABLE TOOLS (name, description, args):\n"
            f"{tools_preview}\n\n"
            "SUCCESSFUL TOOL OUTPUTS WE ALREADY HAVE (preview):\n"
            f"{outputs_preview}\n\n"
            "MISSING INFO / UNANSWERED TARGETS (from reflection/interpretation):\n"
            f"{missing_preview}\n\n"
            "Return ONLY JSON with this schema:\n"
            "{\n"
            "  \"needs_more_tools\": true|false,\n"
            "  \"root_cause\": \"tool_gap\"|\"insufficient_outputs\"|\"data_not_present\"|\"invalid_tool_calls\"|\"other\",\n"
            "  \"why_cannot_answer\": \"<clear explanation, max 500 chars>\",\n"
            "  \"proposed_tools\": [\n"
            "    {\n"
            "      \"tool_name\": \"<descriptive name, not a code identifier>\",\n"
            "      \"goal\": \"<what missing goal it fulfills>\",\n"
            "      \"purpose\": \"<what it does in plain language>\",\n"
            "      \"why_needed\": \"<why current tools can't do it>\",\n"
            "      \"inputs\": [\"<input1>\", \"<input2>\"],\n"
            "      \"outputs\": [\"<output1>\", \"<output2>\"]\n"
            "    }\n"
            "  ],\n"
            "  \"confidence\": 0.0,\n"
            "  \"notes\": \"<optional, max 200 chars>\"\n"
            "}\n"
            "Rules:\n"
            "- If needs_more_tools=false, keep proposed_tools empty or include only optional nice-to-haves.\n"
            "- Do NOT ask for new tools if the answer is achievable with existing tools + different arguments.\n"
        )

        try:
            resp = await self._llm.ainvoke([{"role": "user", "content": prompt}])
            raw = resp.content if hasattr(resp, "content") else str(resp)
            payload = self._extract_json(raw)
            if isinstance(payload, dict) and ("needs_more_tools" in payload):
                # Normalize fields defensively
                out = {
                    "needs_more_tools": bool(payload.get("needs_more_tools", False)),
                    "root_cause": str(payload.get("root_cause") or "other")[:40],
                    "why_cannot_answer": str(payload.get("why_cannot_answer") or "")[:500],
                    "proposed_tools": payload.get("proposed_tools") if isinstance(payload.get("proposed_tools"), list) else [],
                    "confidence": float(payload.get("confidence") or 0.0) if isinstance(payload.get("confidence"), (int, float, str)) else 0.0,
                    "notes": str(payload.get("notes") or "")[:200],
                }
                return out
        except Exception as e:
            self._logger.warning(f"_diagnose_tool_gap: failed to call/parse LLM ({e})")

        # Conservative fallback: declare uncertainty but provide a structured gap.
        return {
            "needs_more_tools": True,
            "root_cause": "tool_gap",
            "why_cannot_answer": "The current toolset could not be mapped to the missing targets (non-actionable missing_info).",
            "proposed_tools": [
                {
                    "tool_name": "Domain-specific analysis tool (proposed)",
                    "goal": "Provide the missing analysis capability requested by the instruction.",
                    "purpose": "Compute or extract the missing evidence not obtainable from existing tools.",
                    "why_needed": "No available tool description/schema matches the missing target.",
                    "inputs": ["file_path", "target_columns"],
                    "outputs": ["result_table", "summary_text"],
                }
            ],
            "confidence": 0.3,
            "notes": "Fallback diagnosis; LLM reasoning unavailable.",
        }
    
    def _get_cached_dataframe(self, dataset_path: str, nrows: Optional[int] = None) -> Optional[Any]:
        """Get DataFrame from cache or load and cache it.
        
        Args:
            dataset_path: Path to CSV file
            nrows: Number of rows to read (None = all rows)
            
        Returns:
            pandas DataFrame or None if failed
        """
        import time
        from pathlib import Path
        
        cache_key = f"{dataset_path}:{nrows}"
        
        # Check cache
        if cache_key in self._dataframe_cache:
            df, timestamp = self._dataframe_cache[cache_key]
            # Check if cache is still valid
            if time.time() - timestamp < self._cache_ttl:
                self._logger.debug(f"Using cached DataFrame for {Path(dataset_path).name} (nrows={nrows})")
                return df
            else:
                # Cache expired
                del self._dataframe_cache[cache_key]
        
        # Load DataFrame
        try:
            import pandas as pd
            path = Path(dataset_path)
            
            if not path.exists():
                self._logger.warning(f"Dataset path does not exist: {dataset_path}")
                return None
            
            if path.suffix.lower() != '.csv':
                self._logger.warning(f"Only CSV files supported for caching, got: {path.suffix}")
                return None
            
            self._logger.info(f"Loading DataFrame from {path.name} (nrows={nrows})...")
            df = pd.read_csv(dataset_path, nrows=nrows)
            
            # Cache it
            self._dataframe_cache[cache_key] = (df, time.time())
            self._logger.info(f"Cached DataFrame: {len(df)} rows, {len(df.columns)} columns")
            
            return df
            
        except Exception as e:
            self._logger.error(f"Failed to load DataFrame: {e}")
            return None
    
    def _clear_expired_cache(self) -> None:
        """Remove expired DataFrame cache entries."""
        import time
        
        expired_keys = []
        now = time.time()
        
        for key, (df, timestamp) in self._dataframe_cache.items():
            if now - timestamp >= self._cache_ttl:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self._dataframe_cache[key]
        
        if expired_keys:
            self._logger.debug(f"Cleared {len(expired_keys)} expired DataFrame cache entries")
    
    async def _build_dataset_inspection(self, state: AnalysisPipelineState) -> Dict[str, Any]:
        """Build structured dataset profile by calling MCP analysis tools.
        
        Returns a clear schema with: columns (list), dtypes (dict), row_count, sample_values.
        This profile is THE authoritative source for planning - no hallucinations allowed.
        """
        try:
            if not state.dataset_path:
                return {"error": "no_dataset_path", "columns": [], "dtypes": {}}
            
            dataset_path_str = str(state.dataset_path)
            
            # Strategy: Use pandas directly for CSV files (fast, reliable, structured)
            # For other formats, try MCP tools as fallback
            from pathlib import Path
            path = Path(dataset_path_str)
            
            # PRIMARY: Direct pandas for CSV (fastest and most reliable)
            if path.exists() and path.suffix.lower() == '.csv':
                # Use cached DataFrame for schema inspection (100 rows is sufficient)
                df = self._get_cached_dataframe(dataset_path_str, nrows=100)
                
                if df is not None:
                    # Compute total row count cheaply without loading full CSV
                    total_rows = None
                    try:
                        with open(path, 'r', encoding='utf-8', errors='ignore') as fh:
                            # Subtract header line if present
                            line_count = sum(1 for _ in fh)
                            total_rows = max(0, line_count - 1) if line_count > 0 else 0
                    except Exception:
                        total_rows = None
                    profile = {
                        "columns": list(df.columns),
                        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                        # Report total rows for user-facing overview; keep sample count separately
                        "row_count": int(total_rows) if isinstance(total_rows, int) else len(df),
                        "row_count_total": int(total_rows) if isinstance(total_rows, int) else len(df),
                        "row_count_sample": len(df),
                        "numeric_columns": df.select_dtypes(include='number').columns.tolist(),
                        "categorical_columns": df.select_dtypes(include='object').columns.tolist(),
                        "datetime_columns": df.select_dtypes(include='datetime').columns.tolist(),
                        "sample_values": {col: df[col].dropna().head(3).tolist() for col in list(df.columns)[:5]},
                        "source": "pandas_direct",
                        "notes": "Planner uses a small snapshot for schema grounding; tools operate on the full dataset."
                    }

                    # --- Semantic classification & relevance scoring (quality-integrated) ---
                    try:
                        instruction_text = (state.instruction or "").lower()
                        intent_tokens = {
                            "correlation": any(t in instruction_text for t in ["correlation","correlat","relationship","associate","relation"]),
                            "distribution": any(t in instruction_text for t in ["histogram","distribution","spread"]),
                            "temporal": any(t in instruction_text for t in ["trend","temporal","time","season","hour","day","weekday"]),
                            "severity": any(t in instruction_text for t in ["severity","injur","fatal","serious"]),
                        }

                        patterns = {
                            "outcome": ["injur","fatal","severity","damage","harm","casualty"],
                            "exposure": ["num_unit","vehicles","volume","duration"],
                            "temporal": ["date","time","hour","weekday","day_of_week","month","season","year"],
                            "environmental": ["weather","lighting","road_surface","visibility","temperature","condition"],
                            "behavioral": ["speed","impair","alcohol","distract","drug","seatbelt"],
                            "infrastructure": ["speed_limit","lane","intersection","road_type","median","control","signal"],
                            "spatial": ["lat","lon","latitude","longitude","segment","location","region","county"],
                            "identifier": ["id","uuid","record","crash_id"],
                            "aggregate": ["total","sum","overall"],
                            "derived": ["rate","ratio","pct","percent","flag"],
                        }

                        def classify(col: str) -> str:
                            low = col.lower()
                            for cat, frags in patterns.items():
                                for f in frags:
                                    if f in low:
                                        return cat
                            return "unknown"

                        semantic = {c: classify(c) for c in profile["columns"]}

                        # Quality metrics: distinct count, non-null count, std for numeric, imbalance for categorical
                        try:
                            MIN_NON_NULL = int(os.getenv("ANALYSIS_MIN_NON_NULL", "30"))
                        except Exception:
                            MIN_NON_NULL = 30
                        try:
                            MIN_DISTINCT = int(os.getenv("ANALYSIS_MIN_DISTINCT", "4"))
                        except Exception:
                            MIN_DISTINCT = 4
                        try:
                            MIN_NUM_STD = float(os.getenv("ANALYSIS_MIN_NUM_STD", "1e-09"))
                        except Exception:
                            MIN_NUM_STD = 1e-9
                        try:
                            IMBALANCE_THRESH = float(os.getenv("ANALYSIS_IMBALANCE_THRESH", "0.9"))
                        except Exception:
                            IMBALANCE_THRESH = 0.9

                        quality = {}
                        for c in profile["columns"]:
                            series = df[c]
                            non_null = series.dropna()
                            q: dict = {
                                "distinct": int(non_null.nunique()),
                                "non_null": int(len(non_null)),
                                "std": None,
                                "imbalance": None,
                                "is_numeric": str(df.dtypes.get(c)).startswith(("int","float","uint","number","decimal")),
                            }
                            # std for numeric
                            try:
                                if q["is_numeric"]:
                                    q["std"] = float(non_null.astype(float).std()) if len(non_null) > 0 else None
                                else:
                                    # category imbalance: max proportion of dominant category in non_null
                                    if len(non_null) > 0:
                                        vc = non_null.astype(str).value_counts(dropna=True)
                                        top_ratio = float(vc.iloc[0] / len(non_null)) if len(vc) else None
                                        q["imbalance"] = top_ratio
                            except Exception:
                                pass
                            quality[c] = q

                        # Deterministic category weights (hierarchy) and quality-integrated relevance scoring
                        CATEGORY_WEIGHTS = {
                            "outcome": 9,
                            "exposure": 8,
                            "environmental": 7,
                            "behavioral": 6,
                            "infrastructure": 5,
                            "temporal": 4,
                            "spatial": 3,
                            "derived": 2,
                            "unknown": 1,
                            "identifier": 0,
                        }
                        relevance = {}
                        for c, cat in semantic.items():
                            base = CATEGORY_WEIGHTS.get(cat, 1)
                            score = base
                            if intent_tokens["correlation"]:
                                # Emphasize outcome + explanatory pairs indirectly via higher base for outcome/explanatory cats
                                if cat == "outcome":
                                    score += 3
                                elif cat in {"environmental","behavioral","infrastructure","temporal","exposure"}:
                                    score += 2
                                elif cat == "spatial":
                                    score += 1
                            if intent_tokens["distribution"]:
                                if cat in {"outcome","exposure"}:
                                    score += 2
                                elif cat == "temporal":
                                    score += 1
                            if intent_tokens["temporal"] and cat == "temporal":
                                score += 2
                            if intent_tokens["severity"] and cat == "outcome":
                                score += 2

                            # Hard exclusions and quality gates
                            q = quality.get(c, {})
                            if cat == "identifier":
                                score = 0
                            # Numeric gates
                            if bool(q.get("is_numeric")):
                                if q.get("non_null", 0) < MIN_NON_NULL or q.get("distinct", 0) < MIN_DISTINCT:
                                    score = 0
                                stdv = q.get("std")
                                if stdv is not None and stdv < MIN_NUM_STD:
                                    score = 0
                            else:
                                # Categorical imbalance penalty (but not zero-out; usable for bars/facets)
                                imb = q.get("imbalance")
                                if imb is not None and imb >= IMBALANCE_THRESH:
                                    score = max(0, score - 2)

                            # Clamp and convert to coarse 0–5 scale for planner familiarity
                            score = max(0, min(15, int(score)))
                            coarse = 0
                            if score >= 12:
                                coarse = 5
                            elif score >= 9:
                                coarse = 4
                            elif score >= 6:
                                coarse = 3
                            elif score >= 3:
                                coarse = 2
                            elif score > 0:
                                coarse = 1
                            relevance[c] = coarse

                        profile["semantic_categories"] = semantic
                        profile["column_quality"] = quality
                        profile["relevance_scores"] = relevance
                    except Exception as _sem_err:
                        profile["semantic_error"] = str(_sem_err)
                    
                    self._logger.info(
                        f"Direct pandas inspection: {len(profile['columns'])} columns, total_rows={profile['row_count_total']}, sample_rows={profile['row_count_sample']}"
                    )
                    return profile
                else:
                    self._logger.warning("Failed to load DataFrame, trying MCP tools...")
            
            # FALLBACK: Try MCP tools for non-CSV or if pandas failed
            if "load_and_analyze_csv" in self._tool_map:
                try:
                    tool = self._tool_map["load_and_analyze_csv"]
                    result = await tool.ainvoke({"file_path": dataset_path_str})
                    output = result if isinstance(result, str) else str(result)
                    
                    # Parse the output to extract column names
                    # The tool returns text like "Columns: Col1, Col2, Col3"
                    columns = []
                    import re
                    
                    # Try to extract column list from output
                    col_match = re.search(r'Columns?:\s*\[([^\]]+)\]', output, re.IGNORECASE)
                    if col_match:
                        cols_str = col_match.group(1)
                        columns = [c.strip().strip("'\"") for c in cols_str.split(',')]
                    else:
                        # Alternative format: "Columns: col1, col2, col3"
                        col_match = re.search(r'Columns?:\s*(.+?)(?:\n|$)', output, re.IGNORECASE)
                        if col_match:
                            cols_str = col_match.group(1)
                            # Only accept if we got actual column names, not just a number
                            potential_cols = [c.strip().strip("'\"") for c in cols_str.split(',') if c.strip()]
                            # Check if it's just a number (e.g., "24") - reject that
                            if len(potential_cols) == 1 and potential_cols[0].isdigit():
                                self._logger.warning(f"load_and_analyze_csv returned column count ({potential_cols[0]}) not column names")
                            else:
                                columns = potential_cols
                    
                    if columns and len(columns) > 1:  # Must have at least 2 columns to be valid
                        self._logger.info(f"Extracted {len(columns)} columns from dataset: {columns[:5]}...")
                        return {
                            "columns": columns,
                            "dtypes": {},  # Could parse if tool provides
                            "row_count": None,  # Could parse if tool provides
                            "source": "load_and_analyze_csv",
                            "raw_output_preview": output[:500]
                        }
                except Exception as e:
                    self._logger.warning(f"load_and_analyze_csv inspection failed: {e}")
            
            # Try analysis_dataset_inspector
            if "analysis_dataset_inspector" in self._tool_map:
                try:
                    tool = self._tool_map["analysis_dataset_inspector"]
                    result = await tool.ainvoke({"file_path": dataset_path_str})
                    
                    # Try to parse structured response
                    if hasattr(result, 'content'):
                        import json
                        content = result.content[0].text if hasattr(result.content[0], 'text') else str(result.content)
                        parsed = json.loads(content)
                        if isinstance(parsed, dict) and 'columns' in parsed:
                            return parsed
                    
                    # Parse text output
                    import re
                    output = str(result)
                    columns = []
                    col_match = re.search(r'Columns?:\s*\[([^\]]+)\]', output, re.IGNORECASE)
                    if col_match:
                        cols_str = col_match.group(1)
                        columns = [c.strip().strip("'\"") for c in cols_str.split(',')]
                        return {"columns": columns, "dtypes": {}, "source": "analysis_dataset_inspector"}
                        
                except Exception as e:
                    self._logger.warning(f"analysis_dataset_inspector failed: {e}")
            
            # Ultimate fallback: basic file info (no columns)
            return {
                "error": "inspection_failed",
                "file_type": path.suffix.upper().lstrip('.') or 'Unknown',
                "file_exists": path.exists(),
                "columns": [],  # Empty but valid
                "dtypes": {},
                "source": "fallback"
            }
            
        except Exception as e:
            self._logger.error(f"Dataset inspection critical error: {e}")
            return {"error": str(e), "columns": [], "dtypes": {}}

    async def _build_workflow_intelligence(self, state: AnalysisPipelineState) -> Dict[str, Any]:
        """Build workflow intelligence context like the successful React agent."""
        return {
            "analysis_intent": f"Comprehensive analysis of {Path(state.dataset_path).name if state.dataset_path else 'provided data'}",
            "analytical_approach": "React-driven autonomous tool selection with intelligent workflow adaptation",
            "planning_reasoning": "Let LLM autonomously discover data patterns and select appropriate analytical tools",
            "success_criteria": "Generate actionable insights with supporting visualizations and statistical evidence",
            "execution_strategy": "Autonomous tool selection guided by data discovery and analytical reasoning"
        }

    def _build_react_analysis_prompt(self, state: AnalysisPipelineState, dataset_inspection: Dict, workflow_intelligence: Dict) -> str:
        """Build comprehensive React analysis prompt like the successful original agent."""
        
        # Analysis Output Template for structured reporting
        analysis_output_template = """
Produce an analytically intelligent report in three sections: Descriptive Statistics, Analysis Results, and Executive Summary.

## Descriptive Statistics
- Dataset size (rows, columns)
- Column names and inferred types
- Missing values and duplicate counts
- Key numeric summaries (mean, median, min, max) for relevant numeric columns

## Analysis Results
For EACH visualization produced, include:
### <Plot Title>
![Plot Title](<filename.png>)
**Overview:** one sentence describing the plot.
**Key Findings:** concise bullets with exact numeric evidence.
**Actionable Insights:** bullets linking findings to recommended actions.

## Text-Extracted Insights
- Consolidate text-analysis findings with representative examples and counts.

## Executive Summary
- Most important findings and top recommended actions with impact estimates.

CRITICAL: Use exact numbers from analyses. Keep language precise and factual.
"""

        return f"""
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.
You are also a highly analytically intelligent and precise expert data analyst conducting a comprehensive analysis.
You have access to sophisticated analytical tools and can use them autonomously to discover insights.

DATASET CONTEXT:
Dataset File Path: {state.dataset_path or 'Not specified'}
Dataset Overview: {dataset_inspection}

RESEARCH QUESTION:
{state.instruction or 'Perform comprehensive data analysis'}

WORKFLOW PLANNING INTELLIGENCE:
ANALYSIS INTENT: {workflow_intelligence['analysis_intent']}
ANALYTICAL APPROACH: {workflow_intelligence['analytical_approach']}
PLANNING REASONING: {workflow_intelligence['planning_reasoning']}
SUCCESS CRITERIA: {workflow_intelligence['success_criteria']}

EXECUTION GUIDELINES:
- Use available tools autonomously to explore and analyze the data
- Start with dataset loading and basic exploration
- Discover patterns through statistical analysis and visualization
- Extract insights from text fields if present
- Generate comprehensive visualizations
- Synthesize findings into actionable insights

VISUALIZATION GUIDELINES:
- Use Latitude/Longitude ONLY for spatial scatterplots/maps
- Exclude coordinates from correlation analyses and statistical summaries
- Always ensure one axis is a quantitative metric (count, mean, rate)
- Color or group plots by relevant categories to reveal patterns

TEXT ANALYTICS INTEGRATION:
- When using text analytics tools, integrate extracted insights immediately
- Cross-reference textual findings with numerical results
- Treat text-derived insights as first-class analytical data

{analysis_output_template}

Begin your analysis by loading and exploring the dataset, then proceed autonomously based on what you discover.
"""

    def _extract_tool_calls_from_messages(self, messages: List) -> List[Dict]:
        """Extract tool calls from React agent message history."""
        tool_calls = []
        for msg in messages:
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    tool_calls.append({
                        "tool": tool_call.get("name", "unknown"),
                        "arguments": tool_call.get("args", {}),
                        "id": tool_call.get("id", "")
                    })
        return tool_calls

    def _extract_tool_results_from_messages(self, messages: List) -> List[Dict]:
        """Extract tool results/outputs from React agent message history."""
        tool_results = []
        for i, msg in enumerate(messages):
            # DEBUG: Log message details for debugging
            msg_type = getattr(msg, 'type', type(msg).__name__)
            
            # Check for ToolMessage (LangChain's tool result message type)
            if hasattr(msg, 'type') and msg.type == 'tool':
                tool_results.append({
                    "tool": getattr(msg, 'name', 'unknown'),
                    "content": getattr(msg, 'content', ''),
                    "tool_call_id": getattr(msg, 'tool_call_id', ''),
                    "timestamp": datetime.now().isoformat()
                })
            # Check for any message that contains 'ToolMessage' in its class name
            elif 'ToolMessage' in type(msg).__name__:
                tool_results.append({
                    "tool": getattr(msg, 'name', getattr(msg, 'tool_call_id', 'unknown')),
                    "content": getattr(msg, 'content', ''),
                    "tool_call_id": getattr(msg, 'tool_call_id', ''),
                    "timestamp": datetime.now().isoformat()
                })
            # Also check for AIMessage that might contain tool results
            elif hasattr(msg, 'content') and isinstance(msg.content, str):
                content = msg.content.lower()
                # Look for patterns that indicate tool execution results
                if any(pattern in content for pattern in [
                    'tool executed', 'analysis complete', 'plot saved', 'visualization',
                    'created chart', 'generated plot', 'saved to', 'correlation matrix',
                    'distribution analysis', 'temporal analysis'
                ]):
                    tool_results.append({
                        "tool": "inferred_from_content",
                        "content": msg.content[:1000],  # Capture more content
                        "timestamp": datetime.now().isoformat()
                    })
        return tool_results

    async def receive_a2a_message(self, request) -> dict:
        """Accept SendMessageRequest and invoke the compiled LangGraph once."""
        if not self.agent:
            await self.setup()

        try:
            message = request.params.message

            dataset_path: Optional[str] = None
            instructions = "Perform comprehensive data analysis"

            for data_obj in self._iter_a2a_data_parts(message.parts):
                if not isinstance(data_obj, dict):
                    continue
                dataset_path = dataset_path or (
                    data_obj.get("dataset_path")
                    or data_obj.get("path")
                    or data_obj.get("dataset")
                )
                # Prefer top-level instruction keys if present
                instructions = (
                    data_obj.get("analysis_instruction")
                    or data_obj.get("instructions")
                    or data_obj.get("task")
                    or instructions
                )
                # Also support nested analysis_request payloads (as used by a2a_client CSV mode)
                if isinstance(data_obj.get("analysis_request"), dict):
                    ar = data_obj["analysis_request"]
                    dataset_path = dataset_path or (
                        ar.get("dataset_path")
                        or ar.get("path")
                        or ar.get("dataset")
                    )
                    instructions = (
                        ar.get("analysis_instruction")
                        or ar.get("instructions")
                        or ar.get("task")
                        or instructions
                    )

            # DEBUG_PRINT: show parsed dataset and instruction length
            self._dprint(f"A2A parsed dataset_path={dataset_path}, instruction_len={len(instructions or '')}")

            seed_text = f"Dataset: {dataset_path or 'unspecified'}\nInstructions: {instructions}"
            initial_state = AnalysisPipelineState(
                messages=[HumanMessage(content=seed_text)],
                dataset_path=dataset_path,
                instruction=instructions,
            )

            # Idempotency guard -------------------------------------------------
            import hashlib, time
            basis = (str(dataset_path or 'no_dataset') + '|' + str(instructions or '')).lower().strip()
            signature = hashlib.sha256(basis.encode()).hexdigest()[:32]
            try:
                ttl = int(os.getenv('ANALYSIS_IDEMPOTENCY_TTL', '120'))
            except Exception:
                ttl = 120
            now = time.time()
            cached = self._run_cache.get(signature)
            if cached and (now - cached.get('ts', 0) < ttl):
                self._dprint(f"Idempotency cache hit (sig={signature}, age={now-cached['ts']:.1f}s)")
                resp = cached['response']
                try:
                    resp['result']['artifacts']['idempotent_cached'] = True
                except Exception:
                    pass
                return resp

            # Allow higher recursion limit to accommodate execution loops
            try:
                rec_limit = int(os.getenv("ANALYSIS_RECURSION_LIMIT", "50"))
            except Exception:
                rec_limit = 50
            # Set a run id for tracing this invocation across nodes
            try:
                import uuid
                self._current_run_id = uuid.uuid4().hex[:8]
                self._logger.info(f"run={self._current_run_id} START dataset={dataset_path} instr_len={len(instructions or '')}")
                # Clear old cache entries at the start of each run
                self._clear_expired_cache()
            except Exception:
                self._current_run_id = None
            cfg = {"recursion_limit": rec_limit}
            try:
                ctx_id = getattr(getattr(getattr(request, "params", None), "message", None), "context_id", None)
                if isinstance(ctx_id, str) and ctx_id.strip():
                    cfg["configurable"] = {"thread_id": ctx_id}
            except Exception:
                pass
            output = await self.agent.ainvoke(initial_state, config=cfg)

            # Accept both dataclass instance and plain dict outputs
            def _out_get(attr: str, default=None):
                try:
                    if hasattr(output, attr):
                        return getattr(output, attr)
                    if isinstance(output, dict):
                        return output.get(attr, default)
                except Exception:
                    pass
                return default

            report_text = _out_get("report_text") or _out_get("summary") or "Analysis completed."
            artifacts = _out_get("artifacts", {}) or {}
            # Attach telemetry to artifacts if present
            try:
                telemetry = getattr(output, 'telemetry', None)
                if isinstance(telemetry, dict):
                    artifacts['telemetry'] = dict(telemetry)
            except Exception:
                pass

            response = {
                "result": {
                    "status": {"state": "completed"},
                    "analysis_result": report_text,
                    "artifacts": artifacts,
                }
            }
            # Cache for idempotency
            try:
                self._run_cache[signature] = {"ts": now, "response": response}
            except Exception:
                pass
            try:
                self._logger.info(f"run={self._current_run_id or '-'} END status={response['result']['status']['state']}")
            except Exception:
                pass
            return response
        except Exception as exc:  # noqa: BLE001
            self._logger.error(f"LangGraph pipeline error: {exc}")
            return {
                "error": {
                    "message": f"Pipeline error: {exc}",
                    "type": type(exc).__name__,
                }
            }

    async def astream_a2a_message(
        self,
        request,
        *,
        context_id: str,
        stream_mode: Optional[Iterable[str]] = None,
        recursion_limit: Optional[int] = None,
    ) -> AsyncIterator[Tuple[str, Any]]:
        """Stream the compiled LangGraph execution for an A2A request.

        This is a transport-layer entrypoint intended for the A2A service to
        forward progress to clients. It uses the same request parsing and
        initial-state seeding as receive_a2a_message(), but yields streaming
        chunks from `self.agent.astream(...)`.

        Notes:
        - This does NOT emit A2A events; the service layer owns that.
        - This does NOT store streaming output in LangGraph state.
        """
        if not self.agent:
            await self.setup()

        message = request.params.message

        dataset_path: Optional[str] = None
        instructions = "Perform comprehensive data analysis"
        resume_payload: Optional[Dict[str, Any]] = None

        for data_obj in self._iter_a2a_data_parts(message.parts):
            if not isinstance(data_obj, dict):
                continue

            # HITL resume payload (plan review / future interrupts)
            try:
                if isinstance(data_obj.get("hitl_resume"), dict):
                    resume_payload = data_obj.get("hitl_resume")
                elif isinstance(data_obj.get("hitl_decision"), str):
                    resume_payload = {
                        "decision": str(data_obj.get("hitl_decision") or "").strip(),
                        "feedback": data_obj.get("feedback") or data_obj.get("notes"),
                    }
                elif isinstance(data_obj.get("decision"), str) and str(data_obj.get("decision")).strip().lower() in ("approve", "revise", "abort"):
                    resume_payload = {
                        "decision": str(data_obj.get("decision") or "").strip(),
                        "feedback": data_obj.get("feedback") or data_obj.get("notes"),
                    }
            except Exception:
                pass

            dataset_path = dataset_path or (
                data_obj.get("dataset_path")
                or data_obj.get("path")
                or data_obj.get("dataset")
            )
            instructions = (
                data_obj.get("analysis_instruction")
                or data_obj.get("instructions")
                or data_obj.get("task")
                or instructions
            )
            if isinstance(data_obj.get("analysis_request"), dict):
                ar = data_obj["analysis_request"]
                dataset_path = dataset_path or (
                    ar.get("dataset_path")
                    or ar.get("path")
                    or ar.get("dataset")
                )
                instructions = (
                    ar.get("analysis_instruction")
                    or ar.get("instructions")
                    or ar.get("task")
                    or instructions
                )

        graph_input: Any
        if resume_payload is not None:
            graph_input = Command(resume=resume_payload)
        else:
            seed_text = f"Dataset: {dataset_path or 'unspecified'}\nInstructions: {instructions}"
            graph_input = AnalysisPipelineState(
                messages=[HumanMessage(content=seed_text)],
                dataset_path=dataset_path,
                instruction=instructions,
            )

        try:
            rec_limit = int(os.getenv("ANALYSIS_RECURSION_LIMIT", "50"))
        except Exception:
            rec_limit = 50
        if recursion_limit is not None:
            rec_limit = recursion_limit

        modes = list(stream_mode) if stream_mode is not None else ["messages", "updates"]
        config = {
            "recursion_limit": rec_limit,
            "configurable": {"thread_id": context_id},
        }

        async for mode, chunk in self.agent.astream(
            graph_input,
            config=config,
            stream_mode=modes,
        ):
            yield mode, chunk


    @staticmethod
    def _extract_json(raw_text: Any) -> Dict[str, Any]:
        """Extract the most relevant JSON object from arbitrary LLM output.

        Improvements over the simple extractor:
        - Strips code fences.
        - Collects ALL top-level JSON objects in the text (handles concatenated objects).
        - Prefers objects containing keys like 'insights' or 'dataset_overview'.
        - Falls back to the first successfully parsed object; returns {} on failure.
        """
        if isinstance(raw_text, dict):
            return raw_text
        if not isinstance(raw_text, str):
            return {}

        text = raw_text.strip()
        # Strip markdown code fences
        if text.startswith("```") and text.endswith("```"):
            inner = text[3:-3].strip()
            # Remove leading language identifier like ```json
            first_line, _, rest = inner.partition("\n")
            text = (rest or inner).strip() if first_line.lower().startswith("json") else inner

        # Try direct parse fast-path
        if text.startswith("{") and text.endswith("}"):
            try:
                obj = json.loads(text)
                return obj if isinstance(obj, dict) else {}
            except Exception:
                pass

        # Stream-scan and collect all top-level objects
        depth = 0
        start_idx = None
        in_string = False
        escape = False
        candidates: List[Dict[str, Any]] = []
        for i, ch in enumerate(text):
            if ch == '"' and not escape:
                in_string = not in_string
            if ch == '\\' and not escape:
                escape = True
            else:
                escape = False
            if in_string:
                continue
            if ch == '{':
                if depth == 0:
                    start_idx = i
                depth += 1
            elif ch == '}':
                if depth > 0:
                    depth -= 1
                    if depth == 0 and start_idx is not None:
                        candidate = text[start_idx:i+1]
                        try:
                            obj = json.loads(candidate)
                            if isinstance(obj, dict):
                                candidates.append(obj)
                        except Exception:
                            pass
                        finally:
                            start_idx = None
        # Heuristic selection
        if not candidates:
            return {}
        # Prefer the one with structured report fields
        def score(d: Dict[str, Any]) -> int:
            keys = set(k.lower() for k in d.keys())
            s = 0
            if 'insights' in keys:
                s += 3
            if 'dataset_overview' in keys:
                s += 2
            if 'title' in keys:
                s += 1
            if 'recommendations' in keys:
                s += 1
            return s
        best = max(candidates, key=score)
        return best if isinstance(best, dict) else candidates[0]


async def create_analysis_pipeline_agent() -> ModularAnalysisPipelineAgent:
    """Factory for the LangGraph-backed analysis agent."""

    try:
        step_cap = int(os.getenv("ANALYSIS_PIPELINE_STEP_CAP", "6"))
    except Exception:  # noqa: BLE001
        step_cap = 6

    agent = ModularAnalysisPipelineAgent(step_cap=step_cap)
    await agent.setup()
    return agent
