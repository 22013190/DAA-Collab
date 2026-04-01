I’ll quickly scan the codebase for where key state fields are written so the “state contract” table reflects actual ownership, not assumptions.

Searched for regex `\bcontext_summary\b\s*=` (`**/agent_system/src/agents/analysis/pipeline/**/*.py`), 2 results

Searched for regex `update_context_summary\(|should_synthesize` (`**/agent_system/src/agents/analysis/pipeline/**/*.py`), 2 results

Searched for regex `\bshould_synthesize\b` (`**/agent_system/src/agents/analysis/pipeline/runner.py`), no results

Searched for regex `\berrors\b\s*[:=]` (`**/agent_system/src/agents/analysis/pipeline/runner.py`), 5 results

State model reference: state.py. Node implementations: runner.py.

### AnalysisPipelineState “state contract” (field → owning node → when updated)

| Field | Owning node(s) (primary writer) | When updated | Invariant / notes |
|---|---|---|---|
| `messages` | **Caller/A2A entrypoint**; `execute_step`, `planner` (HITL) | Caller seeds initial messages; `execute_step` may add a `HumanMessage` on security block; `planner` may add HITL feedback message | Uses LangGraph `add_messages` reducer (append/merge). Not single-writer. |
| `dataset_path` | **Caller/A2A entrypoint** | Set at run start from request | Treated as input; not mutated by nodes in the runner. |
| `instruction` | **Caller/A2A entrypoint** | Set at run start from request | Treated as input; used by planner/interpret/synthesis. |
| `historical_snippet` | `ingest_request` | If prior run history exists for `dataset_path` | Loaded from `MemoryStore`; used to ground planning. |
| `preprocess_profile` | `ingest_request` (load), `planner` (build+persist) | Loaded from persisted profile if available; otherwise built during planning | Becomes the authoritative column list for arg correction + reflect’s “use only real columns” refinement. |
| `column_mappings` | `planner` | First planning pass (LLM column classification), then reused | Used by planner + execute_step for column auto-correction. |
| `output_mode` | `planner` | After LLM plan JSON parse | `"visualization_only" | "insights_only" | "mixed"`; drives synthesis formatting choices. |
| `visualization_only` | `planner` | Derived from `output_mode` | Convenience boolean; synthesis also uses it. |
| `classification_confidence` | `planner` | After plan parse | Optional float (0–1). |
| `classification_rationale` | `planner` | After plan parse | Short string; optional. |
| `plan_steps` | **`planner`** | Every (re)plan | Single-writer by design: only planner should replace the plan. |
| `plan_hash` | `planner` | Every (re)plan | Used to detect “replan produced same plan”. |
| `next_step_index` | **`execute_step`** (increment), `planner` (reset/preserve) | `execute_step` increments after each attempted step; `planner` resets to 0 on plan change or preserves on no-change | Key invariant: “progress pointer”. Execution logic assumes it is monotonic within a plan. |
| `replan_no_change` | `planner` | When new plan hash equals previous | Reflection uses this to cap “replan without progress” loops. |
| `plan_retries` | `reflect` | When `replan_no_change=True` and reflect asks for another replan | Loop guard (caps repeated identical replans). |
| `finalize` | `planner` (empty/exhausted plan), `reflect` (stop loops), `execute_step` (reads) | Set when the system wants to force `synthesis` regardless of remaining logic | Invariant: when `finalize=True`, `execute_step` routes directly to `synthesis`. |
| `refinement_request` | **`reflect`** | When coverage says something actionable is missing | This is the “replan prompt payload” (short, deduped list). |
| `refinement_round` | **`reflect`** | Each time reflect triggers replanning | Capped by `ANALYSIS_MAX_REFINEMENTS`; used to stop infinite loops. |
| `reinterpret_attempts` | `reflect` | When reflect thinks existing outputs suffice but interpretation under-extracted | One-shot reroute back to `interpret_results` (recovery without new tool calls). |
| `coverage` | **`interpret_results`** (primary), `reflect` (override) | `interpret_results` sets `answers_query` + `missing_info`; `reflect` may override to answered after reasoning | Invariant enforced in interpret: if `answers_query=True`, it clears `missing_info`. |
| `insights` | **`interpret_results`** | After tools finish (post-plan execution) | Merge/dedup reducer; synthesis may *render* state insights “as authoritative” but does not write them. |
| `tool_transcript` | **`execute_step`** | After every tool attempt (ok/error/skip), plus security-block events | Append-only semantics via merge reducer (dedup + capped). |
| `artifact_log` | `execute_step` (primary), `synthesis` (late discovery) | After tool runs (from outputs or filesystem detection); synthesis may register artifacts if missing (viz-only path) | Merge + capped; used for embedding plots and traceability. |
| `warnings` | `ingest_request`, `execute_step` | Ingest adds injection warnings; execute_step adds guardrail warnings/policy blocks | Merge + capped list. |
| `errors` | (currently not actively written by runner nodes) | — | Defined in state model, but runner mainly represents failures via `tool_transcript.status="error"` and output envelope `errors`. |
| `telemetry` | `ingest_request`, `planner`, `execute_step`, `interpret_results`, `reflect` | Each node emits small counters (merged by summing) | Not single-writer; used for observability. |
| `token_metrics` | `interpret_results` (via Command), `synthesis` (in-place via `state.record_token_usage`) | Interpret increments `interpret`; synthesis may increment `synthesis` | Merge-by-sum dict reducer. |
| `capability_gap` | `reflect` | When `missing_info` is non-actionable (tool gap) or forced termination wants diagnosis | Explains “can’t answer with current toolset” without looping. |
| `hitl_plan_reviewed` | `planner` (HITL interrupt path) | First plan (or each replan if configured) | HITL gate state. |
| `hitl_plan_decision` | `planner` (HITL) | On HITL decision: approve/revise/abort | Used mainly for metadata/audit. |
| `hitl_plan_feedback` | `planner` (HITL) | On HITL revise/feedback | May also inject `messages` for planner. |
| `stop_after_first_synthesis` | **Caller/config** (not set by nodes) | If enabled in initial state/config | Used to latch single-run behavior after synthesis. |
| `context_summary` | (present, but not actively maintained in runner) | — | Read by `_analyze_tool_response` but not updated in the shown runner code (stays default unless caller sets it). |
| `should_synthesize` | (present, but unused by runner routing) | — | Routing is controlled by `finalize`, step exhaustion, and `coverage`. |
| `report_text` / `summary` / `final_artifacts` | `synthesis` (best-effort in-place via `state.set_report`) | During viz-minimal synthesis path (and sometimes in full path) | The *authoritative* final output is the returned `PipelineOutputState`; state fields are secondary/compat. |

### PipelineOutputState (envelope) ownership (quick)
| Field | Owning node(s) | When updated |
|---|---|---|
| `status`, `report_text`, `summary`, `artifacts`, `steps`, `tool_summaries`, `warnings`, `errors`, `token_metrics` | **`synthesis`** | Built once per run (with caching/guardrails) |
| `artifacts["report_path"]`, `artifacts["__report_saved"]` | `synthesis` (primary), `persist_cleanup` (fallback) | `persist_cleanup` writes only if synthesis didn’t or idempotency allows |