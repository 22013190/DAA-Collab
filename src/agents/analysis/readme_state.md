**Where the “state” lives**
- The graph state is `AnalysisPipelineState` in state.py. Nodes usually update it by returning `Command(update=..., goto=...)` (LangGraph merges updates using the reducers defined on each field).
- The terminal output is `PipelineOutputState` (also in state.py). The `synthesis` node returns this (not a `Command`), then `persist_cleanup` receives it.

**Overall flow (routing)**
- `START → ingest_request → planner → execute_step → …(repeat execute_step per step)… → interpret_results → reflect → (planner | execute_step | synthesis) → persist_cleanup → END`
- Key routing signals in state:
  - `plan_steps` + `next_step_index`: drives the `execute_step` loop
  - `coverage.answers_query` + `coverage.missing_info`: drives `reflect` decisions
  - `finalize=True`: forces early exit to `synthesis`
  - `refinement_request` + `refinement_round`: drives replanning prompts + caps loops

---

## Node-by-node: state reads, writes, and why

### 1) `ingest_request(state) -> Command(goto="planner")`
**Purpose:** run-start hygiene + optional memory/profile preload.

**Reads (inputs)**
- `state.instruction`, `state.dataset_path`, `state.messages`

**Writes (state changes)**
- In-place mutation:
  - `state.trim_message_window()` (keeps last N messages; default window is small)
- `Command(update=...)` may set:
  - `historical_snippet`: a short recap from prior runs (from `MemoryStore`)
  - `preprocess_profile`: persisted dataset profile (columns/types/samples) if found
  - `warnings`: adds prompt-injection risk warning (detection-only)
  - `telemetry`: increments injection flag counter (merged dict)

**Routing**
- Always `goto="planner"` (it either preloads extra context or not, but planning is next).

---

### 2) `planner(state) -> Command(goto="execute_step")` (or `END` in guard case)
**Purpose:** produce a structured plan (`plan_steps`) grounded in dataset + available tools.

**Reads (inputs)**
- `preprocess_profile` (reuse if present), else uses `dataset_path` to build it
- `tool_transcript` (for de-duping already-successful tool calls)
- `refinement_request` / `refinement_round` (reflection feedback to improve the plan)
- `dataset_path`, `instruction`

**Writes (state changes)**
- Builds/updates dataset grounding:
  - `preprocess_profile`: dataset snapshot/profile (columns, dtypes, sample_values, etc.)
  - `column_mappings`: LLM-inferred “semantic → column name” hints (e.g., `datetime_column → ["crash_date", ...]`)
- Produces plan:
  - `plan_steps`: list of `{tool, args, why}`
  - `plan_hash`: SHA-256 hash of `plan_steps` for “replan but no change” detection
  - `next_step_index`:
    - reset to `0` if plan changed
    - preserved if plan unchanged (and can trigger `finalize` if already exhausted)
  - `replan_no_change`: true when same `plan_hash`
  - `finalize`: can be set true if no valid steps were produced
- Output-mode classification (downstream formatting behavior):
  - `output_mode`: `"visualization_only" | "insights_only" | "mixed"`
  - `visualization_only`: boolean convenience flag
  - `classification_confidence`, `classification_rationale`
- Telemetry:
  - `telemetry={"planner_calls": 1}` (merged/summed)

**Important built-in “argument firewall” behaviors in planning**
- Filters tool args to the tool’s schema (drops unknown keys)
- Auto-injects `file_path` when required and `dataset_path` exists
- Proactively corrects invalid column names using `preprocess_profile["columns"]` + `column_mappings`
- Caps/filters repetitive tools (e.g., `assess_data_quality` max once by default)

**Routing**
- Normal: `goto="execute_step"`
- If single-run guard is latched: `goto=END`
- If no valid steps: sets `finalize=True` and still routes to `execute_step` (which will bounce to `synthesis`).

---

### 3) `execute_step(state) -> Command(goto="execute_step" | "interpret_results" | "planner" | "synthesis")`
**Purpose:** execute exactly one planned tool call, append results, advance `next_step_index`.

**Reads (inputs)**
- `plan_steps`, `next_step_index`, `finalize`
- `preprocess_profile` + `column_mappings` (for last-second arg correction)
- tool registry: `self._tool_map`
- security policy config (allowed roots, forced output dir)

**Writes (state changes)**
- On each step it appends a tool event:
  - `tool_transcript: [event]` where `event` includes:
    - `tool`, `args`, output/`result` (may be summarized), `status` (`ok|error|skip`), `artifact`, timestamps, etc.
  - `telemetry={"tool_calls": 1, "tool_errors": 0|1}`
- Advances:
  - `next_step_index = old + 1` (with a defensive monotonicity check)
- Artifacts:
  - `artifact_log: [artifact paths]` (merged + capped)
  - Also calls `state.register_artifact(...)` in-place for immediate availability
- Warnings:
  - `warnings` may include guardrail sanitization warnings or policy block messages
- If blocked by policy (`GuardrailViolation`):
  - Adds an `error` tool event, increments `tool_errors`
  - Injects a new `messages=[HumanMessage(...)]` telling the planner to replan safely
  - Routes back to `planner`

**Routing**
- If `finalize=True`: `goto="synthesis"`
- If no plan or plan exhausted: `goto="interpret_results"` (or `synthesis` for truly empty)
- If step executed and more remain: `goto="execute_step"`
- If last step executed: `goto="interpret_results"`
- If global step cap exceeded: `goto="synthesis"`

---

### 4) `interpret_results(state) -> Command(goto="reflect")`
**Purpose:** convert recent *successful* tool outputs into structured `insights` + `coverage`.

**Reads (inputs)**
- `tool_transcript` (by default uses last 5 events; can be “use all” via env)
- `instruction`, `context_summary`, `preprocess_profile` (for column-aware heuristics)
- `finalize`, plus step completeness (`next_step_index` vs `len(plan_steps)`)

**Writes (state changes)**
- `insights`: list of objects like:
  - `statement`, `evidence_snippet`, `evidence_ref` (points to transcript index/tool), `evidence_source`
  - This list is merged/deduped by `(statement, evidence)` via reducer
- `coverage`: normalized to:
  - `{"answers_query": bool, "missing_info": [0..3 actionable items]}`
  - Invariant: if `answers_query=True`, it clears `missing_info`
- `token_metrics={"interpret": approx_tokens}` (merged/summed)
- `telemetry={"interpret_calls": 1}` (merged/summed)

**Key rule:** it filters out tool events with `status == "error"` before asking the LLM, so insights are grounded in successful outputs.

**Routing**
- Always `goto="reflect"` after producing `insights` + `coverage`.
- If steps aren’t finished yet, it routes back to `execute_step` (to enforce “interpret after execution”).

---

### 5) `reflect(state) -> Command(goto="execute_step" | "planner" | "interpret_results" | "synthesis")`
**Purpose:** decide whether to continue executing, replan, reinterpret, or synthesize.

**Reads (inputs)**
- `coverage.answers_query`, `coverage.missing_info`
- `plan_steps`, `next_step_index`
- `tool_transcript` (checks for `ok` vs `error` patterns)
- `preprocess_profile["columns"]` (to build column-aware refinement messages)
- Loop controls: `refinement_round`, `plan_hash`, `replan_no_change`, `plan_retries`, `reinterpret_attempts`, `finalize`

**Writes (state changes)**
Common updates it may emit:
- Replanning path:
  - `refinement_request`: cleaned actionable items (deduped + capped)
  - `refinement_round += 1`
  - `telemetry={"refinement_rounds": 1}`
  - `plan_retries` incremented if it detects “replan but no change” loops
- Force-stop path:
  - `finalize=True` (to stop loops and go to `synthesis`)
  - `capability_gap`: optional structured diagnosis when missing info is not actionable with current tools
- Re-interpretation recovery:
  - `reinterpret_attempts += 1` and `goto="interpret_results"` once, if it believes existing outputs suffice but interpretation under-extracted

**Routing (decision logic)**
- If reflection is disabled (env flags): either
  - continue `execute_step` if steps remain, or
  - `synthesis` if not
  - plus a small “errors but no ok outputs” recovery that triggers `planner`
- If `finalize=True`: `synthesis`
- If steps remain: `execute_step` (prevents premature synthesis)
- If answered or no missing actionable items: `synthesis`
- If missing exists:
  - may attempt `_reason_about_existing_data`; if it can answer, it marks coverage answered and goes `synthesis` (or triggers a single `interpret_results` retry)
  - otherwise: attaches `refinement_request` and goes `planner`
- Caps:
  - if max refinements reached: `finalize=True` then `synthesis`

---

### 6) `synthesis(state) -> PipelineOutputState`
**Purpose:** produce the final Markdown report + structured envelope; also persists outputs (report files + metadata) with idempotency guards.

**Reads (inputs)**
- Everything needed for final output: `tool_transcript`, `artifact_log`, `insights`, `coverage`, `plan_steps`, `next_step_index`, `instruction`, `output_mode`, `visualization_only`, `token_metrics`, `refinement_round`, `capability_gap`, etc.

**Writes (state changes)**
- This node mainly returns `PipelineOutputState` (not a `Command`), but it also performs side effects:
  - Writes the Markdown report to `reports_path`
  - May write sidecars (structured JSON, tool transcript markdown, state insights markdown) depending on env flags
  - Schedules memory persistence (`MemoryStore.schedule_append`)
- In-place mutations to `state` happen in a few places (best-effort):
  - `state.register_artifact(...)` during late artifact discovery
  - `state.set_report(report_text, summary, {...})` (sets `report_text`, `summary`, `final_artifacts`)
- Internal single-run caching:
  - latches `self._single_run_completed=True`
  - stores `self._cached_final_output=output`

**Envelope output (`PipelineOutputState`) includes**
- `status` (`ok|error`) derived from tool errors / refinement incomplete
- `report_text`, `summary`
- `artifacts` dict with `artifact_log`, `plan_steps`, `token_metrics`, `report_path`, optional `structured_payload_json`, telemetry, etc.
- `steps` (completed flags based on `next_step_index`)
- `tool_summaries` (last N tool events)

**Routing**
- Graph edge is `synthesis → persist_cleanup → END`.

---

### 7) `persist_cleanup(output: PipelineOutputState) -> PipelineOutputState`
**Purpose:** final “always-run” persistence step (belt-and-suspenders), especially if synthesis didn’t write.

**Reads (inputs)**
- `output.dataset_path`, `output.report_text`, `output.artifacts["__report_saved"]`, `output.steps`

**Writes (state changes)**
- No `AnalysisPipelineState` here; it only mutates the output envelope:
  - writes report markdown if not already saved
  - sets `output.artifacts["report_path"]` and `output.artifacts["__report_saved"]=True`
  - schedules memory append again using a reconstructed minimal `AnalysisPipelineState` snapshot (from envelope fields)

**Routing**
- Returns output, then graph goes to `END`.