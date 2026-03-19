"""Prompt factory utilities for the LangGraph analysis pipeline."""

from __future__ import annotations

from typing import Dict, Iterable, List
import os
import json

from textwrap import shorten, dedent

from .state import AnalysisPipelineState, DEFAULT_CONTEXT_TOKEN_BUDGET, DEFAULT_MESSAGE_WINDOW

"""Prompt sizing and trimming guards.

The defaults are intentionally moderate to preserve more context while staying
within typical model windows. All three can be overridden via environment
variables without code changes:
  - ANALYSIS_MAX_CHARS (int): overrides MAX_SECTION_CHARS
  - ANALYSIS_MAX_TOOL_SUMMARY_ITEMS (int): overrides MAX_TOOL_SUMMARY_ITEMS
  - ANALYSIS_SUMMARY_TRIM (int): overrides SUMMARY_TRIM
"""

# Section-level cap applied when passing blocks into prompts (context, profile, plan, etc.)
try:
    MAX_SECTION_CHARS = int(os.getenv("ANALYSIS_MAX_CHARS", "4000"))
except Exception:
    MAX_SECTION_CHARS = 4000

# Number of recent items to include from plans/transcripts/insights in summaries
try:
    MAX_TOOL_SUMMARY_ITEMS = int(os.getenv("ANALYSIS_MAX_TOOL_SUMMARY_ITEMS", "10"))
except Exception:
    MAX_TOOL_SUMMARY_ITEMS = 10

# Per-line trim used in tool transcript lines (keep concise but informative)
try:
    SUMMARY_TRIM = int(os.getenv("ANALYSIS_SUMMARY_TRIM", "1000"))
except Exception:
    SUMMARY_TRIM = 1000
# Max insights to request from the LLM during interpretation (env-overridable).
# Use a moderate default to avoid bloat; report rendering will cap to a smaller number.
try:
    MAX_INSIGHTS_TO_EXTRACT = int(os.getenv("ANALYSIS_MAX_INSIGHTS_REQUEST", "50"))
except Exception:
    MAX_INSIGHTS_TO_EXTRACT = 50

# How many last tool results to include untrimmed for authoritative inspection
try:
    MAX_UNTRIMMED_TOOL_RESULTS = int(os.getenv("ANALYSIS_MAX_UNTRIMMED_TOOL_RESULTS", "3"))
except Exception:
    MAX_UNTRIMMED_TOOL_RESULTS = 3


def _serialize_state_for_prompt(state, last_n_untrimmed: int = None) -> str:
    """Serialize a compact, deterministic JSON 'state' block for inclusion in prompts.

    - Keeps only a compact subset of AnalysisPipelineState that is authoritative for
      factual claims: dataset_profile, plan_steps, artifact_log, preprocess_profile,
      and a small untrimmed tail of tool_transcript entries for deterministic refs.
    - Wraps JSON in explicit markers so LLM won't blend it with instruction text.
    """
    # Respect runtime environment overrides (useful for tests via monkeypatch)
    if last_n_untrimmed is not None:
        last_n = last_n_untrimmed
    else:
        try:
            last_n = int(os.getenv("ANALYSIS_MAX_UNTRIMMED_TOOL_RESULTS", str(MAX_UNTRIMMED_TOOL_RESULTS)))
        except Exception:
            last_n = MAX_UNTRIMMED_TOOL_RESULTS
    try:
        tt = list(getattr(state, "tool_transcript", []) or [])
        untrimmed = tt[-last_n:]
        truncated = len(tt) > last_n
    except Exception:
        untrimmed = []
        truncated = False

    payload = {
        "dataset_profile": getattr(state, "preprocess_profile", None) or {},
        "plan_steps": getattr(state, "plan_steps", []) or [],
        "artifact_log": getattr(state, "artifact_log", []) or [],
        "tool_transcript_untrimmed": untrimmed,
        "state_truncated": truncated,
        "telemetry_hint": getattr(state, "telemetry", {}) or {},
    }
    # Sanitize snapshot-size phrasing in dataset_profile notes to avoid LLM echo like '100 rows'
    try:
        dp = payload.get("dataset_profile")
        if isinstance(dp, dict) and isinstance(dp.get("notes"), str):
            import re as _re
            dp["notes"] = _re.sub(r"\b~?\d+\s*(rows|records)\b", "a small snapshot", dp["notes"], flags=_re.IGNORECASE)
    except Exception:
        pass
    try:
        s = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        s = str(payload)
    return "<STATE_CONTEXT>\n" + s[:MAX_SECTION_CHARS] + "\n</STATE_CONTEXT>"


def _format_plan(plan_steps: Iterable[Dict]) -> str:
    """Format structured plan steps for compact display."""
    lines: List[str] = []
    for index, step in enumerate(plan_steps):
        tool = step.get("tool") or step.get("name") or "unknown_tool"
        why = step.get("why") or step.get("description") or "planned"
        lines.append(f"Step {index + 1}: {tool} — {why}")
        if len(lines) >= MAX_TOOL_SUMMARY_ITEMS:
            break
    return "\n".join(lines)


def _format_tool_transcript(tool_transcript: Iterable[Dict]) -> str:
    lines: List[str] = []
    for idx, entry in enumerate(tool_transcript):
        tool = entry.get("tool") or entry.get("name") or "tool"
        status = entry.get("status") or entry.get("outcome") or "unknown"
        snippet = entry.get("output") or entry.get("message") or ""
        trimmed = shorten(str(snippet), width=SUMMARY_TRIM, placeholder="…") if snippet else ""
        # Prefix with the zero-based transcript index to support structured evidence refs
        lines.append(f"#{idx} {tool} [{status}] {trimmed}")
        if len(lines) >= MAX_TOOL_SUMMARY_ITEMS:
            break
    return "\n".join(lines)


def _format_insights(insights: Iterable[Dict]) -> str:
    """Format insights for planner/reflection/synthesis previews.

    Avoid double-truncation: do not independently shorten each insight here;
    the caller will cap the overall block with MAX_SECTION_CHARS. This
    preserves important quantitative details that might otherwise be clipped.
    """
    lines: List[str] = []
    for idx, item in enumerate(insights):
        title = item.get("title") or item.get("name") or f"Insight {idx+1}"
        key = item.get("key_findings") or item.get("summary") or ""
        # No per-insight shorten; include as-is (stringified) for fidelity.
        value = str(key) if key is not None else ""
        lines.append(f"{title}: {value}")
        if len(lines) >= MAX_TOOL_SUMMARY_ITEMS:
            break
    return "\n".join(lines)


def build_planner_prompt(state: AnalysisPipelineState, available_tools: str) -> Dict:
    state.trim_message_window(DEFAULT_MESSAGE_WINDOW)
    profile_summary = _format_plan(state.plan_steps) if state.plan_steps else "No prior plan steps."

    context = state.context_summary or ""
    if state.historical_snippet:
        context = f"Historical context:\n{state.historical_snippet}\n\nCurrent context:\n{context}" if context else state.historical_snippet
    # Dataset profile grounding (if populated by planner setup). Limit size to avoid bloat.
    dataset_profile_block = ""
    if getattr(state, "preprocess_profile", None):
        try:
            import json as _json
            raw_profile = _json.dumps(state.preprocess_profile, ensure_ascii=False)
            dataset_profile_block = raw_profile[:MAX_SECTION_CHARS]
        except Exception:
            dataset_profile_block = str(state.preprocess_profile)[:MAX_SECTION_CHARS]

    refinement_items = []
    if getattr(state, "refinement_request", None):
        # Use most recent refinement targets first (avoid showing stale early items)
        for r in state.refinement_request[-6:]:
            if isinstance(r, str) and r.strip():
                refinement_items.append(r.strip()[:160])

    refinement_text = (
        "Missing info targets:" + "; ".join(refinement_items) if refinement_items else ""
    )

    # System prompt engineering principles applied:
    # - Explicit JSON schema
    # - Grounding requirement (use only observed columns / dataset profile)
    # - Anti-hallucination (no invented columns/tools/metrics)
    # - Minimal sufficient workflow (no ornamental plots)
    # - Refinement loop integration (address missing info items directly)
    # - Tool argument guidance (provide ONLY required args)
    # Intent/Audience/Outcome context for the planner
    # UPDATED: Replaced 'analysis' mode with 'insights_only' (textual analytical outputs, no visualization steps)
    analysis_intent = dedent("""
            Intent: Classify the user instruction into one of THREE output modes:
                - visualization_only: user asks for plots/charts/visual summaries only (no insights/recommendations/explanations requested).
                - insights_only: user asks for insights, explanations, recommendations, relationships, factors, or a report WITHOUT requesting any plot/chart/visualization.
                - mixed: user explicitly requests one or more visualizations AND also asks for insights, explanations, recommendations, or a report.

            After choosing output_mode, build a plan consistent with it:
                • visualization_only → EXACTLY one step per distinct requested visualization; no exploratory/statistical/insight steps.
                • insights_only → minimal purposeful analytical steps yielding evidence for insights/recommendations; DO NOT add visualization steps.
                • mixed → balanced sequence: essential visuals plus analytical steps (avoid redundancy; every step must add unique value).

            Always prioritize actionability and defensibility; avoid ornamental or redundant steps.
    """)
    analysis_audience = (
        "Audience: Traffic operations and planning stakeholders who need clear, defensible results."
    )
    analysis_outcome = (
        "Outcome: A well structured plan that will produce validated insights and artifacts used in the final report."
    )
    # Refinement-loop guidance: plan well, but minimize replans.
    # ONE_PASS_EVIDENCE = dedent("""
    #         A refinement loop MAY run if coverage.missing_info remains after execution.
    #         Still aim to gather sufficient evidence in the first pass and keep the plan minimal.
    #         Favor essential exploratory/statistical steps to build a sufficient evidence set.
    # """)
    # Deeper evidence gathering guidance (categorical, temporal, text, multi-factor)
    DEEPER_EVIDENCE_GATHERING = dedent("""
        Plan analytical steps that examine categorical, text, and temporal patterns — not only numeric summaries.
        Choose tools that produce evidence suitable for deeper reasoning (multi-factor comparisons).
        When applicable, include steps such as:
            • Categorical profiling and cross-tabulation (group differences, chi-square as appropriate)
            • Temporal segmentation (hour/day/weekday/seasonal patterns; rate-based where meaningful)
            • Multi-factor intersections (e.g., category × temporal × outcome)
            • Clustering/outlier detection for segmentation (numeric features; cap dimensionality)
            • Text profiling of narrative columns (common themes/terms/categories) when text exists
            • Post-hoc group comparisons after omnibus tests (e.g., ANOVA) to identify which groups differ
        Use the dataset profile to validate prerequisites and keep the plan minimal-but-sufficient within the 6-step cap.
    """)
    MODE_DECISION = dedent("""
            0. MODE DECISION (CLASSIFY FIRST):
            Before producing steps, classify the user's request into exactly one output_mode:
                - visualization_only: User primarily requests one or more plots/graphs/charts/visual summaries WITHOUT asking for insights, explanations, recommendations, root cause analysis, or a report.
                - insights_only: User requests insights, explanations, recommendations, factors, relationships, root causes, or a report WITHOUT requesting a specific visualization.
                - mixed: User requests BOTH visualizations AND analytical insights/recommendations/explanations.

            Return a JSON object under plan_meta with keys:
                output_mode (string), confidence (0–1 float), rationale (brief <180 chars justification referencing instruction phrases).

            Rules:
            - ALWAYS decide output_mode first; do not let later rules override classification.
            - If instruction mentions only visualization triggers ('show', 'plot', 'visualize', 'chart') and NO insight/explanation/recommendation language, choose visualization_only.
            - If instruction contains insight/explanation/recommendation/report language and NO visualization request, choose insights_only.
            - If instruction mixes visualization triggers AND analytical/insight language, choose mixed.
            - If instruction lists specific plot types but also asks to 'summarize briefly' or 'explain', choose mixed.
            - Prefer visualization_only when only artifacts are requested even if multiple plot types.
            - Confidence: 0.9 for clear single-mode, 0.6–0.8 for ambiguous mixed.
                - Breadth-only analytical requests containing breadth terms ("comprehensive","thorough","full","in-depth","overall assessment","holistic") AND NO visualization trigger words MUST be classified as insights_only (even without phrases like 'no charts').
    """)
    FEW_SHOT_MODE_EXAMPLES = dedent("""
            0a. OUTPUT_MODE FEW-SHOT EXAMPLES (GUIDANCE ONLY – DO NOT ECHO):
            Example 1 (visualization_only):
                Instruction: "Show a correlation heatmap of injury related columns."
                plan_meta: {"output_mode": "visualization_only", "confidence": 0.92, "rationale": "Only a specific plot requested; no insight/recommendation language."}
                steps: [{"tool": "generate_correlation_heatmap", "args": {"file_path": "<dataset>", "columns": ["injuries_total","injuries_fatal","injuries_incapacitating"], "method": "pearson", "top_k": 8}, "why": "heatmap of injury relationships"}]

            Example 2 (insights_only):
                Instruction: "Identify key factors influencing severe injuries and provide recommendations (no charts)."
                plan_meta: {"output_mode": "insights_only", "confidence": 0.9, "rationale": "Requests factors & recommendations; explicitly no charts."}
                steps (illustrative subset): [
                    {"tool": "assess_data_quality", "args": {"file_path": "<dataset>"}, "why": "validate columns before factor analysis"},
                    {"tool": "generate_correlation_heatmap", "args": {"file_path": "<dataset>", "columns": ["injuries_severe","speed_limit","weather","lighting"], "method": "spearman", "top_k": 10}, "why": "severity vs contextual factors (evidence for recommendations)"}
                ]

            Example 2b (insights_only, breadth-only request):
                Instruction: "Please do a comprehensive analysis and provide key insights and recommendations."
                plan_meta: {"output_mode": "insights_only", "confidence": 0.88, "rationale": "Broad analytical request with insights/recommendations; no visualization requested."}
                steps (illustrative subset): [
                    {"tool": "perform_advanced_eda_on_csv", "args": {"file_path": "<dataset>"}, "why": "overview of distributions, missingness, duplicates"},
                    {"tool": "analyze_temporal_patterns", "args": {"file_path": "<dataset>"}, "why": "hour/day/weekday trends for risk windows"},
                    {"tool": "perform_anova", "args": {"file_path": "<dataset>", "group_column": "weather_condition", "value_column": "injuries_total"}, "why": "injury variation across weather"},
                    {"tool": "perform_t_test", "args": {"file_path": "<dataset>", "group_column": "lighting_condition", "value_column": "injuries_fatal"}, "why": "fatality differences by lighting"}
                ]

            Example 3 (mixed):
                Instruction: "Plot injury severity distribution and explain the main drivers behind high-severity crashes."
                plan_meta: {"output_mode": "mixed", "confidence": 0.85, "rationale": "Requests both a plot (distribution) and analytical explanation of drivers."}
                steps (illustrative subset): [
                    {"tool": "generate_plot", "args": {"file_path": "<dataset>", "plot_type": "histogram", "x_column": "injuries_severe"}, "why": "distribution of severe injuries"},
                    {"tool": "generate_correlation_heatmap", "args": {"file_path": "<dataset>", "columns": ["injuries_severe","speed_limit","weather","hour"], "method": "spearman", "top_k": 8}, "why": "drivers of severity"}
                ]
    """)
    VISUALIZATION_ONLY_OVERRIDE = dedent("""
        0. VISUALIZATION-ONLY OVERRIDE (TOP PRIORITY):
        If the user's instruction explicitly requests one or more charts, plots, graphs, visualizations,
        or temporal patterns, AND does NOT request insights, analysis, recommendations, or a report:

            → Identify each DISTINCT visualization requested.
            → Produce EXACTLY ONE step per visualization.
            → Each step must call the appropriate visualization tool with only required arguments.
            → Do NOT include exploration, analysis, or statistical steps.
            → Ignore all other rules.

        Examples:
        - “Generate a correlation heatmap.” → 1 step
        - “Plot X vs Y and show a histogram of Z.” → 2 steps
        - “Give me weekly counts and a correlation heatmap.” → 2 steps
    """)

    TOOL_ARGUMENT_VALIDATION_GATE = dedent("""
        0b. TOOL ARGUMENT VALIDATION GATE (STRICT):
        Before returning the final JSON steps, VALIDATE the arguments of EACH step against the exact schema shown in the available tools list:

        - Match parameter names exactly (case-sensitive) to the schema.
        - Ensure values conform to declared types and enumerations (Literal[...] options). Do NOT invent new enum values.
        - If the user uses informal terms (e.g., "bar"), TRANSLATE to the exact enum from the schema (e.g., "bar_chart").
        - Provide ONLY required arguments unless an optional argument is explicitly necessary to satisfy the instruction.
        - If a required argument is missing, CHOOSE a valid value based on the instruction and dataset profile; do not return incomplete steps.

    Examples:
    - plot_type must be one of the tool's Literal options; "bar" → "bar_chart".
    - "heatmap of hour by weekday" or "temporal heatmap" → use generate_plot with plot_type="temporal_heatmap" and infer axes (hour × weekday) if not provided.
    - correlation heatmap (generate_correlation_heatmap) requires ≥2 numeric columns; if none exist, skip that visualization or select a more suitable one.
    """)

    SINGLE_TOOL_FOCUS = dedent("""
        0c. PLANNING SEQUENCE (FOCUS ON ONE TOOL):
        For each step, first SELECT the single best tool by reading its description carefully, THEN assemble arguments that satisfy ONLY that tool's schema.
        Do not mix parameters from multiple tools. Validate against that one schema before moving to the next step.
    """)

    VIZ_ARGUMENT_PATTERNS = dedent("""
        0d. VISUALIZATION ARGUMENT PATTERNS (GUIDANCE):
        - Bar chart (categorical counts): set x_column to the categorical/bin (e.g., day_of_week) and OMIT y_column to count occurrences. Do NOT set y_column to the same coded field.
        - Temporal day-of-week: if a datetime exists, prefer temporal_bar with x_column='weekday' to show names; otherwise bar_chart with x=the day-of-week field (counts) is acceptable.
        - Naming: prefer human-friendly axis labels (e.g., 'Weekday' instead of 'crash_day_of_week') and order days Monday→Sunday (or dataset’s convention).
        - Temporal heatmaps: use count-based pivots for temporal matrices (e.g., weekday vs hour). Avoid 2×2 Pearson heatmaps which add little value. For correlation heatmaps, use the dedicated generate_correlation_heatmap tool.
        - Correlation overview: If ≤5 numeric columns, prefer pairplot; if more numeric columns, use generate_correlation_heatmap (limit columns via 'columns' or 'top_k').
        - Meaningful numeric pair selection: choose pairs that could plausibly have explanatory or outcome relationship (e.g., num_units vs injuries_total). Avoid random pairs with no interpretable link.
        - NEVER correlate or scatterplot raw geographic coordinate pairs (latitude vs longitude, lat vs lon, x vs y, easting vs northing). These co-vary due to spatial embedding, not a meaningful relationship. Skip unless user explicitly requests spatial clustering (then choose a different appropriate tool if available, otherwise omit).
        - Exclude identifier fields (id, record_id, crash_id, uuid) from correlations and scatterplots; they are arbitrary keys.
        - Avoid plotting aggregates directly against their component variables (e.g., injuries_total vs injuries_non_incapacitating + injuries_reported_not_evident) when the total is a sum—this inflates trivial correlations. Prefer either the total OR the component set, not both in the same correlation heatmap/pairplot.
        - Drop near-constant columns (low variance) from scatterplots, pairplots, and correlation heatmaps; they add no analytical value.
        - If two columns look like duplicates or simple transformations (hour vs crash_hour; weekday_num vs day_of_week code), retain only one.
        - Before choosing scatterplot: verify both numeric columns have at least 4 distinct values and are not both temporal indices (e.g., hour vs month) unless the instruction explicitly requests that.
        - When selecting bar_chart with numeric y_column, ensure x_column is categorical and y_column is numeric; if both numeric, prefer scatterplot.
    """)

    TOOL_PREREQUISITES = dedent("""
        0e. TOOL PREREQUISITES (STRICT BUT MINIMAL):
        Use the dataset profile (if available) to ensure prerequisites are met before choosing a plot. If uncertain, insert ONE quick profiling step from the available tools to confirm types.

        - generate_plot prerequisites by plot_type:
          • histogram, boxplot → require x_column to be numeric; if not available, choose a categorical chart instead of forcing numeric plots.
          • scatterplot → requires BOTH x_column and y_column and BOTH must be numeric; otherwise pick bar_chart or another suitable plot.
                    • generate_correlation_heatmap → requires at least 2 numeric columns; optionally pass 'columns' subset; choose method (pearson/spearman/kendall); limit width via 'top_k' (~12).
                    • pairplot → numeric columns only (keep ≤5 columns for readability).
          • bar_chart → when plotting categorical counts, provide ONLY x_column and OMIT y_column.
          • temporal_* → requires a parseable datetime column; if unsure, prefer auto_temporal or add one profiling/conversion step. Prefer weekday names and ordered Monday→Sunday.

        - If prerequisites cannot be satisfied from the profile, EITHER select a more suitable plot type OR add a single minimal pre-check step before plotting. Keep visualization-only requests minimal (avoid extra analysis).
        - Correlation sanity filter: automatically exclude geo coordinate pairs (any column names containing ['lat','lon','latitude','longitude','easting','northing']) and pure identifiers (regex: '.*id$','uuid','record_id') from correlation heatmaps unless explicitly requested in instruction text.
        - Pairplot / correlation heatmap column curation order: (1) drop identifiers & geo coords; (2) drop near-constant numeric columns (variance≈0); (3) if aggregate + components present, keep aggregate OR top 2 components (whichever yields more actionable differentiation); (4) apply top_k limit.
    """)

    # === Added: Semantic reasoning & structured WHY requirements ===
    SEMANTIC_REASONING = dedent("""
        0f. SEMANTIC COLUMN REASONING (DOMAIN-DRIVEN):
        Classify ONLY observed columns (no hallucination) into categories:
          outcome, exposure, temporal, environmental, behavioral, infrastructure, spatial, identifier, aggregate, derived.
        Heuristic name fragments (case-insensitive):
          outcome: injur, fatal, severity, damage, harm, casualty
          exposure: num_unit, vehicles, volume, duration
          temporal: date, time, hour, weekday, day_of_week, month, season, year
          environmental: weather, lighting, road_surface, visibility, temperature, condition
          behavioral: speed, impair, alcohol, distract, drug, seatbelt
          infrastructure: speed_limit, lane, intersection, road_type, median, control, signal
          spatial: lat, lon, latitude, longitude, segment, location, region, county
          identifier: id, uuid, record, crash_id
          aggregate: total, sum, overall
          derived: rate, ratio, pct, percent, flag

        Relevance scoring (0–5):
          correlation/relationship intent → outcome vs (environmental|behavioral|infrastructure|temporal|exposure)=5; outcome vs spatial categorical=4.
          distribution intent → high-variance outcome/exposure=5; temporal numeric=4; environmental numeric=3.
          temporal pattern intent → temporal+outcome/exposure=5; temporal+environmental=4.
          severity comparison intent → outcome severity vs environmental/behavioral/infrastructure=5; vs temporal=4.

        Quality gates: numeric distinct ≥4; non-null count sufficient; avoid near-constant variance; for categorical imbalance (>90% one level) either annotate or choose alternative.
        Fallback: if no outcome column, use incident_count proxy and annotate limitation.
    """)

    WHY_FIELD_REQUIREMENTS = dedent("""
            0g. WHY FIELD STRUCTURE (RELAXED):
            Prefer a compact JSON object with keys:
                intent_alignment, selected_columns, semantic_categories, relevance_scores, data_quality
                Optional: method_choice, exclusions, actionability_note, fallback_note
            If JSON formatting fails, provide concise text; the pipeline will accept either. Keep <800 chars.
    """)

    COLUMN_SELECTION_POLICY = dedent("""
        0h. COLUMN SELECTION POLICY (USE SCORES):
        When selecting columns for ANY visualization or analytical step:
          - Use relevance_scores and choose only HIGH-SCORE columns (score ≥4) that pass quality gates (distinct ≥4 for numeric analytical use, non_null sufficient, not near-constant).
          - NEVER include all available columns; limit to a purposeful subset:
              • correlation_heatmap: top ranked outcome + up to (environmental|behavioral|infrastructure|temporal|exposure) factors (max 12 total).
              • pairplot: ≤5 high-score numeric columns (avoid clutter).
              • scatterplot: exactly 2 numeric columns (highest complementary scores, outcome vs factor).
              • boxplot/histogram: 1 numeric outcome/exposure OR outcome stratified by one categorical factor.
              • temporal_heatmap: 2 temporal/context dimensions (weekday vs hour) OR temporal vs outcome rate.
          - PRIORITIZE pairings: outcome ↔ (exposure|environmental|behavioral|temporal) for correlation/relationship tasks.
          - EXCLUDE identifiers, raw geo coordinate pairs, near-constant fields, duplicates, and aggregates paired with their direct components.
          - If fewer than 2 high-score columns exist for requested visualization, FALLBACK gracefully (e.g., single distribution plot) and set fallback_note in 'why'.
          - Reflect chosen subset explicitly in selected_columns and ensure relevance_scores JSON shows their scores.
          - Document exclusions (with reasons) in exclusions list inside 'why'.
    """)

    STRUCTURED_SUBTASKS = dedent("""
        0i. STRUCTURED SUBTASKS (FOLLOW EXACTLY):
        1. Read dataset_profile.preprocess_profile: use semantic_categories, relevance_scores, column_quality.
        2. Identify user intent (correlation, distribution, temporal, comparison, severity).
        3. Build candidate set by excluding identifiers and raw geo coords; apply quality gates (numeric distinct≥4, non_null≥30; near-constant removed).
        4. Sort candidates by relevance_scores DESC; break ties by category weight: outcome>exposure>environmental>behavioral>infrastructure>temporal>spatial>derived>unknown>identifier.
        5. Select minimal subset per visualization:
           - correlation: top outcome + top 3–10 explanatory columns (limit width)
           - pairplot: ≤5 numeric
           - scatter: exactly 2 numeric (outcome vs factor)
           - histogram/box: 1 numeric (optionally stratify by one categorical if balanced)
           - temporal heatmap: weekday×hour or temporal×outcome rate
        6. Write 'why' including chosen columns, scores, categories, quality, and any exclusions.
    """)

    system_msg = (
        "You are an expert data analysis workflow planner for traffic & incident datasets. "
        "Return ONLY JSON with schema {\"plan_meta\": {\"output_mode\": \"visualization_only|insights_only|mixed\", \"confidence\": <float>, \"rationale\": str}, \"steps\": [{\"tool\": str, \"args\": {}, \"why\": str}]}\n"
        f"{analysis_intent}\n{analysis_audience}\n{analysis_outcome}\n"
        "Rules:\n"
        "CRITICAL COLUMN SELECTION RULE: Use semantic_categories and relevance_scores from preprocess_profile (dataset_profile) to choose columns. "
        "Select top-ranked columns that align with the user instruction intent. Avoid columns with relevance <2 unless no higher-score alternatives exist.\n"
        f"{MODE_DECISION}\n{FEW_SHOT_MODE_EXAMPLES}\n{VISUALIZATION_ONLY_OVERRIDE}\n{TOOL_ARGUMENT_VALIDATION_GATE}\n{SINGLE_TOOL_FOCUS}\n{VIZ_ARGUMENT_PATTERNS}\n{TOOL_PREREQUISITES}\n{SEMANTIC_REASONING}\n{WHY_FIELD_REQUIREMENTS}\n{COLUMN_SELECTION_POLICY}\n{STRUCTURED_SUBTASKS}\n"
        # f"{ONE_PASS_EVIDENCE}\n"
        f"{DEEPER_EVIDENCE_GATHERING}\n"
        "1. MAX 6 steps; each step must contribute unique analytical value.\n"
        "2. EVERY step must reference real columns/fields from the dataset profile block.\n"
        "3. Do NOT invent columns, metrics, filenames, or tools not shown in available tools list.\n"
        "4. READ tool descriptions carefully - choose tools whose descriptions match your analytical needs.\n"
        "5. Build a logical analytical sequence:\n"
        "   - Start with exploration (if needed): understand data distributions, missing values\n"
        "   - Add analysis: statistical tests, correlations, temporal patterns\n"
        "   - Generate evidence: visualizations that support findings\n"
        "6. Each tool's 'why' field must explain: what question does this tool answer?\n"
        "7. If refinement targets exist, select tools whose descriptions indicate they can provide that missing information.\n"
        "8. Avoid redundant analysis - if a tool provides multiple outputs, don't duplicate with another tool.\n"
        "9. For arguments: provide only required fields. Use exact column names from dataset profile and obey literal enum values exactly.\n"
        "10. Never add explanation outside JSON; no markdown or commentary."
    )

    # Structured state block for grounding (authoritative for factual claims)
    state_block = _serialize_state_for_prompt(state)

    # Provide planner context object
    return {
        "system": system_msg,
        "context": context[:MAX_SECTION_CHARS],
        "state": state_block,
        "dataset_profile": dataset_profile_block,
        "refinement": refinement_text[:MAX_SECTION_CHARS],
        "profile": profile_summary[:MAX_SECTION_CHARS],
        "instruction": state.instruction,
        "tools": available_tools,
    }


def build_reflection_prompt(state: AnalysisPipelineState) -> Dict:
    transcript = _format_tool_transcript(state.tool_transcript)
    insight_block = _format_insights(state.insights) if state.insights else ""
    state_block = _serialize_state_for_prompt(state)
    return {
        "system": "Evaluate recent tool results and decide next best action. Use the STATE JSON block for facts and context; you may use the context_summary for intent only.",
        "context": state.context_summary[:MAX_SECTION_CHARS],
        "state": state_block,
        "tool_transcript": transcript[:MAX_SECTION_CHARS],
        "insights": insight_block[:MAX_SECTION_CHARS],
        "instruction": state.instruction,
        "token_budget": DEFAULT_CONTEXT_TOKEN_BUDGET,
    }


def build_synthesis_prompt(state: AnalysisPipelineState) -> Dict:
    """
    SYNTHESIS = FORMATTING ONLY.
    - MUST NOT create new insights.
    - MUST NOT reinterpret evidence.
    - MUST ONLY use insights already stored in state.insights.
    """

    transcript = _format_tool_transcript(state.tool_transcript)
    artifacts = "\n".join(state.artifact_log[-MAX_TOOL_SUMMARY_ITEMS:])
    insights_text = _format_insights(state.insights) if state.insights else ""
    dataset = state.dataset_path or "unspecified"
    state_block = _serialize_state_for_prompt(state)

    # Provide both strict system/human prompts and explicit fields the runner expects
    plan_summary = _format_plan(state.plan_steps) if state.plan_steps else "No prior plan steps."

    return {
        "system": (
            "You are an expert report writer, generating a clean, defensible report. "
            "You MUST NOT introduce new insights — interpretation already generated the full set.\n\n"

            "YOUR ROLE:\n"
            "- Read the INSIGHTS block (these are final and authoritative).\n"
            # "- Rephrase them for a non-analyst audience.\n"
            "- Group them logically when needed.\n"
            "- Build a structured report using ONLY those insights.\n"
            "- If referencing evidence, use the evidence_ref already provided.\n\n"

            "ABSOLUTE CONSTRAINTS:\n"
            "- DO NOT invent new findings.\n"
            "- DO NOT weaken or take away depth from insights.\n"
            "- DO NOT add new numbers.\n"
            "- DO NOT add new insights.\n"
            "- DO NOT reinterpret the provided insights.\n\n"
            "- DO NOT reinterpret the tool outputs — rely ONLY on the provided insights.\n\n"

            "No additional analysis regarding insights occurs after this step.\n"
            "Do not attempt to reinterpret evidence or generate new insights.\n"
            "If an insight appears incomplete, you must still use it as-is without expanding it.\n\n"

            "RETURN JSON ONLY with the schema:\n"
            "{\n"
            '  "title": str,\n'
            '  "analysis_summary": str,\n'
            '  "dataset_overview": {\n'
            '        "path": str,\n'
            '        "rows": int | null,\n'
            '        "columns": int | null,\n'
            '        "notes": str\n'
            '  },\n'
            '  "insights": [  # rephrased but meaning-preserving\n'
            '     {\n'
            '       "statement": str,\n'
            '       "evidence_snippet": str,\n'
            '       "evidence_source": "transcript" | "artifact",\n'
            '       "artifact": str | null,\n'
            '       "evidence_ref": {\n'
            '            "source": "transcript" | "artifact",\n'
            '            "tool": str | null,\n'
            '            "line_index": int | null,\n'
            '            "artifact": str | null\n'
            '        }\n'
            '     }\n'
            '  ],\n'
            '  "next_steps": [str],\n'
            '  "recommendations": [\n'
            '      {\n'
            '          "recommendation": str,\n'
            '          "category": "safety" | "operations" | "infrastructure" | "policy" | "data",\n'
            '          "evidence_snippet": str,\n'
            '          "evidence_source": "transcript" | "artifact",\n'
            '          "artifact": str | null,\n'
            '          "evidence_ref": {\n'
            '                "source": "transcript" | "artifact",\n'
            '                "tool": str | null,\n'
            '                "line_index": int | null,\n'
            '                "artifact": str | null\n'
            '          }\n'
            '      }\n'
            '  ]\n'
            "}\n"
        ),

        "human": (
            f"DATASET: {dataset}\n\n"
            "STATE CONTEXT (for grounding only):\n"
            f"{state_block}\n\n"
            "INSIGHTS (AUTHORITATIVE – DO NOT ADD OR MODIFY):\n"
            f"{insights_text}\n\n"
            "Task: Format these insights into the structured JSON report. "
            "Do not introduce new facts.\n"
        ),
        # Extra fields for runner compatibility
        "context_summary": state.context_summary,
        "state": state_block,
        "plan_summary": plan_summary,
        "tool_transcript": transcript,
        "artifacts": artifacts,
        "insights": insights_text,
        "instruction": state.instruction,
    }



def build_interpretation_prompt(state: AnalysisPipelineState, tool_outputs_text: str) -> Dict:
    """
    INTERPRETATION = SINGLE SOURCE OF TRUTH FOR INSIGHTS.
    Extract ALL insights strictly from tool outputs.
    Synthesis will not generate new insights — only format them.
    """

    state_block = _serialize_state_for_prompt(state)

    return {
        "system": (
            "You are a highly analytically intelligent data analysis and traffic engineer. "
            "Your output is the ONLY SOURCE of analytically intelligent insights for the final report. "
            "Synthesis will NOT create new insights, so extract EVERYTHING analytically intelligent and meaningful now.\n\n"

            "TASK:\n"
            "- Understand and ponder the the provided tool outputs, use them as a point of reference for evidence to generate analytically intelligent and meaningful insights.\n"
            "- Extract ALL valuable analytically intelligent and meaningful insights (both obvious and subtle).\n"
            "- Write each insight describing and explaining in plain language and in a analytically intelligent and meaningful way so a non-analyst understands it.\n"
            "- Include Who/Where/When and concrete numeric evidence whenever available.\n"
            "- Capture insights across ALL analytical dimensions: distributions, correlations, group differences, "
            "temporal patterns, anomalies, clusters, high-risk segments, etc.\n"
            "- Avoid duplicates. Keep unique insights even if related.\n\n"
            "REASONING PERMISSION (GROUNDED):\n"
            "- Perform analytically intelligent reasoning ON TOP of the evidence.\n"
            "- Combine multiple tool outputs to infer multi-factor patterns (e.g., time × category × outcome).\n"
            "- Identify patterns and impacts implied by the evidence even if not explicitly computed.\n"
            # "- Stay hallucination-free: EVERY claim must trace to quoted evidence (transcript/artifact).\n\n"

            "DEEPER INSIGHT CATEGORIES (COVER WHERE RELEVANT):\n"
            "- Temporal segmentation (hour/day/weekday/season) and trends.\n"
            "- Categorical distributions and group comparisons.\n"
            "- Multi-factor intersections (category × temporal × numeric outcome).\n"
            "- Clustering/segments and notable outliers.\n"
            "- Interactions between numeric and categorical variables.\n"
            "- Text-theme patterns if narrative/text columns exist (common terms/categories).\n\n"

            "REASONING STEPS (INTERNAL ONLY — DO NOT OUTPUT):\n"
            "1. Identify major numeric patterns (distributions, correlations).\n"
            "2. Identify categorical differences and imbalances.\n"
            "3. Identify temporal/seasonal windows and rate changes.\n"
            "4. Identify multi-factor interactions (e.g., category × time × outcome).\n"
            "5. Identify anomalies/outliers and meaningful clusters.\n"
            "6. Rank findings by operational impact and strength of evidence.\n"
            "Only return the JSON schema specified below; do not include your reasoning steps.\n\n"

            "INSIGHT QUALITY (MEANINGFUL & ACTIONABLE):\n"
            "- All insights must be analytically intelligent and meaningful.\n"
            "- Prefer insights that explain impact and plausible contributing factors over mere observations.\n"
            "- When suggesting causes, stay grounded: describe associations and contributors without overstating causality.\n"
            "- Tie insights to decisions: safety, operations, infrastructure, policy, or data follow-ups.\n\n"

            "A refinement loop MAY follow if coverage.missing_info remains.\n"
            "Still be exhaustive: extract all meaningful insights now and make missing_info concrete and actionable.\n\n"

            "EVIDENCE REQUIREMENTS (STRICT BUT FLEXIBLE FORMAT):\n"
            "- Every insight MUST be grounded in the provided tool outputs or artifacts or sound logic derived from analysis of the dataset and tool outputs. When quoting is impractical, use an evidence_summary that cites specific references via ‘ref’ or ‘refs’ and stays strictly within the evidence.\n"
            "- Prefer a literal quoted snippet for key numbers; OR provide a concise evidence_summary with precise references.\n"
            "- Provide at least one structured reference: 'ref' (single) or 'refs' (list of refs).\n"
            "- Do NOT invent columns or numbers. Derived metrics are allowed only if computed from cited values.\n"
            "- All claims must be traceable to the cited references; otherwise omit or move to missing_info.\n\n"

            "COVERAGE REQUIREMENTS:\n"
            "- answers_query = true only if insights fully answer the user's instruction.\n"
            "- missing_info MUST list specific gaps (e.g., 'relationship between weather and severity'). "
            "Do NOT use vague phrases ('needs more analysis').\n\n"

            "RETURN JSON ONLY with the schema below. Optional fields are allowed (reasoning, inference_level, refs, evidence_summary) — include them when helpful for deeper reasoning:\n"
            "{\n"
            '  "insights": [\n'
            '    {\n'
            '      "statement": str,\n'
            '      "evidence": str,\n'
            '      "evidence_summary": str | null,\n'
            '      "ref": {\n'
            '         "source": "transcript" | "artifact",\n'
            '         "tool": str | null,\n'
            '         "line_index": int | null,\n'
            '         "artifact": str | null\n'
            '      },\n'
            '      "refs": [ {"source": "transcript" | "artifact", "tool": str | null, "line_index": int | null, "artifact": str | null} ] | null,\n'
            '      "reasoning": str | null,\n'
            '      "inference_level": "direct" | "cross_evidence" | "hypothesis" | null\n'
            '    }\n'
            '  ],\n'
            '  "coverage": {\n'
            '      "answers_query": bool,\n'
            '      "missing_info": [str]\n'
            '  }\n'
            "}\n"
        ),

        "human": (
            f"USER INSTRUCTION:\n{state.instruction}\n\n"
            f"CONTEXT SUMMARY:\n{state.context_summary or ''}\n\n"
            f"STATE CONTEXT (for grounding only):\n{state_block}\n\n"
            "TOOL OUTPUTS (use ONLY these for evidence; do not invent anything):\n"
            f"{tool_outputs_text}\n"
        ),

        # Runner compatibility (runner.interpret_results expects these keys)
        "instruction": state.instruction,
        "tool_results": tool_outputs_text,
    }
