"""Minimal LangGraph-based CSV analysis agent.

Flow: User supplies --instruction and --file (CSV).
1. Planner node builds JSON plan of tool steps.
2. Execute node runs each step sequentially and prints raw concatenated output.

Single-file implementation.
"""
from __future__ import annotations
import argparse
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime

from pathlib import Path

# LangGraph minimal imports
from langgraph.graph import StateGraph, END

# -------------------------------
# State definition
# -------------------------------
@dataclass
class AgentState:
    instruction: str
    dataset_path: str
    seed_text: str
    profile_report: Optional[str] = None
    profile_meta: Optional[Dict[str, Any]] = None
    plan_json: Optional[str] = None
    plan_steps: List[Dict[str, Any]] = field(default_factory=list)
    next_step_index: int = 0
    raw_execution_output: Optional[str] = None
    error_log: List[str] = field(default_factory=list)

# -------------------------------
# Utility helpers
# -------------------------------
FOOTER_PATTERN = r"<!--output_json:(.+?)-->"


def make_seed_text(instruction: str, dataset_path: str) -> str:
    return f"Dataset: {dataset_path or 'unspecified'}\nInstructions: {instruction}"


def parse_footer(md: str) -> Dict[str, Any]:
    m = re.search(FOOTER_PATTERN, md, re.DOTALL)
    if not m:
        return {}
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return {}


def classify_intent(instr: str) -> Dict[str, bool]:
    low = instr.lower()
    return {
        "visualization": any(k in low for k in ["plot", "chart", "graph", "histogram", "scatter", "box", "bar", "visualize", "pairplot"]),
        "advanced": any(k in low for k in ["eda", "analysis", "correlation", "cluster", "regression", "patterns", "pairplot"]),
    }


def infer_plot_type(instr: str) -> str:
    low = instr.lower()
    if "histogram" in low: return "histogram"
    if "scatter" in low: return "scatterplot"
    if "box" in low: return "boxplot"
    if "pairplot" in low: return "pairplot"
    if "bar" in low or "chart" in low: return "bar_chart"
    # fallback
    return "histogram"


def infer_plot_columns(plot_type: str, numeric: List[str], categorical: List[str]) -> Optional[Dict[str, Any]]:
    if plot_type == "histogram" and numeric:
        return {"x_column": numeric[0]}
    if plot_type == "scatterplot" and len(numeric) >= 2:
        return {"x_column": numeric[0], "y_column": numeric[1]}
    if plot_type == "boxplot" and numeric and categorical:
        return {"x_column": numeric[0], "hue_column": categorical[0]}
    if plot_type == "bar_chart" and categorical:
        return {"x_column": categorical[0]}
    if plot_type == "pairplot" and len(numeric) >= 2:
        return {"columns_for_pairplot": numeric[:5]}
    return None

"""Tool functions are now sourced from external mcp_tools.py."""
from mcp_tools import (
    load_and_analyze_csv,
    perform_advanced_eda_on_csv,
    generate_basic_plot,
)

# -------------------------------
# Planner node
# -------------------------------

def planner_node(state: AgentState) -> AgentState:
    # Resolve dataset path relative to current working directory when needed
    ds_path = Path(state.dataset_path or "")
    if not ds_path.is_absolute():
        ds_path = Path.cwd() / ds_path
    ds_path = ds_path.resolve()
    if not ds_path.exists():
        state.plan_json = json.dumps({"steps": []})
        state.plan_steps = []
        state.error_log.append(f"Dataset file not found: {state.dataset_path}")
        return state
    # normalize to absolute path for downstream tools
    state.dataset_path = str(ds_path)

    if state.profile_meta is None:
        rep = load_and_analyze_csv(state.dataset_path)
        state.profile_report = rep
        state.profile_meta = parse_footer(rep)

    meta = state.profile_meta or {}
    numeric = meta.get("numeric_columns", [])
    categorical = meta.get("text_columns", [])
    intent = classify_intent(state.instruction)

    steps: List[Dict[str, Any]] = []
    # Always first overview
    steps.append({
        "tool": "load_and_analyze_csv",
        "args": {"file_path": state.dataset_path},
        "why": "dataset overview"
    })

    if intent["advanced"] and len(numeric) >= 2:
        steps.append({
            "tool": "perform_advanced_eda_on_csv",
            "args": {"file_path": state.dataset_path, "auto_detect_analysis": True},
            "why": "advanced EDA"
        })

    if intent["visualization"]:
        plot_type = infer_plot_type(state.instruction)
        col_args = infer_plot_columns(plot_type, numeric, categorical)
        if col_args:
            args = {"file_path": state.dataset_path, "plot_type": plot_type, **col_args}
            steps.append({
                "tool": "generate_basic_plot",
                "args": args,
                "why": "requested visualization"
            })

    state.plan_json = json.dumps({"steps": steps}, ensure_ascii=False)
    state.plan_steps = steps
    state.next_step_index = 0
    return state

# -------------------------------
# Execute node
# -------------------------------
TOOL_MAP = {
    "load_and_analyze_csv": load_and_analyze_csv,
    "perform_advanced_eda_on_csv": perform_advanced_eda_on_csv,
    "generate_basic_plot": generate_basic_plot,
}

def execute_node(state: AgentState) -> AgentState:
    outputs: List[str] = []
    for step in state.plan_steps:
        tool_name = step.get("tool")
        fn = TOOL_MAP.get(tool_name)
        if not fn:
            outputs.append(f"Skipped unknown tool: {tool_name}")
            continue
        try:
            out = fn(**step.get("args", {}))
        except Exception as e:
            err = f"Error executing {tool_name}: {e}"
            state.error_log.append(err)
            out = err
        outputs.append(out)
    state.raw_execution_output = "\n\n--- STEP END ---\n\n".join(outputs)
    return state

# -------------------------------
# Graph construction
# -------------------------------

def build_graph():
    g = StateGraph(AgentState)
    g.add_node("planner", planner_node)
    g.add_node("executor", execute_node)
    g.set_entry_point("planner")
    g.add_edge("planner", "executor")
    g.add_edge("executor", END)
    return g.compile()

# -------------------------------
# CLI / Runner
# -------------------------------

def run(instruction: str, dataset_path: str):
    state = AgentState(
        instruction=instruction,
        dataset_path=dataset_path,
        seed_text=make_seed_text(instruction, dataset_path),
    )
    app = build_graph()
    final_state = app.invoke(state)
    # Support both AgentState return and mapping-like return (e.g., AddableValuesDict)
    try:
        plan_json = final_state.plan_json
        raw_output = final_state.raw_execution_output
        errors = final_state.error_log
    except Exception:
        # final_state may be mapping-like; try .get
        try:
            plan_json = final_state.get("plan_json")
            raw_output = final_state.get("raw_execution_output")
            errors = final_state.get("error_log")
        except Exception:
            plan_json = None
            raw_output = None
            errors = None

    print("==== PLAN JSON ====\n" + (plan_json or "{}"))
    print("\n==== RAW EXECUTION OUTPUT ====\n" + (raw_output or ""))
    if errors:
        print("\nErrors:")
        for e in errors:
            print("-", e)


def main():
    parser = argparse.ArgumentParser(description="Minimal CSV analysis agent")
    parser.add_argument("--instruction", required=True, help="User instruction/query")
    parser.add_argument("--file", required=True, help="Path to CSV dataset")
    args = parser.parse_args()
    run(args.instruction, args.file)

if __name__ == "__main__":
    main()
