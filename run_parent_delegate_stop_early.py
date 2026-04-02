import asyncio
import json
import os
from typing import Any

from langchain_core.messages import HumanMessage

from src.agents.analysis.pipeline import create_analysis_pipeline_agent
from src.agents.analysis.pipeline.state import AnalysisPipelineState


def _as_dict(value: Any) -> dict:
    if isinstance(value, dict):
        return value
    try:
        return dict(value)
    except Exception:
        return {}


def _contains_delegate_tool_event(obj: Any, max_depth: int = 6) -> bool:
    """Best-effort scan of streamed update payloads for a delegate tool transcript event."""
    if max_depth <= 0:
        return False
    if isinstance(obj, dict):
        # Common shapes:
        # - {"tool_transcript": [ ... ]}
        # - {"delegate_code_interpreter": {"tool_transcript": [ ... ]}}
        tt = obj.get("tool_transcript")
        if isinstance(tt, list):
            for ev in tt:
                if isinstance(ev, dict) and ev.get("tool") == "delegate_code_interpreter":
                    return True
        for v in obj.values():
            if _contains_delegate_tool_event(v, max_depth=max_depth - 1):
                return True
        return False
    if isinstance(obj, list):
        for v in obj:
            if _contains_delegate_tool_event(v, max_depth=max_depth - 1):
                return True
        return False
    return False


async def main() -> None:
    # End-to-end defaults (do not override user-provided values).
    os.environ.setdefault("ANALYSIS_ENABLE_CODE_INTERPRETER_DELEGATION", "1")
    os.environ.setdefault("ANALYSIS_MAX_CI_DELEGATIONS", "1")
    os.environ.setdefault("TOOLGEN_SANDBOX_MODE", "subprocess")
    os.environ.setdefault("ANALYSIS_FORCE_DELEGATION_FOR_TOOL_REQUEST", "1")

    print("ENV_FORCE_DELEGATE=", os.getenv("ANALYSIS_FORCE_DELEGATION_FOR_TOOL_REQUEST"))
    print("ENV_DELEGATION_ENABLED=", os.getenv("ANALYSIS_ENABLE_CODE_INTERPRETER_DELEGATION"))

    agent = await create_analysis_pipeline_agent()

    dataset = r"C:\Users\Tmr\Desktop\DAA\DAA-Collab\traffic_incidents_data.csv"
    instruction = "Create and promote a reusable Python tool that outputs incident counts by Type as JSON."

    seed_text = f"Dataset: {dataset}\nInstructions: {instruction}"
    initial_state = AnalysisPipelineState(
        messages=[HumanMessage(content=seed_text)],
        dataset_path=dataset,
        instruction=instruction,
    )

    cfg = {"recursion_limit": 80}

    saw_delegate_start = False
    saw_delegate_event_in_updates = False

    async for mode, chunk in agent.agent.astream(
        initial_state,
        config=cfg,
        stream_mode=["custom", "updates"],
    ):
        if mode == "custom" and isinstance(chunk, dict):
            if chunk.get("event") == "node_start" and chunk.get("node") == "delegate_code_interpreter":
                saw_delegate_start = True
                print("DELEGATE_NODE_SEEN")
                continue
            if (
                saw_delegate_start
                and chunk.get("event") == "node_start"
                and chunk.get("node") == "interpret_results"
            ):
                print("INTERPRET_AFTER_DELEGATE_NODE_SEEN")
                if saw_delegate_event_in_updates:
                    print("DELEGATE_EVENT_IN_UPDATES_SEEN")
                return

        if mode == "updates" and saw_delegate_start and not saw_delegate_event_in_updates:
            try:
                if _contains_delegate_tool_event(chunk):
                    saw_delegate_event_in_updates = True
            except Exception:
                pass

    print("DELEGATE_NODE_NOT_SEEN")


if __name__ == "__main__":
    asyncio.run(main())
