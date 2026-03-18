from __future__ import annotations

import json


def _print(title: str, payload: object) -> None:
    print("=" * 72)
    print(title)
    print("=" * 72)
    print(json.dumps(payload, indent=2, ensure_ascii=True))
    print()


def main() -> int:
    _print(
        "Appendix B-1 — Status update payload (snapshot)",
        {
            "agent_task_id": "<task-id>",
            "content": "...accumulated streaming text...",
            "tool_call_chunks": [],
            "current_action": "Working in execute_step",
            "current_node": "execute_step",
        },
    )

    _print(
        "Appendix B-2 — Artifact update payload (ongoing_artifact)",
        {
            "agent_task_id": "<task-id>",
            "content": "output/temporal_plots/plot_001.png",
            "tool_calls": [],
            "current_action": "Artifact generated",
            "current_node": "execute_step",
        },
    )

    _print(
        "Appendix B-3 — HITL plan review payload (plan_review)",
        {
            "agent_task_id": "<task-id>",
            "content": [
                {
                    "plan_hash": "<hash>",
                    "plan_steps": [
                        {
                            "step_id": 1,
                            "tool": "load_and_analyze_csv",
                            "args": {"file_path": "..."},
                        },
                        {
                            "step_id": 2,
                            "tool": "generate_plot",
                            "args": {"plot_type": "..."},
                        },
                    ],
                }
            ],
            "tool_calls": [],
            "current_action": "Plan review required",
            "current_node": "planner",
        },
    )

    _print(
        "Appendix B-4 — Client resume decision payload (hitl_resume DataPart.data)",
        {
            "hitl_resume": {
                "decision": "approve",
                "feedback": "LGTM",
            }
        },
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
