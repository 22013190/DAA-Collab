"""Optional CLI entrypoints.

This repo is a partial extraction of a larger multi-agent system.
Some CLI scripts (db-agent, supervisor-agent) are intentionally optional.

These wrappers let `pyproject.toml` expose stable entrypoints without
hard-requiring those optional modules to exist in the extracted repo.
"""

from __future__ import annotations

import importlib
import sys
from typing import Callable


def _run_optional(target: str, fallback_help: str) -> None:
    """Import-and-run an optional callable.

    Args:
        target: Dotted path in the form `some.module:callable`.
        fallback_help: Human-readable next steps when the target is absent.
    """

    module_name, _, func_name = target.partition(":")
    if not module_name or not func_name:
        raise SystemExit(f"Invalid target: {target!r}")

    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as e:
        # Only treat it as optional when the missing module is the target module itself.
        if getattr(e, "name", None) == module_name.split(".")[0] or getattr(e, "name", None) == module_name:
            raise SystemExit(
                "Optional component is not present in this repo.\n"
                f"Tried to import: {module_name}\n\n"
                f"{fallback_help}\n"
            ) from None
        raise

    func: Callable[..., object] | None = getattr(module, func_name, None)
    if func is None:
        raise SystemExit(
            "Optional component was found, but entry function is missing.\n"
            f"Expected: {target}\n"
        )

    # Call the target function.
    try:
        func()
    except TypeError:
        # Some mains accept argv; pass through best-effort.
        func(sys.argv[1:])


def start_db_agent() -> None:
    _run_optional(
        "src.agents.database.__main__:start_a2a_db_agent",
        "This extraction does not include the database agent.\n"
        "If you need it, copy `src/agents/database/` from the original repo (or install the full package).",
    )


def start_supervisor_agent() -> None:
    _run_optional(
        "src.agents.planner.__main__:main",
        "This extraction does not include the supervisor/planner agent.\n"
        "If you need it, copy `src/agents/planner/` from the original repo (or install the full package).",
    )
