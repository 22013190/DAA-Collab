from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterable


def _is_mcp_tool_decorator(dec: ast.expr) -> bool:
    # Accept:
    # - @mcp.tool
    # - @mcp.tool()
    if isinstance(dec, ast.Call):
        dec = dec.func
    return (
        isinstance(dec, ast.Attribute)
        and isinstance(dec.value, ast.Name)
        and dec.value.id == "mcp"
        and dec.attr == "tool"
    )


def _extract_tool_function_names(py_file: Path) -> list[str]:
    tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
    names: list[str] = []
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if any(_is_mcp_tool_decorator(dec) for dec in node.decorator_list or []):
            names.append(node.name)
    return sorted(names)


def _print_section(title: str, names: Iterable[str]) -> None:
    names_list = list(names)
    print("=" * 72)
    print(title)
    print(f"Registered tools: {len(names_list)}")
    print("=" * 72)
    for n in names_list:
        print(n)
    print()


def main() -> int:
    # Deterministic extraction: read the server source files and list all
    # @mcp.tool-decorated function names. This avoids depending on FastMCP
    # internal registries (which may be async or version-dependent).
    repo_root = Path(__file__).resolve().parents[2]
    analysis_server_file = repo_root / "agent_system" / "src" / "mcp_servers" / "analysis" / "fastmcp_server.py"
    data_process_server_file = repo_root / "agent_system" / "src" / "mcp_servers" / "analysis" / "data_processing_server.py"

    _print_section(
        "MCP TOOL NAMES (from @mcp.tool decorators) — analysis server",
        _extract_tool_function_names(analysis_server_file),
    )
    _print_section(
        "MCP TOOL NAMES (from @mcp.tool decorators) — data processing server",
        _extract_tool_function_names(data_process_server_file),
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
