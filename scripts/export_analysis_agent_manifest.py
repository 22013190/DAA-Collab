"""Export a manifest for extracting the Analysis Agent into a new repo.

Outputs:
- Internal file list (paths to copy)
- External import modules (3rd-party dependencies)

Design goals:
- No importing project modules (avoids side effects)
- Deterministic AST-based import scanning
- Understands the current package layout:
  - import roots like `agents.*`, `mcp_servers.*`, `structures.*` live under `agent_system/src/`
  - `packages.*` lives under `agent_system/packages/`
"""

from __future__ import annotations

import ast
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class ScanConfig:
    agent_system_root: Path
    src_root: Path

    entry_files: tuple[Path, ...]

    # Module roots that should be considered internal, even if they don't map
    # neatly to src_root (e.g., `packages.*`)
    internal_root_prefixes: tuple[str, ...] = (
        "agents",
        "mcp_servers",
        "structures",
        "src",
        "packages",
        "config",
        "utils",
    )

    # For an analysis-agent-only repo, exclude database agent/server code paths.
    # These are optional in the current project and can be omitted safely.
    exclude_rel_prefixes: tuple[str, ...] = (
        "src/agents/database/",
        "src/mcp_servers/database/",
        "src/structures/interfaces/database_agent_schema.py",
    )


def _is_excluded_file(path: Path, *, cfg: ScanConfig) -> bool:
    try:
        rel = path.resolve().relative_to(cfg.agent_system_root.resolve()).as_posix()
    except Exception:
        return False
    return any(rel.startswith(prefix) for prefix in cfg.exclude_rel_prefixes)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _module_name_for_file(file_path: Path, *, src_root: Path) -> str | None:
    """Compute dotted module name for a file under src_root."""
    try:
        rel = file_path.resolve().relative_to(src_root.resolve())
    except Exception:
        return None
    parts = list(rel.parts)
    if not parts:
        return None
    if parts[-1] == "__init__.py":
        parts = parts[:-1]
    else:
        parts[-1] = parts[-1].rsplit(".", 1)[0]
    return ".".join(parts)


def _resolve_module_to_file(module: str, *, cfg: ScanConfig) -> Path | None:
    """Resolve a dotted module to a .py or package __init__.py without importing."""
    if not module:
        return None

    # Decide base root.
    if module.startswith("packages."):
        base = cfg.agent_system_root
    else:
        base = cfg.src_root

    rel = Path(*module.split("."))

    file_candidate = base / (str(rel) + ".py")
    if file_candidate.exists():
        return file_candidate

    init_candidate = base / rel / "__init__.py"
    if init_candidate.exists():
        return init_candidate

    # Namespace packages (folder without __init__.py) are valid; return None here,
    # but the caller may still want to copy the directory. We'll handle dirs later.
    return None


def _iter_imports(tree: ast.AST) -> Iterable[tuple[str, str]]:
    """Yield (kind, module_string) where kind in {'import','from'}.

    For `import x.y`, yields 'import', 'x.y'
    For `from x.y import z`, yields 'from', 'x.y'
    For `from . import z`, yields 'from', '' (handled via relative logic)
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name:
                    yield "import", alias.name
        elif isinstance(node, ast.ImportFrom):
            yield "from", node.module or ""


def _apply_relative_import(module: str, level: int, *, file_module: str | None) -> str:
    if level <= 0:
        return module
    if not file_module:
        return module

    base_parts = file_module.split(".")
    # If file_module is a module (not a package), drop last segment
    # because relative imports are from the containing package.
    if base_parts:
        base_parts = base_parts[:-1]

    # Move up `level-1` levels from the containing package.
    ups = max(level - 1, 0)
    if ups > 0:
        base_parts = base_parts[: -ups] if ups <= len(base_parts) else []

    if module:
        return ".".join([*base_parts, module]) if base_parts else module
    return ".".join(base_parts)


def _top_level(mod: str) -> str:
    return mod.split(".", 1)[0] if mod else ""


def _is_internal_module(mod: str, *, cfg: ScanConfig) -> bool:
    top = _top_level(mod)
    return top in set(cfg.internal_root_prefixes)


def scan_manifest(cfg: ScanConfig) -> dict:
    stdlib = getattr(sys, "stdlib_module_names", set())

    seen_files: set[Path] = set()
    to_scan: list[Path] = [p.resolve() for p in cfg.entry_files if p.exists()]

    internal_files: set[Path] = set()
    external_modules: set[str] = set()

    while to_scan:
        file_path = to_scan.pop()
        if _is_excluded_file(file_path, cfg=cfg):
            continue
        if file_path in seen_files:
            continue
        seen_files.add(file_path)

        internal_files.add(file_path)

        try:
            tree = ast.parse(_read_text(file_path))
        except Exception:
            continue

        file_module = _module_name_for_file(file_path, src_root=cfg.src_root)

        # Local sibling modules (common in subpackages like packages/simple_py_logger/src)
        sibling_dir = file_path.parent

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    mod = alias.name or ""
                    if not mod:
                        continue
                    top = _top_level(mod)
                    if _is_internal_module(mod, cfg=cfg):
                        resolved = _resolve_module_to_file(mod, cfg=cfg)
                        if resolved and resolved not in seen_files and not _is_excluded_file(resolved, cfg=cfg):
                            to_scan.append(resolved)
                    else:
                        # If the import matches a sibling module file, treat as internal.
                        try:
                            sib = sibling_dir / f"{top}.py"
                            if sib.exists() and sib not in seen_files and not _is_excluded_file(sib, cfg=cfg):
                                to_scan.append(sib)
                                continue
                        except Exception:
                            pass
                        if top and top not in stdlib:
                            external_modules.add(top)

            elif isinstance(node, ast.ImportFrom):
                base_mod = node.module or ""
                base_mod = _apply_relative_import(base_mod, node.level or 0, file_module=file_module)
                top = _top_level(base_mod)

                if _is_internal_module(base_mod, cfg=cfg):
                    resolved = _resolve_module_to_file(base_mod, cfg=cfg)
                    if resolved and resolved not in seen_files and not _is_excluded_file(resolved, cfg=cfg):
                        to_scan.append(resolved)
                else:
                    # `from handler_helpers import X` pattern in subpackages.
                    try:
                        sib = sibling_dir / f"{top}.py"
                        if sib.exists() and sib not in seen_files and not _is_excluded_file(sib, cfg=cfg):
                            to_scan.append(sib)
                            continue
                    except Exception:
                        pass
                    if top and top not in stdlib:
                        external_modules.add(top)

    # Normalize output paths
    def _rel(p: Path) -> str:
        try:
            return p.resolve().relative_to(cfg.agent_system_root.resolve()).as_posix()
        except Exception:
            return p.as_posix()

    internal_rel = sorted({_rel(p) for p in internal_files})
    internal_rel = [p for p in internal_rel if not any(p.startswith(prefix) for prefix in cfg.exclude_rel_prefixes)]

    # Also include directories needed for namespace packages or data files.
    internal_dirs = sorted({str(Path(p).parent).replace("\\", "/") for p in internal_rel})

    return {
        "entry_files": [
            _rel(p) for p in cfg.entry_files if p.exists()
        ],
        "internal_files": internal_rel,
        "internal_dirs": internal_dirs,
        "external_top_level_modules": sorted(external_modules),
        "notes": {
            "how_to_use": "Copy internal_files (and any needed sibling non-.py assets) into the new repo preserving paths; install pip deps providing external_top_level_modules.",
            "warning": "Mapping module names -> exact pip package names is not always 1:1 (e.g., sklearn -> scikit-learn). Use pyproject.toml as source of truth.",
        },
    }


def main() -> None:
    agent_system_root = Path(__file__).resolve().parents[1]
    src_root = agent_system_root / "src"

    entry_files = (
        src_root / "agents" / "analysis" / "a2a_compliant_service.py",
        src_root / "agents" / "analysis" / "langgraph_react_analysis_agent.py",
        src_root / "agents" / "analysis" / "analysis_config.py",
        src_root / "agents" / "analysis" / "pipeline" / "runner.py",
        src_root / "agents" / "analysis" / "pipeline" / "nodes.py",
        src_root / "mcp_servers" / "config.py",
        src_root / "mcp_servers" / "utils.py",
        src_root / "mcp_servers" / "analysis" / "fastmcp_server.py",
        src_root / "mcp_servers" / "analysis" / "data_processing_server.py",
        src_root / "structures" / "interfaces" / "a2a_schema.py",
        src_root / "structures" / "interfaces" / "tool.py",
        src_root / "config.py",
        src_root / "utils.py",
    )

    cfg = ScanConfig(
        agent_system_root=agent_system_root,
        src_root=src_root,
        entry_files=entry_files,
    )

    manifest = scan_manifest(cfg)

    out_dir = agent_system_root / "output" / "analysis_agent_extraction"
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "manifest.json"
    txt_path = out_dir / "manifest.txt"

    json_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    lines: list[str] = []
    lines.append("ANALYSIS AGENT EXTRACTION MANIFEST")
    lines.append("=" * 34)
    lines.append("")
    lines.append("Entry files:")
    for p in manifest["entry_files"]:
        lines.append(f"  - {p}")
    lines.append("")

    lines.append(f"Internal files to copy ({len(manifest['internal_files'])}):")
    for p in manifest["internal_files"]:
        lines.append(f"  - {p}")
    lines.append("")

    lines.append("External top-level modules (pip deps, approximate):")
    for m in manifest["external_top_level_modules"]:
        lines.append(f"  - {m}")
    lines.append("")

    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote: {json_path}")
    print(f"Wrote: {txt_path}")


if __name__ == "__main__":
    main()
