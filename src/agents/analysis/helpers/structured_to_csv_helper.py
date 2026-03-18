"""Dependency-free helper to locate nested tabular payloads and write CSVs.
This module intentionally avoids heavy external imports at import time; pandas is imported only inside the function.
"""
import json
from pathlib import Path
from datetime import datetime
import re as _re
from typing import Any


def create_temp_dataset_from_structured_data(structured_data: Any, base_path: Path | str = "temp_datasets") -> str:
    """Create a temporary dataset file from structured data.

    Args:
        structured_data: The JSON-like payload to inspect.
        base_path: Optional base directory to write temp files to (Path or str).

    Returns:
        Path to the created CSV or JSON file as string.
    """
    temp_dir = Path(base_path)
    temp_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def _find_best_tabular_candidate(obj, max_depth=6, depth=0):
        if depth > max_depth:
            return None
        if isinstance(obj, list) and obj and all(isinstance(x, dict) for x in obj):
            return (None, None, obj)
        if isinstance(obj, dict):
            for k in ("db_data", "dataset_json", "dataset", "data"):
                if k in obj:
                    v = obj[k]
                    if isinstance(v, list) and v and all(isinstance(x, dict) for x in v):
                        return (obj, k, v)
                    if isinstance(v, list) and v and isinstance(v[0], str):
                        try:
                            parsed = json.loads(v[0])
                            if isinstance(parsed, list) and parsed and all(isinstance(x, dict) for x in parsed):
                                return (obj, k, parsed)
                        except Exception:
                            pass
            for key in obj.keys():
                if _re.search(r"(dataset|_data|data$|_json|json$)", key, _re.I):
                    v = obj.get(key)
                    if isinstance(v, list) and v and all(isinstance(x, dict) for x in v):
                        return (obj, key, v)
                    if isinstance(v, str):
                        try:
                            parsed = json.loads(v)
                            if isinstance(parsed, list) and parsed and all(isinstance(x, dict) for x in parsed):
                                return (obj, key, parsed)
                        except Exception:
                            pass
            for k, v in obj.items():
                result = _find_best_tabular_candidate(v, max_depth=max_depth, depth=depth + 1)
                if result:
                    return result
        if isinstance(obj, list):
            for item in obj:
                result = _find_best_tabular_candidate(item, max_depth=max_depth, depth=depth + 1)
                if result:
                    return result
        return None

    candidate = _find_best_tabular_candidate(structured_data)
    if candidate:
        parent, key, candidate_value = candidate
        parsed_data = None
        if isinstance(candidate_value, list) and candidate_value and all(isinstance(x, dict) for x in candidate_value):
            parsed_data = candidate_value
        elif isinstance(candidate_value, str):
            try:
                parsed = json.loads(candidate_value)
                if isinstance(parsed, list):
                    parsed_data = parsed
                elif isinstance(parsed, dict):
                    parsed_data = [parsed]
            except Exception:
                parsed_data = None
        if parsed_data and isinstance(parsed_data, list):
            # Local import of pandas to avoid import-time dependency for callers who don't need this function
            import pandas as pd

            df = pd.DataFrame(parsed_data)
            temp_file = temp_dir / f"structured_dataset_{timestamp}.csv"
            df.to_csv(temp_file, index=False, encoding="utf-8")
            return str(temp_file)

    # fallback: save entire structured_data as json
    temp_file = temp_dir / f"structured_dataset_{timestamp}.json"
    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(structured_data, f, indent=2)
    return str(temp_file)
