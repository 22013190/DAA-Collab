"""
Contains functions that may be used by all aspects of the system
"""

from typing import Any


async def format_error_msg(error_path: str, data: Any) -> str:
    return f"""
    File: {error_path}
    Error: {str(data)}
    """
