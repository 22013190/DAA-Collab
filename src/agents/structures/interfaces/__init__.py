from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from structures.interfaces.tool import ToolInterface, ToolStatusType

__all__ = ("ToolInterface", "ToolStatusType")
