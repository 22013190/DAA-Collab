from typing import NamedTuple, Protocol

from fastmcp import Client as MCPClient
from langchain_core.tools import BaseTool
from mcp.types import Tool
from pydantic.config import ConfigDict
from pydantic.dataclasses import dataclass


class ToolInterface(Protocol):
    """
    Defines the necessary methods that needs to be interfaced by various
    tool classes.
    """

    def check_tool_tags(self):
        pass


class ToolStatusType(NamedTuple):
    """
    Defines the type hint for a custom tuple of status after checking
    of tool call tags in a message.

    Indexes are a tuple of 2 integers with the ends of the tags.
    """

    status: bool
    start_tag_indexes: tuple[int, int] | None
    end_tag_indexes: tuple[int, int] | None


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class MCPTools:
    """
    Defines the attribute for agents that requires utilisation of tools.
    This works specifically for FastMCP related objects only.

    Conversions will be required to make tools compatible with other frameworks.
    E.g. FastMCP (mcp.Tool()) -> LangGraph's (StructuredTool())
    """

    # pgsql: SchemaTimestamp
    # mysql: SchemaTimestamp
    # oracle: SchemaTimestamp
    # cql: SchemaTimestamp
    mcp_client: MCPClient
    # Default FastMCP tools type
    tools: list[Tool]
    # Support for LangGraph
    langgraph_tools: list[BaseTool]
