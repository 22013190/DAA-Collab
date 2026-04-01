"""
Utilities file for various reusable functions for MCP
"""

import datetime
import decimal
import json
import sys
from gc import get_referents
from types import FunctionType, ModuleType
from typing import Any, List

from fastmcp import Client as MCPClient
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.tools import convert_mcp_tool_to_langchain_tool
from packages.simple_py_logger.src.logger import Logger
from sqlalchemy.engine.cursor import CursorResult
from structures.interfaces.tool import MCPTools

parent_logger = Logger("MCP_Utils")
logger = parent_logger.get_current_logger()


async def exceed_size_limit(data: Any) -> bool:
    # https://github.com/modelcontextprotocol/python-sdk/issues/1012
    # https://github.com/modelcontextprotocol/python-sdk/blob/679b22970e12b8eec3897108ac7d6b0624809b9e/src/mcp/server/streamable_http.py#L50
    # While STDIO has no limit, memory leaks happen when data size is too
    # large therefore, keep size limit to 4MB

    MAX_SIZE = 4 * 1024 * 1024  # 4MB

    # Adapted from: https://stackoverflow.com/a/30316760

    # Custom objects know their class.
    # Function objects seem to know way too much, including modules.
    # Exclude modules as well.
    BLACKLIST = type, ModuleType, FunctionType

    def getsize(obj):
        """Sum size of object & members."""
        if isinstance(obj, BLACKLIST):
            raise TypeError(
                "Getsize() does not take argument of type: " + str(type(obj))
            )
        seen_ids = set()
        size = 0
        objects = [obj]
        while objects:
            need_referents = []
            for obj in objects:
                if not isinstance(obj, BLACKLIST) and id(obj) not in seen_ids:
                    seen_ids.add(id(obj))
                    size += sys.getsizeof(obj)
                    need_referents.append(obj)
            objects = get_referents(*need_referents)
        return size

    if getsize(data) > MAX_SIZE:
        return True

    return False


async def create_mcp_client(configs: list[dict]) -> MCPClient:
    """Creates a FastMCP Client instance

    Make sure to close() the client and delete the instance after use to help
    free up the resource.

    Potential issues/bugs:
    https://github.com/modelcontextprotocol/python-sdk/issues/262

    Args:
        Variable arg of mcpconfigs in dict types.

    Returns:
        A fastmcp client instance.
    """
    # Always create a new object to avoid overrides in reference
    # mcp_configs = dict(base_config)
    mcp_configs = {"mcpServers": {}}
    if configs is not None:
        for config in configs:
            mcp_configs["mcpServers"].update(config)

    # Create client
    client = MCPClient(mcp_configs)
    return client


async def get_mcp_tools(client: MCPClient) -> MCPTools | None:
    """Get MCPTools in FastMCP and LangGraph formats

    The process should be to obtain MCPTools in respective formats and bind
    with the LLM accordingly. In nested event loops / threads, function calling
    may not work and require a new MCPClient instance.

    Original idea was to initialise a client once and let it run the entire
    lifecycle. But this led to event loop issues as coroutines are tied to
    parent event loop which does not execute unless the right event loop is set.
    Curr

    Unsure how this will impact I/O due to stale thread and when garbage
    collection will occur.

    https://gofastmcp.com/clients/client#connection-lifecycle

    MCP session is tied with event_loop, need find a way to reuse parent
    event_loop or reinitialise MCP session.

    https://github.com/jlowin/fastmcp/issues/166
    https://github.com/modelcontextprotocol/python-sdk/issues/262
    """
    # Currently only supports STDIO transport for security reasons
    try:
        async with client as c:
            tools = await c.list_tools()
            langgraph_tools: list[BaseTool] = []
            for tool in tools:
                converted_tool = convert_mcp_tool_to_langchain_tool(
                    c.session, tool
                )
                langgraph_tools.append(converted_tool)
        session_tools = MCPTools(client, tools, langgraph_tools)
        return session_tools
    except Exception as e:
        err_msg = f"""
        File: {__file__},
        Error: Exception as occurred when connecting to MCP Server: {e}
        """
        logger.error(err_msg)


# https://stackoverflow.com/questions/11875770/how-can-i-overcome-datetime-datetime-not-json-serializable?page=1&tab=scoredesc#tab-top
# https://stackoverflow.com/questions/65309377/typeerror-object-of-type-decimal-is-not-json-serializable
def _json_serial(obj):
    """Handle data types that cannot be directly converted using json library"""
    if isinstance(obj, datetime.datetime):
        return obj.isoformat()
    if isinstance(obj, decimal.Decimal):
        return float(obj)
    raise TypeError(f"Type {type(obj)} is not serializable.")


async def to_json_list(result: CursorResult) -> List[dict]:
    """Converts query results into valid JSON object

    Args:
        result: A valid CursorResult instance from successful SQL query

    Returns:
        A list of dictionary objects that are JSON-valid
    """
    cols = result.keys()
    data = [dict(zip(cols, row)) for row in result.fetchall()]
    for row in data:
        # Iterate keys within each row
        for key in row:
            row[key] = json.dumps(row[key], default=_json_serial)
    return data
