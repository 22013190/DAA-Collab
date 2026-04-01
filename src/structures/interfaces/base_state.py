"""
This module defines the basic structure of state to be passed between LangGraph nodes.
"""

from typing import Annotated

from langgraph.graph.message import add_messages
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass


@dataclass(config=ConfigDict(extra="allow", arbitrary_types_allowed=True))
class BaseState:
    """Basic state schema for passing data between LangGraph nodes with pydantic"""

    # The 'add_messages' func ensures new messages are appended to the list
    # messages: Annotated[list[AnyMessage], add_messages]
    messages: Annotated[list, add_messages]


# class StdBaseState(TypedDict, total=True):
#     """Basic state schema for passing data between LangGraph nodes using stdlib"""
#
#     # The 'add_messages' func ensures new messages are appended to the list
#     messages: Annotated[list, add_messages]
