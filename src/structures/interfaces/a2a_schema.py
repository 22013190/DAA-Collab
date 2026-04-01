"""Handles langgraph astream output for A2A updates

Messages: Yields each token from the CompiledStateGraph instance resulting in
AIMessageChunk() or BaseMessageChunk().
Does not hold previous output.
The data type is a dict.

Updates: Returns the latest output in a dict {"node": {"graph_schema_property": [...]}}
    The 'node' is the function name and will change. Does not hold previous output.
    The 'property' refers to the state schema defined for the particular Compiled LangGraph
    agent. There can be more than 1 property depending on the schema used. Always validate
    the structure to ensure correctness of the resulting output.
Does not hold previous output.
The data type is a dict.

Values: Returns all accumulated output in a dict { "graph_schema_property": [BaseMessage] }.
    The 'property' refers to the state schema defined for the particular Compiled LangGraph
    agent. There can be more than 1 property depending on the schema used. Always validate
    the structure to ensure correctness of the resulting output.
Holds previous output from the start.
The data type is a tuple.
"""

from typing import Any, TypedDict

from pydantic import ConfigDict, TypeAdapter, ValidationError
from pydantic.dataclasses import dataclass

### General purpose schemas for most/all agents


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class A2ABaseSchema:
    """Additional information to be embedded with LangGraph BaseMessage

    These fields dictacts the response format for A2A messages to be
    sent back to the client via Parts and Artifacts.
    """

    # This is for the overall task of the agent
    is_task_complete: bool
    """
    This is used for individual node functions within an agent.

    The last message chunk should be true.
    """
    is_subtask_complete: bool
    # For human in the loop purposes
    require_user_input: bool
    allow_interrupt: bool
    # Message for A2A Parts from server -> client
    response_parts: str | dict[str, Any]


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class A2AMetadataSchema(TypedDict):
    """Wrapper to ensure consistent key format when appending to Messages"""

    a2a_metadata: A2ABaseSchema


A2AMetadataAdapter = TypeAdapter(A2AMetadataSchema)


class A2AUpdaterStatusSchema(TypedDict):
    """Generic schema for A2A TaskUpdater Parts returning to client"""

    agent_task_id: str
    content: str | list
    tool_call_chunks: list[Any]
    current_action: str | dict | list
    current_node: str


A2AUpdaterStatusAdapter = TypeAdapter(A2AUpdaterStatusSchema)


class A2AUpdaterArtifactSchema(TypedDict):
    """Generic schema for A2A TaskUpdater Artifact returning to client

    Uses DataPart as default data type.
    """

    agent_task_id: str
    content: str | list
    tool_calls: list[dict]
    current_action: str | dict | list
    current_node: str


A2AUpdaterArtifactAdapter = TypeAdapter(A2AUpdaterArtifactSchema)

# # Consider moving this to generic LangGraph schema without A2A
# class A2ALanggraphStreamUpdateSchema(TypedDict):
#     """For A2A use to check on Compiled Graph Stream Updates type
#
#     This only checks the inner object value. Only support LangGraph
#     stream or astream methods with multiple modes
#     E.g. stream_mode = [ "values", "updates"]
#
#     Format: { "node_name": { "messages": [BaseMessage] } }
#     The node_name changes name as execution traverse the graph's nodes thus
#     it is not constant. The inner value will also change depending on how the
#     LangGraph schema is formatted to be passed between nodes so make sure to
#     use consistent schema format.
#     """
#
#     messages: list[BaseMessage]
#
#
# A2ALanggraphStreamUpdateAdapter = TypeAdapter(A2ALanggraphStreamUpdateSchema)

### Utilities for schemas


# TO REFORMAT ERROR MESSAGE
async def SchemaValidator(schema_class: TypeAdapter, input: dict) -> bool:
    """Basic validator to compare input against custom schema

    Return boolean based on validation results.
    """
    try:
        schema_class.validate_python(input)
        return True
    except ValidationError as e:
        print(e)
        return False
    except Exception:
        print(Exception)
        raise Exception
