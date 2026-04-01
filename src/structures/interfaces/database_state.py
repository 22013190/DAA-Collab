"""
This module defines the basic structure of state to be passed between LangGraph nodes.
"""

from typing import Annotated

from pydantic import ConfigDict, Field, TypeAdapter
from pydantic.dataclasses import dataclass
from structures.interfaces.a2a_schema import A2ABaseSchema
from structures.interfaces.base_state import BaseState

###############################################################################
# LangGraph


@dataclass(config=ConfigDict(extra="allow", arbitrary_types_allowed=True))
class DatabaseOutputState:
    """Final output state schema for passing data back"""

    db_data: Annotated[
        list,
        Field(
            description="""
            This is a list of strings representing the database data.

            Each element should be a valid JSON-string to allow easy conversion.
            """,
            default_factory=list,
        ),
    ]

    conclusion: Annotated[
        str,
        Field(
            description="""
            A small summary of the data obtained.

            The details are a correlation between user's query and obtained data.
            """,
            default_factory=str,
        ),
    ]

    schema: Annotated[
        str | None,
        Field(
            description="""
            This contains the schema table(s) that were used to aid in SQL query
            generation to achieve the results for user's query objectives.
            """,
            default_factory=str,
        ),
    ]

    # ERROR: cannot give this default value as it crashes A2A server instance leaving
    # it in terminal state
    # TO FIX!
    a2a_metadata: Annotated[
        A2ABaseSchema | None,
        Field(
            description="""
            A small summary of the data obtained.

            The details are a correlation between user's query and obtained data.
            """,
            # default=None,
            # default=A2ABaseSchema(
            #     is_task_complete=False,
            #     is_subtask_complete=False,
            #     require_user_input=False,
            #     allow_interrupt=False,
            #     response_parts="",
            # ),
        ),
    ]


DatabaseOutputAdapter = TypeAdapter(DatabaseOutputState)


@dataclass(config=ConfigDict(extra="allow"))
class DatabaseInternalState(BaseState):
    """Generic input state data between nodes into and within the Database Agent"""

    # Additional properties to be inserted here

    pass


DatabaseInternalAdapter = TypeAdapter(DatabaseInternalState)


@dataclass(config=ConfigDict(extra="allow"))
class DatabasePostClassificationState(BaseState):
    # Additional properties to be inserted here
    mschema_tables: Annotated[
        str | None,
        Field(
            description="This data should have been passed into the LLM for SQL tool call and is derived from to_mschema()"
        ),
    ]


DatabasePostClassificationAdapter = TypeAdapter(DatabasePostClassificationState)


@dataclass(config=ConfigDict(extra="allow", arbitrary_types_allowed=True))
class DatabaseOverallState(
    DatabaseOutputState, DatabaseInternalState, DatabasePostClassificationState
):
    """Combination of both Input and Output State

    Avoid adding properties here. Create new states and
    let overall state inherit it.
    """

    pass


DatabaseOverallAdapter = TypeAdapter(DatabaseOverallState)


###############################################################################
# A2A


@dataclass(config=ConfigDict(extra="allow", arbitrary_types_allowed=True))
class DatabaseInputState:
    """For A2A Clients to structure message for this agent

    Do not inherit from
    """

    message: Annotated[
        str,
        Field(
            description="""
            A text message of the user query (mostly derived from agent).

            Do not provide SQL statements in this text message.
            """,
        ),
    ]


DatabaseInputAdapter = TypeAdapter(DatabaseInputState)
