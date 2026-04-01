from typing import Annotated, TypeAlias
from uuid import uuid4

from agents.database.mschema.m_schema import MSchema
from agents.database.mschema.schema_engine import SchemaEngine
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    TypeAdapter,
    computed_field,
)
from pydantic.dataclasses import dataclass
from sqlalchemy import Engine
from sqlalchemy.ext.asyncio import AsyncEngine


# For env config
class MCPDatabaseSchema(BaseModel):
    _id: str = PrivateAttr(default_factory=lambda: "db_" + str(uuid4()))

    @computed_field
    @property
    def id(self) -> str:
        return self._id

    name: Annotated[
        str,
        Field(
            description="""Identifier for the given connection string and context

            Avoid leaving this empty or giving duplicate names.

            TO FIX!: Currently no checks for unique names yet but created id with
            uuid4 as guarantor.
            """
        ),
    ] = "db"

    CONN_STR: Annotated[
        str,
        Field(
            description="""
            A full connection string to the destinated database.

            Postgres example:
            postgresql+psycopg://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}

            MySQL example:
            """
        ),
    ] = ""
    CONTEXT: Annotated[
        str,
        Field(
            description="""
            This is to give context about the
            database that will be connected to.
            """
        ),
    ] = ""

    # TO FIX
    # @model_validator(mode="after")
    # def validate_db_prefix(self) -> Self:
    #     if self.CONN_STR.startswith("postgresql+psycopg://"):
    #         return self
    #     raise ValueError()


# For state property storage
@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class MSchemaTimestamp:
    """
    Uses MSchema object from database agent to define the current mschema contents.

    ref_timestamp is a POSIX compliant epoch timestamp in float value. Use this to
    compare with current timestamp to determine when to refresh schema data again.
    Update value if schema data is to be invocated/refresh.
    Implementation to be handled within the respective agents requiring schema data.

    """

    mschema_engine: SchemaEngine
    mschema: MSchema
    ref_timestamp: float


# Deprecated, to remove in the future
@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class SchemaEngines:
    pgsql: MSchemaTimestamp
    # mysql: SchemaTimestamp
    # oracle: SchemaTimestamp
    # cql: SchemaTimestamp


@dataclass(config=ConfigDict(extra="allow", arbitrary_types_allowed=True))
class EngineContextSchema:
    async_engine: AsyncEngine  # For general use of MCP
    engine: Engine  # Only for MSchema use as async not supported yet
    context: str
    mschema_timestamp: MSchemaTimestamp | None = None


EngineContextAdapter = TypeAdapter(EngineContextSchema)

MCPEngineStore: TypeAlias = dict[str, EngineContextSchema]
MCPEngineStoreAdapter = TypeAdapter(MCPEngineStore)


# For structuring output of LLM
# Not working with ChatOpenAI for pure invocation
class DBClassifierSchema(BaseModel):
    """Structure for database classification after user's initial query"""

    db_id: str = Field(description="The database id")
    dialect: str = Field(description="The database dialect name")
