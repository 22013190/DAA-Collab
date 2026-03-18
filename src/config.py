from typing import Annotated

from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# NOTE: Keep analysis-agent extraction lightweight.
# The database schema types live under the database agent package; in an
# analysis-only repo those modules may not exist. Make this optional so that
# importing `config.py` does not hard-require the database agent.
try:
    from structures.interfaces.database_agent_schema import MCPDatabaseSchema
except Exception:
    class MCPDatabaseSchema(BaseModel):
        name: str = "db"
        CONN_STR: str = ""
        CONTEXT: str = ""


class CommonSettings(BaseSettings):
    model_config = SettingsConfigDict(env_ignore_empty=True)

    ENVIRONMENT: Annotated[str, Field(description="")] = "development"
    DEBUG: Annotated[bool, Field(description="")] = True


class LLMProviderSettings(BaseSettings):
    model_config = SettingsConfigDict(env_ignore_empty=True)

    OPENAI_HOSTED_LLM_URL: Annotated[str, Field(description="")] = (
        "http://127.0.0.1:60000"
    )
    OPENAI_HOSTED_LLM_PORT: Annotated[int, Field(description="")] = 60000

    OPENAI_API_KEY: Annotated[SecretStr, Field(description="")] = SecretStr("somekey")
    OPENAI_MODEL: Annotated[str, Field(description="")] = "somemodel"

    ANTHROPIC_API_KEY: Annotated[SecretStr, Field(description="")] = SecretStr(
        "somekey"
    )
    ANTHROPIC_MODEL: Annotated[str, Field(description="")] = "somemodel"

    GOOGLE_API_KEY: Annotated[SecretStr, Field(description="")] = SecretStr("somekey")
    GOOGLE_MODEL: Annotated[str, Field(description="")] = "somemodel"

    TOKEN_MAX_LIMIT: Annotated[int, Field(description="")] = 4096
    MODEL_NAME: Annotated[str, Field(description="")] = "somemodel"


class LLMLogSettings(BaseSettings):
    model_config = SettingsConfigDict(env_ignore_empty=True)

    LANGFUSE_SECRET_KEY: Annotated[str, Field(description="")] = ""
    LANGFUSE_PUBLIC_KEY: Annotated[str, Field(description="")] = ""
    LANGFUSE_HOST: Annotated[str, Field(description="")] = ""


class DatabaseAgentSettings(BaseSettings):
    """This is for internal system usage,
    This does not serve db agent/mcp purposes"""

    model_config = SettingsConfigDict(env_ignore_empty=True)

    PG_DB_USER: Annotated[str, Field(description="")] = "local"
    PG_DB_PASSWORD: Annotated[str, Field(description="")] = "local"
    PG_DB_HOST: Annotated[str, Field(description="")] = "127.0.0.1"
    PG_DB_PORT: Annotated[int, Field(description="")] = 5432
    PG_DB_NAME: Annotated[str, Field(description="")] = "mcp"


class A2ASettings(BaseSettings):
    model_config = SettingsConfigDict(env_ignore_empty=True)

    DOMAIN_PREFIX: Annotated[str, Field(description="")] = "http://"
    AGENT_DOMAIN: Annotated[str, Field(description="")] = "127.0.0.1"
    A2A_PLANNER_AGENT_PORT: Annotated[int, Field(description="")] = 10000
    A2A_DB_AGENT_PORT: Annotated[int, Field(description="")] = 10001
    A2A_ANALYSIS_AGENT_PORT: Annotated[int, Field(description="")] = 10002


class MCPDatabaseSettings(BaseSettings):
    """This is for db agent/mcp purposes"""

    model_config = SettingsConfigDict(
        env_ignore_empty=True,
        env_prefix="MCP_DB_",
        arbitrary_types_allowed=True,
        # env_nested_delimiter="__",
    )

    PG_CONN_LIST: Annotated[list[MCPDatabaseSchema], Field()] = []
    MYSQL_CONN_LIST: Annotated[list[MCPDatabaseSchema], Field()] = []


common_settings = CommonSettings()
llm_provider_settings = LLMProviderSettings()
llm_log_settings = LLMLogSettings()
a2a_settings = A2ASettings()
mcp_database_settings = MCPDatabaseSettings()
