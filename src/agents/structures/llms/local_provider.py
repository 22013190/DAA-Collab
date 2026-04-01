"""
This is a model class for subsequent locally hosted LLMs.

IMPORTANT: Not to be confused for this class to work only with local
models running on your current hardware. Models can be accessed remotely
via their respective functions. This is targeted for all/most LLMs that
you are self-hosting or hosted remotely that you can control.

Currently supported Local LLM Integrations:
- Llama.cpp
- LangGraph / LangChain

Currently supported Remote LLM servers:
- OpenAI (Llama.cpp can make expose compatible servers)

"""

from abc import ABC, abstractmethod

from langchain_core.messages import AIMessage
from pydantic import BaseModel, ConfigDict
from structures.interfaces.tool import ToolStatusType


class LocalProvider(ABC, BaseModel):
    """Specifies the required methods and properties necessary for Local LLM models.

    Provide abstract methods for different LLM model classes to follow to conform
    tool calling and unpredictable formats to be used with MCP context.

    LLM Creation are to be instantiated on respective agent itself.

    OpenAI package is used to support self hosted Llama CPP LLM server. This makes use of
    `llama-server` which exposes compatible OpenAI structure of API calling over http.

    NOTE: Actual OpenAI cloud models can be called but avoid putting actual openai api keys
    unless there is a need to utilise their services.

    Refer to: https://python.langchain.com/api_reference/openai/chat_models/langchain_openai.chat_models.base.ChatOpenAI.html

    To consider switching to dataclass if BaseModel is heavy.
    https://stackoverflow.com/questions/62011741/pydantic-dataclass-vs-basemodel
    """

    # Pydantic configs for this model
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="allow",
    )

    @abstractmethod
    def check_tool_tags(self) -> ToolStatusType:
        """
        Check if tool call tags are present and return true with start and
        end index.

        If no tags present, return false with None values as index.
        """
        pass

    @abstractmethod
    def format_tool_message(self) -> AIMessage | str:
        pass
