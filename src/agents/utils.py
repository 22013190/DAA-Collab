"""
Contains functions that may be used by various agents
"""

import json
import math
import sys
import tempfile
from datetime import datetime
from typing import Any, AsyncIterator

import httpx
from a2a.types import DataPart, Part
from config import llm_provider_settings
from langchain_anthropic import ChatAnthropic
from langchain_core.runnables import RunnableWithFallbacks
from langchain_core.runnables.utils import Output
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langgraph.graph.message import (
    BaseMessage,
    BaseMessageChunk,
    message_chunk_to_message,
)
from langgraph.graph.state import CompiledStateGraph, Runnable
from pydantic import ValidationError
from structures.interfaces.a2a_schema import (
    A2ABaseSchema,
    A2AUpdaterArtifactAdapter,
    A2AUpdaterStatusAdapter,
    SchemaValidator,
)
from transformers import AutoTokenizer
from utils import format_error_msg


async def get_remaining_tokens(input_tokens: int) -> int:
    """Get remaining tokens left to be used

    Use with count_tokens() to find size of input tokens and how much
    size left.

    WARNING: this could return negative numbers if input exceeds max
    limit token size!

    See:
    https://github.com/huggingface/tokenizers/issues/875
    https://community.latenode.com/t/token-counting-discrepancy-between-langsmith-and-openai-tokenizer/33087
    """
    token_max_limit = llm_provider_settings.TOKEN_MAX_LIMIT
    return token_max_limit - input_tokens


async def count_tokens(messages: BaseMessage | list[BaseMessage]) -> int:
    """Uses HuggingFace AutoTokenizer to compare token size of given text

    However due to langgraph's abstraction layers, there is about 2-3% of
    token discrepancy. This should mostly be used to calculate input tokens,
    output tokens can be retrieved from the generated AIMessage from the LLM.

    Args:
        messages: LangGraph's BaseMessage or a list of BaseMessage.

    Returns:
        The estimated total tokens in all the supplied messages.
    """
    discrepancy_offset = 1.025
    tokenizer = AutoTokenizer.from_pretrained(llm_provider_settings.MODEL_NAME)
    assert tokenizer is not None, "tokenizer failed to load."

    async def _calculate_content_tool_tokens(message: BaseMessage):
        tokens = 0
        encoded_text = tokenizer.encode(message.content)
        tokens += math.ceil(len(encoded_text) * discrepancy_offset)
        if hasattr(message, "tool_calls"):
            encoded_text = tokenizer.encode(
                json.dumps(getattr(message, "tool_calls"))
            )
            tokens += math.ceil(len(encoded_text) * discrepancy_offset)
        return tokens

    total_tokens = 0
    if isinstance(messages, list):
        for message in messages:
            total_tokens += await _calculate_content_tool_tokens(message)
        return total_tokens

    return await _calculate_content_tool_tokens(messages)


# TO FIX: do not use Any as typing here
def llm_with_fallbacks(
    struct_schema: Any | None = None,
) -> RunnableWithFallbacks:
    """Creates a default chat model instance with fallbacks

    Defaults to OpenAI compatible server connection running on hosted
    LlamaCPP server.

    Fallbacks to Anthropic -> Google.
    """
    if struct_schema:
        llamacpp_openai_llm = ChatOpenAI(
            model=llm_provider_settings.OPENAI_MODEL,
            timeout=None,
            base_url=llm_provider_settings.OPENAI_HOSTED_LLM_URL,
            http_client=httpx.Client(verify=False),  # Is there a better way?
            api_key=llm_provider_settings.OPENAI_API_KEY,
            stream_usage=True,
        ).with_structured_output(struct_schema, method="json_mode")
        anthropic_llm = ChatAnthropic(
            model_name=llm_provider_settings.ANTHROPIC_MODEL,
            api_key=llm_provider_settings.ANTHROPIC_API_KEY,
            timeout=None,
            stop=None,
        ).with_structured_output(struct_schema, method="json_mode")
        google_llm = ChatGoogleGenerativeAI(
            model=llm_provider_settings.GOOGLE_MODEL,
            api_key=llm_provider_settings.GOOGLE_API_KEY,
        ).with_structured_output(struct_schema, method="json_mode")
        llm = llamacpp_openai_llm.with_fallbacks([anthropic_llm, google_llm])
        return llm

    llamacpp_openai_llm = ChatOpenAI(
        model=llm_provider_settings.OPENAI_MODEL,
        timeout=None,
        base_url=llm_provider_settings.OPENAI_HOSTED_LLM_URL,
        http_client=httpx.Client(verify=False),  # Is there a better way?
        api_key=llm_provider_settings.OPENAI_API_KEY,
        stream_usage=True,
    )
    anthropic_llm = ChatAnthropic(
        model_name=llm_provider_settings.ANTHROPIC_MODEL,
        api_key=llm_provider_settings.ANTHROPIC_API_KEY,
        timeout=None,
        stop=None,
    )
    google_llm = ChatGoogleGenerativeAI(
        model=llm_provider_settings.GOOGLE_MODEL,
        api_key=llm_provider_settings.GOOGLE_API_KEY,
    )
    llm = llamacpp_openai_llm

    return llm


async def aquery(
    llm: RunnableWithFallbacks | Runnable,
    messages: str | list[BaseMessage],
    stream: bool = True,
    a2a_task_complete: bool = False,
    a2a_subtask_complete: bool = False,
    a2a_require_user_input: bool = False,
    a2a_allow_interrupt: bool = False,
    a2a_update_content: str | dict = "",
) -> AsyncIterator[Output] | Output | BaseMessage:
    """Query LLM via astream or ainvoke (fallback)

    Make sure tools are embedded with LLM if tools invocation results
    are expected.

    Custom A2A response data object are also added into each Chunk/Result.
    They are placed in Pydantic's model_extra: {}.

    Returns back Output/Message object that was generated from LLM instance.
    """
    if stream:
        output: BaseMessageChunk | None = None
        try:
            async for chunk in llm.astream(messages):
                # Chunk messages subtask should always be False
                chunk.a2a_metadata = A2ABaseSchema(
                    is_task_complete=a2a_task_complete,
                    is_subtask_complete=False,
                    require_user_input=a2a_require_user_input,
                    allow_interrupt=a2a_allow_interrupt,
                    response_parts=a2a_update_content,
                )
                output = chunk if output is None else output + chunk
            if output:
                response = message_chunk_to_message(output)
                response.a2a_metadata = A2ABaseSchema(
                    is_task_complete=a2a_task_complete,
                    is_subtask_complete=a2a_subtask_complete,
                    require_user_input=a2a_require_user_input,
                    allow_interrupt=a2a_allow_interrupt,
                    response_parts=a2a_update_content,
                )
                return response
        except Exception as e:
            err_msg = await format_error_msg(
                __file__,
                f"Error with streaming response from chat model, falling back to ainvoke!: {e}",
            )
            print(err_msg)

    try:
        response = await llm.ainvoke(messages)
        response.a2a_metadata = A2ABaseSchema(
            is_task_complete=a2a_task_complete,
            is_subtask_complete=a2a_subtask_complete,
            require_user_input=a2a_require_user_input,
            allow_interrupt=a2a_allow_interrupt,
            response_parts=a2a_update_content,
        )
        return response
    except Exception as e:
        err_msg = await format_error_msg(
            __file__, f"Error with response from chat model: {e}"
        )
        raise ValueError(err_msg)


async def a2a_default_stream_metadata() -> A2ABaseSchema:
    """Returns a default a2a stream data type"""
    return A2ABaseSchema(
        is_task_complete=False,
        is_subtask_complete=False,
        require_user_input=False,
        allow_interrupt=False,
        response_parts="Generating output tokens...",
    )


async def a2a_default_updater_datapart(
    data: dict[str, Any], metadata: dict[str, Any] | None = None
) -> Part:
    """Returns a default Part object with DataPart for TaskUpdater

    If other schema structure is needed for DataPart or requiring other
    Part types, do not use this function.

    Validations are to be performed before passing into this function.
    """
    return Part(
        DataPart(
            data=data,
            kind="data",
            metadata=metadata,
        )
    )


async def a2a_process_message_status(
    message: BaseMessage | BaseMessageChunk, current_node: str
) -> dict:
    """Process message and format for A2A TaskUpdater update_status()"""

    a2a_metadata = await a2a_default_stream_metadata()
    tool_call_chunks = []
    if "a2a_metadata" in message.model_extra:
        a2a_metadata = message.model_extra.get("a2a_metadata")
    if hasattr(message, "tool_call_chunks"):
        tool_call_chunks = message.tool_call_chunks
    response_data = {
        "agent_task_id": message.id,
        "content": message.content,
        "tool_call_chunks": tool_call_chunks,
        "current_action": a2a_metadata.response_parts,
        "current_node": current_node,
    }
    if not await SchemaValidator(A2AUpdaterStatusAdapter, response_data):
        err_msg = await format_error_msg(
            __file__,
            "Error validating schema for A2A TaskUpdater status",
        )
        print(err_msg)
        raise ValidationError(err_msg)
    return response_data


async def a2a_process_message_artifact(
    message: BaseMessage, current_node: str
) -> dict:
    """Process message and format for A2A TaskUpdater add_artifact()"""

    a2a_metadata = await a2a_default_stream_metadata()
    tool_calls = []
    if "a2a_metadata" in message.model_extra:
        a2a_metadata = message.model_extra.get("a2a_metadata")
    if hasattr(message, "tool_calls"):
        tool_calls = message.tool_calls
    response_data = {
        "agent_task_id": message.id,
        "content": message.content,
        "tool_calls": tool_calls,
        "current_action": a2a_metadata.response_parts,
        "current_node": current_node,
    }
    if not await SchemaValidator(A2AUpdaterArtifactAdapter, response_data):
        err_msg = await format_error_msg(
            __file__,
            "Error validating schema for A2A TaskUpdater artifact",
        )
        print(err_msg)
        raise ValidationError(err_msg)
    return response_data


async def current_posix_time() -> float:
    """Returns current time in unix epoch timestamp"""
    return datetime.now().timestamp()

async def graph_visual(graph: CompiledStateGraph, *, xray: bool = True) -> bool:
    """
    Visualise an image of the given compiled graph.

    Args:
        graph: A compiled LangGraph State Graph

    Returns:
        A boolean if image was succesfully created.
    """
    try:
        from PIL import Image
        # Only works in jupyter notebook
        # display(Image(graph.get_graph().draw_mermaid()))
    except Exception as e:
        err_msg = await format_error_msg(
            __file__, f"Exception error at agents/utils: {e}"
        )
        print(err_msg)
        return False

    # LangGraph can hide conditional edges unless xray mode is enabled.
    # xray=True renders all potential branches, which is usually what you want
    # when validating routing logic.
    try:
        g = graph.get_graph(xray=xray)
    except TypeError:
        # Back-compat for older LangGraph versions
        g = graph.get_graph()
    graph_img = g.draw_mermaid_png()
    tmp_filepath = tempfile.gettempdir() + "/graph.png"
    if sys.platform == "win32" or sys.platform == "cygwin":
        tmp_filepath = tempfile.gettempdir() + "\\graph.png"
    with open(tmp_filepath, "wb") as file:
        file.write(graph_img)
    img = Image.open(tmp_filepath)
    img.show()
    return True
