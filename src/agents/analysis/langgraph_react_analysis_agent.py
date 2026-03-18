"""
LangGraph React Analysis Agent with langchain_mcp_adapters
=========================================================

A React agent for quantitative analysis using:

- LangGraph's create_react_agent for React pattern
- langchain_mcp_adapters for MCP integration
- Anthropic Claude (swappable LLM)
- A2A (Agent-to-Agent) communication protocol

Key Features:
- React agent pattern with step-by-step reasoning
- Dynamic FastMCP tool discovery via langchain_mcp_adapters
- LLM flexibility (Anthropic, OpenAI, etc.)
- A2A protocol for orchestrator communication
- Session persistence and memory
"""

"""
CODING STANDARDS FOR THIS FILE:
================================
1. NEVER USE UNICODE EMOJIS OR SPECIAL CHARACTERS
   - Windows console (cp1252 codec) cannot display Unicode emojis
   - This causes UnicodeEncodeError and server crashes
   - All emojis have been completely removed from this file

2. STICK TO ASCII TEXT ONLY:
   - Use plain text descriptions instead of emojis
   - Use status indicators like "OK", "ERROR", "WARNING"
   - Use ASCII symbols if needed: *, +, -, =, |, etc.

3. ALWAYS TEST ON WINDOWS CONSOLE BEFORE DEPLOYMENT
================================
"""




import asyncio
import base64
import json
import re
import shutil
import traceback

import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TypedDict

import pandas as pd
import httpx

# Official A2A SDK imports
from a2a.types import (
    Message,
    SendMessageRequest,
    FilePart,
    FileWithBytes,
    FileWithUri,
)
from a2a.utils.message import get_text_parts, get_file_parts, get_data_parts

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# MCP Integration
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langfuse import Langfuse, get_client

# Langfuse integration for tracing
from langfuse.langchain import CallbackHandler

# LangGraph and LangChain imports
from langgraph.prebuilt import create_react_agent
from mcp_servers.config import (
    analysis_server_config,
    data_process_server_config,
)

# Configuration and utilities
from src.agents.analysis.analysis_config import AgentConfig
from config import llm_provider_settings
# Logging setup
from packages.simple_py_logger.src.logger import Logger

parent_logger = Logger("Analysis Agent")
logger = parent_logger.get_current_logger()


# Workflow Progress Tracker for AI-driven execution
class WorkflowProgressTracker:
    """Minimal progress tracker used in single-run executor.

    Purpose: keep a tiny record of executed tools and provide a simple completion check.
    This replaces the previous heavy enforcement tracker when iterative workflow is removed.
    """

    def __init__(self, planned_tools: List[str]):
        self.planned_tools = planned_tools or []
        self.executed_tools: List[str] = []

    def record_tool_execution(self, tool_name: str, success: bool = True):
        if tool_name and tool_name not in self.executed_tools:
            self.executed_tools.append(tool_name)

    def get_next_recommended_tools(self, max_suggestions: int = 3) -> List[str]:
        return [t for t in self.planned_tools if t not in self.executed_tools][:max_suggestions]

    def is_workflow_complete(self) -> Tuple[bool, str]:
        if not self.planned_tools:
            return True, "No planned tools"
        if set(self.planned_tools).issubset(set(self.executed_tools)):
            return True, "All planned tools executed"
        return False, f"Executed {len(self.executed_tools)}/{len(self.planned_tools)} tools"


# Rate Limiter for API calls
class APIRateLimiter:
    """Rate limiter to prevent API quota exhaustion"""

    def __init__(self, requests_per_minute: int = 50, requests_per_day: int = 1000):
        """
        Initialize rate limiter with analysis-optimized limits

        Args:
            requests_per_minute: Max requests per minute (default 50, good for analysis workloads)
            requests_per_day: Max requests per day (default 1000, supports multiple full analyses)
        """
        self.rpm_limit = requests_per_minute
        self.rpd_limit = requests_per_day

        # Track requests in the last minute
        self.minute_requests = deque()

        # Track requests in the current day
        self.day_requests = 0
        self.day_reset_time = datetime.now().replace(
            hour=0, minute=0, second=0, microsecond=0
        ) + timedelta(days=1)

        logger.info(
            f"Rate limiter initialized: {self.rpm_limit} RPM, {self.rpd_limit} RPD"
        )

    async def wait_if_needed(self):
        """Wait if necessary to respect rate limits"""
        now = datetime.now()

        # Check daily limit reset
        if now >= self.day_reset_time:
            self.day_requests = 0
            self.day_reset_time = now.replace(
                hour=0, minute=0, second=0, microsecond=0
            ) + timedelta(days=1)
            logger.info("Daily rate limit reset")

        # Check daily limit
        if self.day_requests >= self.rpd_limit:
            wait_until = self.day_reset_time
            wait_seconds = (wait_until - now).total_seconds()
            logger.warning(
                f"Daily rate limit reached. Waiting {wait_seconds:.0f} seconds until reset."
            )
            await asyncio.sleep(wait_seconds)
            return

        # Clean old minute requests
        minute_ago = now - timedelta(minutes=1)
        while self.minute_requests and self.minute_requests[0] < minute_ago:
            self.minute_requests.popleft()

        # Check minute limit
        if len(self.minute_requests) >= self.rpm_limit:
            # Wait until the oldest request is more than a minute old
            oldest_request = self.minute_requests[0]
            wait_until = oldest_request + timedelta(minutes=1)
            wait_seconds = (wait_until - now).total_seconds() + 1  # Add 1 second buffer

            if wait_seconds > 0:
                logger.info(
                    f"Rate limit: waiting {wait_seconds:.1f} seconds (RPM: {len(self.minute_requests)}/{self.rpm_limit})"
                )
                await asyncio.sleep(wait_seconds)

    def record_request(self):
        """Record that a request was made"""
        now = datetime.now()
        self.minute_requests.append(now)
        self.day_requests += 1

        logger.debug(
            f"Request recorded. RPM: {len(self.minute_requests)}/{self.rpm_limit}, RPD: {self.day_requests}/{self.rpd_limit}"
        )


@dataclass
class WorkflowPlan:
    """Intelligent workflow planning for analysis tasks"""

    instruction: str
    dataset_info: Dict[str, Any]
    suggested_tools: List[str]
    execution_order: List[Dict[str, Any]]
    estimated_duration: int  # in seconds
    complexity_score: float  # 0.0 to 1.0
    requirements: List[str]
    expected_outputs: List[str]


@dataclass
class PerformanceMetrics:
    """Track agent performance and optimize execution"""

    total_analyses: int = 0
    successful_analyses: int = 0
    average_execution_time: float = 0.0
    tool_usage_stats: Dict[str, int] = field(default_factory=dict)
    error_patterns: Dict[str, int] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=datetime.now)


class AgentState(TypedDict):
    """Enhanced state for the analysis agent session"""

    messages: List[Any]
    session_id: str
    dataset_path: Optional[str]
    dataset_info: Optional[Dict[str, Any]]  # Cached dataset metadata
    current_instruction: Optional[str]
    workflow_plan: Optional[WorkflowPlan]
    analysis_results: List[Dict[str, Any]]
    orchestrator_feedback: Optional[str]
    report_ready: bool
    performance_metrics: PerformanceMetrics
    context_memory: Dict[str, Any]  # Long-term context storage
    active_tools: List[str]  # Currently loaded tools
    execution_history: List[Dict[str, Any]]  # Detailed execution log


class LangGraphReactAnalysisAgent:
    """
    LangGraph React Analysis Agent with Dynamic Workflow Planning

    Features:
    - Dynamic workflow planning based on orchestrator instructions
    - Intelligent tool selection from available MCP servers
    - A2A protocol for orchestrator communication
    - React agent pattern with step-by-step reasoning
    - Session persistence and memory
    """

    def __init__(self, config_path: Optional[str] = None):
        self.config = (
            AgentConfig(_env_file=config_path) if config_path else AgentConfig()
        )
        self.agent_id = f"analysis_agent_{uuid.uuid4().hex[:8]}"
        self.mcp_client = None
        self.tools = []
        self.react_agent = None
        self.current_state: Optional[AgentState] = None
        self._shutdown_requested = False  # Flag to handle graceful shutdown

        # Initialize rate limiter for API calls (provider-aware limits)
        # Respects ENABLE_RATE_LIMITING environment variable via config
        enable_rate_limiting = getattr(self.config, "enable_rate_limiting", True)  # Respect config/env with safe default
        if enable_rate_limiting:
            # Get provider-specific rate limits
            rpm, rpd = self._get_provider_rate_limits()
            self.rate_limiter = APIRateLimiter(
                requests_per_minute=rpm, requests_per_day=rpd
            )
            logger.info(
                f"Rate limiting enabled for {getattr(self.config, 'llm_provider', 'unknown')} provider: {rpm} RPM, {rpd} RPD"
            )
        else:
            self.rate_limiter = None
            logger.info("Rate limiting disabled for maximum performance")

        # Initialize Langfuse for tracing
        self.langfuse_handler = None
        self._initialize_langfuse()

        # Initialize LLM
        self.llm = self._initialize_llm()

        logger.info(f"Initialized LangGraph React Analysis Agent: {self.agent_id}")

    async def __aenter__(self):
        """Async context manager entry"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with cleanup"""
        await self.cleanup()
        return False  # Don't suppress exceptions

    def _initialize_langfuse(self):
        """Initialize Langfuse client and callback handler for tracing"""
        try:
            # Check if Langfuse credentials are available
            if self.config.langfuse_secret_key and self.config.langfuse_public_key:
                # Initialize Langfuse client with constructor arguments
                Langfuse(
                    secret_key=self.config.langfuse_secret_key,
                    public_key=self.config.langfuse_public_key,
                    host=self.config.langfuse_host,
                )

                # Initialize the Langfuse callback handler
                self.langfuse_handler = CallbackHandler()

                logger.info("Langfuse tracing initialized successfully")
            else:
                logger.warning("Langfuse credentials not found - tracing disabled")
                self.langfuse_handler = None
        except Exception as e:
            logger.warning(f"Failed to initialize Langfuse: {e}")
            self.langfuse_handler = None

    def _add_langfuse_trace_metadata(
        self,
        session_id: str,
        instruction: str,
        analysis_type: str = "data_analysis",
    ) -> Dict[str, Any]:
        """Add metadata for Langfuse traces via enclosing span"""
        metadata = {}

        if self.langfuse_handler:
            try:
                langfuse_client = get_client()
                if langfuse_client:
                    # Use metadata fields approach for dynamic trace attributes
                    metadata = {
                        "langfuse_session_id": session_id,
                        "langfuse_user_id": self.agent_id,
                        "langfuse_tags": [
                            analysis_type,
                            "langgraph",
                            "react_agent",
                            "mcp",
                        ],
                        "langfuse_release": "v1.0.0",
                        "langfuse_metadata": {
                            "agent_id": self.agent_id,
                            "instruction_preview": instruction[:100] + "..."
                            if len(instruction) > 100
                            else instruction,
                            "analysis_type": analysis_type,
                            "mcp_tools_count": len(self.tools),
                        },
                    }
                    logger.debug(f"Added Langfuse metadata for session {session_id}")
            except Exception as e:
                logger.warning(f"Failed to add Langfuse metadata: {e}")

        return metadata

    def _get_provider_rate_limits(self) -> tuple[int, int]:
        """Get rate limits based on LLM provider and user configuration"""
        # Check if user has explicitly set rate limits
        user_rpm = getattr(self.config, "rate_limit_rpm", None)
        user_rpd = getattr(self.config, "rate_limit_rpd", None)

        # If user set explicit limits, use those
        if user_rpm is not None and user_rpd is not None:
            return user_rpm, user_rpd

        # Otherwise, use provider-specific defaults
        provider = getattr(self.config, "llm_provider", "anthropic").lower()

        if provider == "gemini":
            # Gemini Free Tier limits by model: 
            # Gemini 2.0 Flash-Lite: 30 RPM, 200 RPD (BEST for free tier!)
            # Gemini 2.0 Flash: 15 RPM, 200 RPD 
            # Gemini 2.5 Flash: 10 RPM, 250 RPD 
            # Use very conservative limits to avoid quota exhaustion
            return 10, 120  # Very conservative: 10 RPM, 120 RPD to stay well under limits        elif provider == "anthropic":
            # Anthropic token rate limits can be more restrictive than request limits
            # Conservative settings to avoid input token rate limits  
            return 40, 2000  # Reduced from 60, 3000 to handle token rate limits

        elif provider == "openai":
            # OpenAI has moderate limits
            # Conservative but not as restrictive as Gemini
            return 30, 1000

        else:
            # Unknown provider - use conservative Gemini limits for safety
            return 8, 150

    def _handle_errors(self, func: str, error: Exception) -> Dict[str, Any]:
        """Simple error handling with logging and clean responses"""
        error_msg = f"Error in {func}: {str(error)}"
        logger.error(error_msg, exc_info=True)

        return {
            "success": False,
            "error": error_msg,
            "error_type": type(error).__name__,
            "function": func,
        }

    def _initialize_llm(self):
        """Initialize LLM with flexible provider support"""
        llm_provider = getattr(self.config, "llm_provider", "anthropic")

        if llm_provider == "anthropic":
            return ChatAnthropic(
                model_name=getattr(
                    self.config, "anthropic_model", "claude-3-5-haiku-20241022"
                ),
                temperature=getattr(self.config, "temperature", 0.1),
                max_tokens_to_sample=getattr(self.config, "max_output_tokens", 4000),
                api_key=getattr(self.config, "anthropic_api_key"),
                timeout=None,
                stop=None,
            )
        elif llm_provider == "openai":
            return ChatOpenAI(
                model=llm_provider_settings.OPENAI_MODEL,
                timeout=None,
                base_url=llm_provider_settings.OPENAI_HOSTED_LLM_URL,
                http_client=httpx.Client(verify=False), 
                api_key=llm_provider_settings.OPENAI_API_KEY,
            )
        elif llm_provider == "gemini":
            return ChatGoogleGenerativeAI(
                model=getattr(self.config, "gemini_model", "gemini-2.0-flash-lite"),
                temperature=getattr(self.config, "temperature", 0.1),
                max_output_tokens=getattr(self.config, "max_output_tokens", 4000),
                google_api_key=getattr(self.config, "gemini_api_key", None),
            )
        else:
            # Default to Anthropic
            return ChatAnthropic(
                model_name="claude-3-5-haiku-20241022",
                temperature=0.1,
                max_tokens_to_sample=4000,
                timeout=None,
                stop=None,
            )

    async def _rate_limited_llm_call(
        self, messages: List, operation_name: str = "LLM call"
    ):
        """Make a rate-limited LLM call with Langfuse tracing"""
        if self.rate_limiter:
            await self.rate_limiter.wait_if_needed()

        # Additional delay for provider-specific rate limits
        provider = getattr(self.config, "llm_provider", "anthropic").lower()
        if provider == "anthropic":
            await asyncio.sleep(10.0)  # 10 second delay between requests to handle strict token acceleration limits
        elif provider == "gemini":
            await asyncio.sleep(6.0)  # 6 second delay for Gemini free tier to avoid quota exhaustion
        
        try:
            logger.debug(f"Making {operation_name}")

            # Add Langfuse tracing if available with better metadata
            config = {}
            if self.langfuse_handler:
                config["callbacks"] = [self.langfuse_handler]
                config["run_name"] = operation_name
                config["tags"] = [
                    "llm_call",
                    operation_name.lower().replace(" ", "_"),
                ]
                config["metadata"] = {
                    "agent_id": self.agent_id,
                    "operation": operation_name,
                    "provider": getattr(self.config, "llm_provider", "unknown"),
                    "model": getattr(
                        self.config,
                        f"{getattr(self.config, 'llm_provider', 'anthropic')}_model",
                        "unknown",
                    ),
                    "message_count": len(messages),
                }

            response = await self.llm.ainvoke(messages, config=config)
            logger.debug(f"CHECK WHICH PROVIDER: {self.llm}")

            if self.rate_limiter:
                self.rate_limiter.record_request()
            return response
        except Exception as e:
            logger.error(f"Error in {operation_name}: {e}")
            # Still record the request attempt for rate limiting
            if self.rate_limiter:
                self.rate_limiter.record_request()
            raise

    async def initialize_mcp_tools(self):
        """Initialize MCP tools using langchain_mcp_adapters - simplified without data processing server"""
        try:
            # Get FastMCP server configurations
            fastmcp_server_path = getattr(
                self.config, "fastmcp_server_path", "./fastmcp_server.py"
            )
            abs_server_path = str(Path(fastmcp_server_path).resolve())

            # Use only the main analysis server configuration
            server_config = {}
            for key, val in analysis_server_config.items():
                server_config[key] = val
            for key, val in data_process_server_config.items():
                server_config[key] = val

            logger.info(f"Starting MCP servers: {list(server_config.keys())}")

            # Create MCP client with better error handling
            try:
                self.mcp_client = MultiServerMCPClient(server_config)

                # Load tools from MCP server with connection error handling
                try:
                    self.tools = await self.mcp_client.get_tools()
                    logger.info(
                        f"Loaded {len(self.tools)} tools from {len(server_config)} FastMCP servers"
                    )
                    logger.info(f"Active servers: {', '.join(server_config.keys())}")
                except (
                    ConnectionError,
                    ConnectionResetError,
                    OSError,
                ) as conn_error:
                    logger.warning(f"Connection error loading tools: {conn_error}")
                    # Try to continue with partial tools if any were loaded
                    if not self.tools:
                        raise

            except Exception as mcp_error:
                logger.error(f"MCP client initialization failed: {mcp_error}")
                logger.error(f"Full traceback: {traceback.format_exc()}")
                # Clean up any partial connections
                if hasattr(self, "mcp_client") and self.mcp_client:
                    try:
                        await self.mcp_client.close()
                    except Exception:
                        pass  # Ignore cleanup errors
                    self.mcp_client = None
                raise

            # Create React agent with tools
            self.react_agent = create_react_agent(
                self.llm, self.tools
            )

            logger.info("React agent created successfully with MCP tools")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize MCP tools: {e}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            
            # FALLBACK: Create a React agent without tools for basic functionality
            logger.warning("Creating fallback React agent without MCP tools")
            try:
                self.react_agent = create_react_agent(self.llm, [])
                self.tools = []
                logger.info("Fallback React agent created (no tools available)")
                return True
            except Exception as fallback_error:
                logger.error(f"Failed to create fallback React agent: {fallback_error}")
                self.react_agent = None
                return False

    async def _rate_limited_react_execution(
        self,
        messages: List,
        config: Dict,
        operation_name: str = "React agent execution",
    ):
        """Execute React agent with rate limiting, exponential backoff, and Langfuse tracing"""
        max_retries = 3
        base_delay = 15.0
        
        for attempt in range(max_retries):
            try:
                if self.rate_limiter:
                    await self.rate_limiter.wait_if_needed()

                # Additional delay for provider-specific rate limits
                provider = getattr(self.config, "llm_provider", "anthropic").lower()
                if provider == "anthropic":
                    # Exponential backoff for retries
                    delay = base_delay * (2 ** attempt)
                    await asyncio.sleep(delay)  # Progressive delay for token acceleration limits
                    logger.debug(f"Attempt {attempt + 1}: Using {delay}s delay for token acceleration limits")
                elif provider == "gemini":
                    # Conservative delay for Gemini free tier
                    delay = 8.0 * (1.5 ** attempt)  # Progressive delay: 8s, 12s, 18s
                    await asyncio.sleep(delay)
                    logger.debug(f"Attempt {attempt + 1}: Using {delay}s delay for Gemini quota limits")
                
                logger.debug(f"Starting {operation_name}")

                # Add Langfuse tracing to the config if available with enhanced metadata
                if self.langfuse_handler:
                    if "callbacks" not in config:
                        config["callbacks"] = []
                    config["callbacks"].append(self.langfuse_handler)

                    # Add better metadata for React execution
                    config["run_name"] = operation_name
                    config["tags"] = ["react_agent", "analysis", "mcp_tools"]
                    if "metadata" not in config:
                        config["metadata"] = {}
                    config["metadata"].update(
                        {
                            "agent_id": self.agent_id,
                            "operation": operation_name,
                            "tool_count": len(self.tools),
                            "available_tools": [
                                tool.name for tool in self.tools[:10]
                            ],  # First 10 tools
                            "message_count": len(messages),
                        }
                    )

                # SAFETY CHECK: Ensure react_agent is available
                if not self.react_agent:
                    raise RuntimeError("React agent not initialized - MCP tools may have failed to load")
                
                response = await self.react_agent.ainvoke(
                    {"messages": messages},
                    {"recursion_limit": 50, **config} 
                )

                if self.rate_limiter:
                    self.rate_limiter.record_request()
                return response
                
            except Exception as e:
                error_msg = str(e)
                if self.rate_limiter:
                    self.rate_limiter.record_request()
                    
                # Handle rate limiting errors (both Anthropic and Gemini)
                if ("rate_limit" in error_msg.lower() or 
                    "quota" in error_msg.lower() or 
                    "429" in error_msg or
                    "resourceexhausted" in error_msg.lower()):
                    if attempt < max_retries - 1:
                        if provider == "gemini":
                            wait_time = 45 * (attempt + 1)  # 45s, 90s, 135s for Gemini
                            logger.warning(f"Gemini quota limit hit on attempt {attempt + 1}. Waiting {wait_time}s before retry...")
                        else:
                            wait_time = 60 * (attempt + 1)  # 60s, 120s, 180s for others
                            logger.warning(f"Rate limit hit on attempt {attempt + 1}. Waiting {wait_time}s before retry...")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        logger.error(f"Failed after {max_retries} attempts due to rate/quota limits")
                        raise
                else:
                    # Non-rate-limit error, don't retry
                    logger.error(f"Error in {operation_name}: {e}")
                    raise
        
        # This should never be reached due to the raise in the except block
        raise Exception(f"Max retries ({max_retries}) exceeded")

    async def start_session(self) -> str:
        """Start a new analysis session"""
        session_id = str(uuid.uuid4())

        self.current_state = AgentState(
            messages=[],
            session_id=session_id,
            dataset_path=None,
            current_instruction=None,
            analysis_results=[],
            orchestrator_feedback=None,
            report_ready=False,
        )

        logger.info(f"Started new session: {session_id}")
        return session_id

    # A2A Protocol Implementation
    async def receive_a2a_message(self, request: SendMessageRequest) -> Dict[str, Any]:
        """
        Receive and process A2A message from orchestrator using official A2A SDK format
        
        Processes only FilePart and DataPart - no TextPart processing.
        Goes directly to instruction handling.
        """
        message = request.params.message
        logger.info(
            f"Received A2A message: {message.message_id} in context {message.context_id}"
        )

        # Extract content from message parts using proper A2A SDK helper functions
        uploaded_files = []
        
        # Handle DataPart objects using SDK helper
        data_parts = get_data_parts(message.parts)
        for data_obj in data_parts:
            # Normalize known key variants for robustness (plural -> singular)
            if isinstance(data_obj, dict):
                # If some components emit 'analysis_instructions' (plural), normalize to singular
                if "analysis_instructions" in data_obj and "analysis_instruction" not in data_obj:
                    try:
                        data_obj["analysis_instruction"] = data_obj.get("analysis_instructions")
                    except Exception:
                        # Non-critical: leave as-is if assignment fails
                        pass

            uploaded_files.append(
                {
                    "name": "data_part",
                    "data": data_obj,
                    "type": "data",
                }
            )
            if isinstance(data_obj, dict):
                logger.info(f"Received DataPart with keys: {list(data_obj.keys())}")
            else:
                logger.info(f"Received DataPart with data type: {type(data_obj)}")

        # Handle FilePart objects using SDK helper
        file_parts = get_file_parts(message.parts)
        for file_info in file_parts:
            if hasattr(file_info, "bytes"):
                # FileWithBytes - bytes field contains base64-encoded content
                try:
                    import base64
                    # Decode base64 string to bytes, then decode to UTF-8
                    decoded_bytes = base64.b64decode(file_info.bytes)
                    file_content = decoded_bytes.decode("utf-8")
                except Exception as e:
                    logger.warning(f"Error decoding FileWithBytes: {e}")
                    # Fallback: treat as plain string if base64 decode fails
                    file_content = file_info.bytes
                
                file_name = getattr(file_info, "name", "uploaded_file.txt")
                uploaded_files.append({
                    "name": file_name,
                    "content": file_content,
                    "type": "bytes",
                })
                logger.info(f"Received uploaded file: {file_name} ({len(file_content)} chars)")
            elif hasattr(file_info, "uri"):
                # FileWithUri
                file_uri = file_info.uri
                file_name = getattr(file_info, "name", "uploaded_file.txt")
                uploaded_files.append({
                    "name": file_name,
                    "uri": file_uri,
                    "type": "uri",
                })
                logger.info(f"Received file URI: {file_name} -> {file_uri}")

        # Go directly to instruction handling - no message type determination needed
        return await self._handle_instruction_a2a(message, "", uploaded_files)

    def _detect_structured_data(self, content: str) -> Optional[Dict[str, Any]]:
        """
        Smart structured data detection - improved to distinguish real data from metadata.
        Returns None for A2A metadata, actual data for conversion.
        """
        # Quick check: If this looks like A2A format, don't treat as structured data
        if self._is_a2a_format(content):
            logger.debug("Content appears to be A2A format, skipping structured data detection")
            return None
            
        return self._detect_structured_data_regex(content)

    def _is_a2a_format(self, content: str) -> bool:
        """
        Smart A2A format detection that allows JSON processing when appropriate.
        
        Returns True (skip structured data detection) only when:
        1. It's A2A format AND
        2. There's no JSON data that needs processing in the Instructions field
        """
        lines = content.strip().split('\n')
        has_dataset_line = any(line.startswith('Dataset:') for line in lines)
        has_instructions_line = any(line.startswith('Instructions:') for line in lines)
        has_session_line = any(line.startswith('Session ID:') for line in lines)
        
        # Check if this is A2A format structure
        is_a2a_structure = has_dataset_line or (has_instructions_line and has_session_line)
        
        if not is_a2a_structure:
            return False
        
        # If it's A2A format, check if Instructions contain JSON data that needs processing
        for line in lines:
            if line.startswith('Instructions:'):
                instructions_content = line.replace('Instructions:', '').strip()
                # Continue reading multi-line instructions until Session ID or end
                line_index = lines.index(line)
                for next_line in lines[line_index + 1:]:
                    if next_line.startswith('Session ID:'):
                        break
                    instructions_content += '\n' + next_line
                
                # Check if instructions contain actual JSON data (not just metadata)
                if self._contains_processable_json_data(instructions_content):
                    logger.debug("A2A format contains JSON data in Instructions - allowing structured data processing")
                    return False  # Allow structured data processing
                
                break
        
        # A2A format without JSON data - skip structured data processing
        logger.debug("Pure A2A format without embedded JSON data - skipping structured data processing")
        return True

    def _contains_processable_json_data(self, content: str) -> bool:
        """Check if content contains JSON data that should be processed (not just metadata)."""
        try:
            # Try to parse as JSON
            parsed = json.loads(content)
            if isinstance(parsed, dict):
                # Check for actual data fields that indicate a real dataset
                data_indicators = ['data', 'records', 'dataset_json', 'db_data']
                for indicator in data_indicators:
                    if indicator in parsed and parsed[indicator]:
                        # Verify it's substantial data, not just metadata
                        data_value = parsed[indicator]
                        if isinstance(data_value, list) and len(data_value) > 0:
                            # Check if it's array of objects (actual records)
                            if isinstance(data_value[0], dict):
                                logger.debug(f"Found processable JSON data in '{indicator}' field with {len(data_value)} records")
                                return True
                        elif isinstance(data_value, str) and (data_value.startswith('[') or data_value.startswith('{')):
                            logger.debug(f"Found processable JSON string in '{indicator}' field")
                            return True
                
                # Check if the whole structure looks like it contains significant data
                if len(str(parsed)) > 500:  # Arbitrary threshold for "substantial" content
                    logger.debug("Found substantial JSON content that might contain data")
                    return True
                    
        except (json.JSONDecodeError, TypeError):
            # Not valid JSON
            pass
        
        return False

    def _detect_structured_data_regex(self, content: str) -> Optional[Dict[str, Any]]:
        """
        Smart regex-based parsing that handles all known formats dynamically.
        """
        try:
            # Try to parse the entire content as JSON first
            try:
                parsed_content = json.loads(content)
                if isinstance(parsed_content, dict):
                    # Look for any field that contains data (dataset_*, *_data, data)
                    for key, value in parsed_content.items():
                        if any(pattern in key.lower() for pattern in ['dataset_', '_data', 'data']) and value:
                            return {
                                'dataset_path': str(value) if not isinstance(value, (list, dict)) else json.dumps(value),
                                'analysis_instructions': parsed_content.get('instructions', ''),
                                'metadata': {k: v for k, v in parsed_content.items() if k not in [key, 'instructions']},
                                'confidence': 0.9
                            }
            except json.JSONDecodeError:
                pass  # Not pure JSON, continue with pattern matching
            
            # Dynamic pattern for any dataset field
            dataset_pattern = r'"((?:dataset_\w+|\w*_data|data))":\s*(?:"([^"]*(?:\\.[^"]*)*)"|(\[[^\]]*\])|(\{[^}]*\}))'
            
            for match in re.finditer(dataset_pattern, content):
                field_name = match.group(1)
                string_value = match.group(2)
                array_value = match.group(3)
                object_value = match.group(4)
                
                try:
                    if string_value is not None:
                        # Handle string format
                        json_string = self._enhanced_unescape(string_value)
                        parsed_data = json.loads(json_string)
                        logger.info(f"Detected structured data in {field_name}")
                        return {
                            'dataset_path': json_string,
                            'analysis_instructions': '',
                            'metadata': {'field_name': field_name, 'format': 'string'},
                            'confidence': 0.8
                        }
                    elif array_value is not None:
                        # Handle array format
                        logger.info(f"Detected structured data in {field_name} (array)")
                        return {
                            'dataset_path': array_value,
                            'analysis_instructions': '',
                            'metadata': {'field_name': field_name, 'format': 'array'},
                            'confidence': 0.8
                        }
                    elif object_value is not None:
                        # Handle object format
                        logger.info(f"Detected structured data in {field_name} (object)")
                        return {
                            'dataset_path': object_value,
                            'analysis_instructions': '',
                            'metadata': {'field_name': field_name, 'format': 'object'},
                            'confidence': 0.8
                        }
                except (json.JSONDecodeError, ValueError) as e:
                    logger.debug(f"Failed to parse {field_name}: {e}")
                    continue
            
            # Check if this is just A2A metadata without data
            if self._has_only_metadata(content):
                logger.debug("Content contains only A2A metadata, no structured data")
                return None
            
            # Look for simple patterns like "Dataset: <path>" or structured markers
            lines = content.strip().split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith('Dataset:'):
                    dataset_path = line.replace('Dataset:', '').strip()
                    if dataset_path:
                        return {
                            'dataset_path': dataset_path,
                            'analysis_instructions': content,
                            'metadata': {'format': 'simple_marker'},
                            'confidence': 0.7
                        }
            
            return None
                    
        except Exception as e:
            logger.debug(f"Error in structured data detection: {e}")
            return None

    def _extract_dataset_info(self, content: str = None, uploaded_files: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Smart dataset extraction that handles both TextPart content and DataPart uploads.
        Priority: DataPart dataset paths > DataPart structured data > A2A file paths > A2A embedded JSON > File references > Structured data > None
        
        Args:
            content: TextPart content string (optional, for backward compatibility)
            uploaded_files: DataPart uploads list (optional, for modern A2A processing)
        """
        dataset_info = {
            "type": "unknown",
            "path": None,
            "structured_data": None,
            "raw_content": content or ""
        }
        
        # PRIORITY 1: DataPart Dataset Paths (highest priority for modern A2A)
        if uploaded_files:
            for uploaded_file in uploaded_files:
                if uploaded_file.get("type") == "data" and isinstance(uploaded_file.get("data"), dict):
                    data_obj = uploaded_file["data"]
                    
                    # Check for direct dataset path
                    path_keys = ["dataset_path", "path", "file_path", "dataset"]
                    for key in path_keys:
                        if key in data_obj and data_obj[key]:
                            path_value = data_obj[key]
                            if (path_value != "No explicit dataset provided" and
                                "no dataset" not in path_value.lower() and
                                ('.' in path_value or '/' in path_value or '\\' in path_value)):
                                logger.info(f"Found dataset path in DataPart key '{key}': {path_value}")
                                dataset_info.update({
                                    "type": "file_path",
                                    "path": path_value,
                                    "metadata": data_obj,
                                    "source": "datapart_path"
                                })
                                return dataset_info
        
        # PRIORITY 2: DataPart Structured Data (second highest priority)
        if uploaded_files:
            for uploaded_file in uploaded_files:
                if uploaded_file.get("type") == "data" and isinstance(uploaded_file.get("data"), dict):
                    data_obj = uploaded_file["data"]
                    
                    # Check for structured data
                    data_keys = ["db_data", "dataset_json", "records", "data"]
                    for key in data_keys:
                        if key in data_obj and data_obj[key]:
                            # Verify this contains ACTUAL data, not just metadata
                            has_actual_data = any(data_key in data_obj for data_key in ['db_data', 'dataset_json', 'records', 'data'])
                            
                            if has_actual_data:
                                logger.info(f"Found structured data in DataPart key '{key}'")
                                dataset_info.update({
                                    "type": "structured",
                                    "structured_data": data_obj,
                                    "path": f"datapart_structured_{uuid.uuid4().hex[:8]}",
                                    "source": "datapart_structured"
                                })
                                return dataset_info
        
        # PRIORITY 3: A2A Protocol Format from TextPart - Handle file paths
        if content:
            lines = content.strip().split("\n")
            for line in lines:
                if line.startswith("Dataset:"):
                    path = line.replace("Dataset:", "").strip()
                    # Skip placeholder paths but allow real file paths
                    if path and path != "No explicit dataset provided" and "no dataset" not in path.lower():
                        logger.info(f"Found A2A dataset path: {path}")
                        dataset_info.update({
                            "type": "file_path",
                            "path": path,
                            "source": "a2a_format"
                        })
                        return dataset_info
                    else:
                        logger.debug(f"A2A format detected with placeholder: {path}")
                        # Continue to check for JSON data in Instructions
                        break
        
        # PRIORITY 4: A2A with JSON data in Instructions field from TextPart
        if content:
            lines = content.strip().split("\n")
            for line in lines:
                if line.startswith('Instructions:'):
                    instructions_content = line.replace('Instructions:', '').strip()
                    # Continue reading multi-line instructions until Session ID or end
                    line_index = lines.index(line)
                    for next_line in lines[line_index + 1:]:
                        if next_line.startswith('Session ID:'):
                            break
                        instructions_content += '\n' + next_line
                    
                    # Check if instructions contain processable JSON data
                    if self._contains_processable_json_data(instructions_content):
                        try:
                            parsed_instructions = json.loads(instructions_content)
                            logger.info(f"Found A2A with embedded JSON data: {list(parsed_instructions.keys())}")
                            dataset_info.update({
                                "type": "structured",
                                "structured_data": parsed_instructions,
                                "path": f"a2a_embedded_data_{uuid.uuid4().hex[:8]}",
                                "source": "a2a_embedded_json"
                            })
                            return dataset_info
                        except json.JSONDecodeError:
                            logger.debug("Instructions content is not valid JSON")
                    break
        
        # PRIORITY 5: Look for CSV file mentions in plain text from TextPart
        if content:
            import re
            csv_matches = re.findall(r'[\w\-_/\\]+\.csv', content)
            if csv_matches:
                logger.info(f"Found CSV file reference: {csv_matches[0]}")
                dataset_info.update({
                    "type": "file_path", 
                    "path": csv_matches[0],
                    "source": "file_reference"
                })
                return dataset_info
        
        # PRIORITY 6: Check for structured data (JSON with actual datasets) from TextPart
        if content:
            structured_data = self._detect_structured_data(content)
            if structured_data:
                # Verify this contains ACTUAL data, not just metadata
                has_actual_data = any(key in structured_data for key in ['db_data', 'dataset_json', 'records', 'data'])
                
                # Check if dataset_path contains a real file path
                dataset_path = structured_data.get('dataset_path', '')
                is_real_file_path = (
                    dataset_path and 
                    dataset_path != "No explicit dataset provided" and
                    "no dataset" not in dataset_path.lower() and
                    not dataset_path.startswith("structured_data_") and
                    ('.' in dataset_path or '/' in dataset_path or '\\' in dataset_path)
                )
                
                if is_real_file_path and not has_actual_data:
                    # This is metadata with a real file path
                    logger.info(f"Found real dataset_path in structured metadata: {dataset_path}")
                    dataset_info.update({
                        "type": "file_path",
                        "path": dataset_path,
                        "metadata": structured_data,
                        "source": "structured_metadata"
                    })
                    return dataset_info
                elif has_actual_data:
                    # This contains actual data that needs conversion
                    logger.info(f"Found structured data for conversion: {list(structured_data.keys())}")
                    dataset_info.update({
                        "type": "structured",
                        "structured_data": structured_data,
                        "path": f"structured_data_{uuid.uuid4().hex[:8]}",
                        "source": "structured_data"
                    })
                    return dataset_info
        
        # PRIORITY 7: No dataset detected
        logger.debug("No dataset detected in content or uploaded files")
        dataset_info.update({
            "type": "none",
            "path": None,
            "source": "none"
        })
        
        return dataset_info

    async def _create_temp_dataset_from_structured_data(self, structured_data: Dict[str, Any]) -> str:
        """Convert structured data to temporary file for analysis tools"""
        temp_dir = Path(getattr(self.config, "temp_datasets_path", "temp_datasets"))
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            # Helper to normalize input into Python object (list/dict)
            def _ensure_parsed(obj):
                # If bytes, decode
                if isinstance(obj, (bytes, bytearray)):
                    try:
                        obj = obj.decode('utf-8')
                    except Exception:
                        obj = obj.decode('latin-1')

                # If it's already a dict or list, return as-is
                if isinstance(obj, (dict, list)):
                    return obj

                # If it's a string, try to load JSON, otherwise return string
                if isinstance(obj, str):
                    try:
                        return json.loads(obj)
                    except Exception:
                        # Not JSON, return original string
                        return obj

                # Unknown type - return as-is
                return obj

            # If the incoming structured_data is a string containing JSON, parse it up-front.
            if isinstance(structured_data, str):
                try:
                    structured_data = json.loads(structured_data)
                    logger.debug("Normalized structured_data: parsed JSON string into object")
                except Exception:
                    # Try to extract a JSON object blob inside the string as a last resort
                    import re
                    m = re.search(r"(\{\s*\"db_data\"[\s\S]*\})", structured_data)
                    if m:
                        try:
                            structured_data = json.loads(m.group(1))
                            logger.debug("Normalized structured_data: extracted inner JSON blob")
                        except Exception:
                            logger.debug("Could not parse inner JSON blob from structured_data string")

            if 'db_data' in structured_data:
                # Handle nested JSON format like from your report
                db_list = structured_data.get('db_data')

                # Defensive: ensure we have a non-empty list
                if not isinstance(db_list, list) or len(db_list) == 0:
                    logger.warning("db_data present but empty or not a list; falling back to structured_data content")
                    # Try to use the db_data object directly if possible, otherwise use the whole structured_data
                    parsed_data = _ensure_parsed(structured_data.get('db_data', structured_data))
                else:
                    # If db_list already contains dicts (list-of-dicts), treat the whole list as the dataset
                    if all(isinstance(item, dict) for item in db_list):
                        parsed_data = db_list
                    else:
                        raw = db_list[0]
                        parsed_data = _ensure_parsed(raw)

                        # Some legacy clients send a JSON-encoded string inside a list (double-encoded).
                        # If parsed_data is still a string, attempt to json.loads it again (up to 2 attempts).
                        if isinstance(parsed_data, str):
                            for _ in range(2):
                                try:
                                    parsed_candidate = json.loads(parsed_data)
                                    parsed_data = parsed_candidate
                                except Exception:
                                    # Stop trying if it can't be parsed further
                                    break

                # Convert to CSV for analysis tools
                if isinstance(parsed_data, list):
                    df = pd.DataFrame(parsed_data)
                elif isinstance(parsed_data, dict):
                    df = pd.DataFrame([parsed_data])
                else:
                    # Couldn't parse into structured rows - dump as JSON
                    temp_file = temp_dir / f"structured_dataset_{timestamp}.json"
                    with open(temp_file, 'w', encoding='utf-8') as f:
                        json.dump(parsed_data, f, indent=2)
                    logger.warning(f"db_data not tabular, saved as JSON: {temp_file}")
                    logger.info(f"Created temporary dataset file: {temp_file}")
                    return str(temp_file)

                temp_file = temp_dir / f"structured_dataset_{timestamp}.csv"
                df.to_csv(temp_file, index=False)
                logger.info(f"Converted db_data to CSV: {temp_file}")

            elif 'dataset_json' in structured_data:
                # Handle dataset_json format (your colleague's format)
                if 'parsed_data' in structured_data:
                    # Use the pre-parsed data
                    parsed_data = structured_data['parsed_data']
                    logger.info(f"Using pre-parsed dataset_json data: {type(parsed_data)} with {len(parsed_data) if isinstance(parsed_data, list) else 'N/A'} items")
                else:
                    # Parse the JSON string or take the object
                    raw = structured_data['dataset_json'][0]
                    parsed_data = _ensure_parsed(raw)
                    logger.info(f"Parsed dataset_json data type: {type(parsed_data)}")

                # Convert to CSV for analysis tools
                if isinstance(parsed_data, list):
                    df = pd.DataFrame(parsed_data)
                elif isinstance(parsed_data, dict):
                    df = pd.DataFrame([parsed_data])
                else:
                    temp_file = temp_dir / f"structured_dataset_{timestamp}.json"
                    with open(temp_file, 'w', encoding='utf-8') as f:
                        json.dump(parsed_data, f, indent=2)
                    logger.warning(f"dataset_json not tabular, saved as JSON: {temp_file}")
                    logger.info(f"Created temporary dataset file: {temp_file}")
                    return str(temp_file)

                temp_file = temp_dir / f"structured_dataset_{timestamp}.csv"
                df.to_csv(temp_file, index=False)
                logger.info(f"Converted dataset_json to CSV: {temp_file}")
                
            else:
                # Handle other structured data formats
                # Try to convert to DataFrame if possible
                try:
                    if isinstance(structured_data, dict):
                        # If it's a dict with list values, try to make a DataFrame
                        list_values = [v for v in structured_data.values() if isinstance(v, list)]
                        if list_values:
                            df = pd.DataFrame(list_values[0])
                            temp_file = temp_dir / f"structured_dataset_{timestamp}.csv"
                            df.to_csv(temp_file, index=False)
                        else:
                            # Save as JSON if can't convert to tabular
                            temp_file = temp_dir / f"structured_dataset_{timestamp}.json"
                            with open(temp_file, 'w') as f:
                                json.dump(structured_data, f, indent=2)
                    else:
                        # Fallback to JSON
                        temp_file = temp_dir / f"structured_dataset_{timestamp}.json"
                        with open(temp_file, 'w') as f:
                            json.dump(structured_data, f, indent=2)
                            
                except Exception:
                    # Final fallback - save as JSON
                    temp_file = temp_dir / f"structured_dataset_{timestamp}.json"
                    with open(temp_file, 'w') as f:
                        json.dump(structured_data, f, indent=2)
                        
            logger.info(f"Created temporary dataset file: {temp_file}")
            return str(temp_file)
            
        except Exception as e:
            logger.error(f"Error creating temp dataset: {e}")
            # Create a simple JSON file as fallback
            temp_file = temp_dir / f"fallback_dataset_{timestamp}.json"
            with open(temp_file, 'w') as f:
                json.dump({"error": str(e), "original_data": str(structured_data)}, f, indent=2)
            return str(temp_file)

    async def _create_temp_dataset_from_file_content(self, file_content: str, file_name: str) -> str:
        """Save uploaded file content to temporary file for analysis tools"""
        temp_dir = Path(getattr(self.config, "temp_datasets_path", "temp_datasets"))
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            # Determine file extension
            file_ext = Path(file_name).suffix if '.' in file_name else '.txt'
            temp_file = temp_dir / f"uploaded_dataset_{timestamp}{file_ext}"
            
            # Save file content
            with open(temp_file, 'w', encoding='utf-8') as f:
                f.write(file_content)
            
            logger.info(f"Uploaded file saved to: {temp_file}")
            return str(temp_file)
            
        except Exception as e:
            logger.error(f"Error saving uploaded file: {e}")
            # Fallback to basic text file
            temp_file = temp_dir / f"uploaded_dataset_{timestamp}.txt"
            with open(temp_file, 'w', encoding='utf-8') as f:
                f.write(file_content)
            return str(temp_file)


    def _is_a2a_csv_mode(self, instructions: str) -> bool:
        """
        Targeted detection for A2A CSV analysis mode to prevent unnecessary JSON conversion.
        
        This specifically identifies A2A protocol requests for CSV analysis where:
        - analysis_type is "csv" 
        - dataset_path points to a CSV file
        - No embedded data arrays that need conversion
        
        This is much more targeted than general A2A metadata detection.
        
        Args:
            instructions: The instruction string to check
            
        Returns:
            True if this is A2A CSV mode that should skip JSON workflow, False otherwise
        """
        import re
        
        if not instructions:
            return False
        
        # Check for specific A2A CSV mode indicators
        has_csv_analysis_type = bool(re.search(r'"analysis_type"\s*:\s*"csv"', instructions))
        has_csv_dataset_path = bool(re.search(r'"dataset_path"\s*:\s*"[^"]*\.csv"', instructions))
        
        # If both CSV indicators are present, this is A2A CSV mode
        if has_csv_analysis_type and has_csv_dataset_path:
            logger.debug("A2A CSV mode detected: analysis_type=csv and dataset_path=*.csv")
            return True
            
        return False

    # ENHANCED JSON WORKFLOW METHODS
    async def _enhance_instructions_with_json_conversion(
        self, instructions: str, dataset_path: str = None
    ) -> tuple[str, str]:
        """
        Enhanced instruction processing - only converts ACTUAL JSON data, not A2A metadata.
        
        Returns:
            Tuple of (enhanced_instructions, converted_file_path)
        """
        try:
            # PERFORMANCE OPTIMIZATION: Early exit for simple CSV analysis
            # Check if dataset_path exists and instructions are simple
            if dataset_path and Path(dataset_path).exists() and len(instructions.strip()) < 500:
                # Simple case: existing file + short instructions = skip JSON processing
                logger.debug("Simple CSV analysis detected - skipping JSON conversion workflow")
                return instructions, dataset_path

            # EARLY EXIT 1: A2A Format Detection - Skip JSON processing for A2A messages
            if self._is_a2a_format(instructions):
                logger.debug("A2A format detected - skipping JSON conversion workflow")
                return instructions, dataset_path

            # EARLY EXIT 2: CSV Mode Detection - Skip for explicit CSV analysis
            if self._is_a2a_csv_mode(instructions):
                logger.debug("A2A CSV mode detected - skipping JSON conversion workflow")
                return instructions, dataset_path
            
            # EARLY EXIT 3: Already Processed - Skip if we already converted structured data
            if dataset_path and "structured_dataset_" in str(dataset_path):
                logger.debug("Structured data already processed - skipping JSON conversion")
                return instructions, dataset_path
            
            # EARLY EXIT 4: No JSON patterns - Quick regex check before expensive parsing
            if not any(pattern in instructions for pattern in ['{', '[', '"data":', '"records":']):
                logger.debug("No JSON patterns detected - skipping JSON conversion")
                return instructions, dataset_path
            
            # ONLY NOW: Check for embedded JSON data that needs conversion
            json_data = self._extract_json_fallback_regex(instructions)
            
            if not json_data:
                logger.debug("No convertible JSON data detected in instructions")
                return instructions, dataset_path
            
            logger.info(f"Detected convertible JSON data: {list(json_data.keys())}")
            
            # Convert JSON to temporary CSV file
            temp_csv_path = await self._convert_json_to_temp_csv(json_data)
            
            if not temp_csv_path:
                logger.warning("Failed to convert JSON to CSV, using original instructions")
                return instructions, dataset_path
            
            # Enhance instructions to reference the converted file
            enhanced_instructions = self._enhance_instructions_with_file_reference(
                instructions, temp_csv_path, json_data
            )
            
            logger.info(f"JSON successfully converted to CSV: {temp_csv_path}")
            logger.debug(f"Enhanced instructions created: {enhanced_instructions[:200]}...")
            
            return enhanced_instructions, temp_csv_path
            
        except Exception as e:
            logger.error(f"Error in JSON workflow enhancement: {e}")
            return instructions, dataset_path
    
    def _extract_json_fallback_regex(self, instructions: str) -> dict:
        """
        Smart JSON extraction using json package for robust parsing.
        Handles any dataset format dynamically: dataset_json, dataset_abc, db_data, etc.
        
        This is the consolidated JSON extraction method that works for both sync and async contexts.
        """
        try:
            # Try to parse the entire instructions as JSON first
            try:
                parsed_instructions = json.loads(instructions)
                if isinstance(parsed_instructions, dict):
                    # Look for any field that contains ACTUAL data (not just metadata)
                    for key, value in parsed_instructions.items():
                        if self._is_actual_dataset_field(key, value):
                            result = self._process_found_data(key, value, instructions)
                            if result:
                                return result
            except json.JSONDecodeError:
                pass  # Not pure JSON, continue with regex extraction
            
            # Enhanced dynamic pattern for dataset fields - more specific patterns
            dataset_pattern = r'"((?:dataset_json|dataset_data|db_data|records|data_records))":\s*(?:"([^"]*(?:\\.[^"]*)*)"|(\[[^\]]*\])|(\{[^}]*\}))'
            
            for match in re.finditer(dataset_pattern, instructions):
                field_name = match.group(1)
                string_value = match.group(2)  # String format: "field": "json_string"
                array_value = match.group(3)   # Array format: "field": [...]
                object_value = match.group(4)  # Object format: "field": {...}
                
                try:
                    if string_value is not None:
                        # Handle string format with enhanced unescaping
                        json_string = self._enhanced_unescape(string_value)
                        parsed_data = json.loads(json_string)
                        logger.info(f"Extracted {field_name}: {type(parsed_data)} with {len(parsed_data) if isinstance(parsed_data, list) else 'N/A'} items")
                        return {"dataset_json": [json_string], "parsed_data": parsed_data}
                    
                    elif array_value is not None:
                        # Handle array format
                        parsed_array = json.loads(array_value)
                        logger.info(f"Extracted {field_name}: array with {len(parsed_array)} items")
                        return {"dataset_json": parsed_array, "parsed_data": parsed_array}
                    
                    elif object_value is not None:
                        # Handle object format
                        parsed_object = json.loads(object_value)
                        logger.info(f"Extracted {field_name}: object")
                        return {"dataset_json": [json.dumps(parsed_object)], "parsed_data": parsed_object}
                        
                except (json.JSONDecodeError, ValueError) as e:
                    logger.debug(f"Failed to parse {field_name}: {e}")
                    continue
            
            # Check for A2A protocol metadata only (no actual data)
            if self._has_only_metadata(instructions):
                logger.debug("Detected A2A protocol metadata only")
                return None
                
            return None
                    
        except Exception as e:
            logger.debug(f"Error in JSON extraction: {e}")
            return None

    def _is_actual_dataset_field(self, key: str, value) -> bool:
        """Check if a field contains actual dataset content vs metadata."""
        # Specific patterns for ACTUAL dataset fields
        actual_data_patterns = ['dataset_json', 'dataset_data', 'db_data', 'records', 'data_records']
        
        # Metadata fields that should NOT be treated as datasets
        metadata_patterns = [
            'total_records_in_file', 'sampled_records', 'sampling_method', 'data_source',
            'task', 'analysis_instruction', 'dataset_path', 'analysis_type', 'output_requirements'
        ]
        
        key_lower = key.lower()
        
        # If it's a known metadata field, it's not actual data
        if any(pattern in key_lower for pattern in metadata_patterns):
            return False
        
        # If it matches actual data patterns, check the value
        if any(pattern in key_lower for pattern in actual_data_patterns):
            # Must have substantial content (arrays/objects, not simple strings)
            if isinstance(value, list) and len(value) > 0:
                return True
            if isinstance(value, str) and (value.startswith('[') or value.startswith('{')):
                return True
            if isinstance(value, dict):
                return True
        
        return False

    def _process_found_data(self, key: str, value, instructions: str) -> dict:
        """Process found data from parsed JSON instructions."""
        try:
            if isinstance(value, str):
                # Try to parse string value as JSON with enhanced unescaping
                json_string = self._enhanced_unescape(value)
                parsed_data = json.loads(json_string)
                return {"dataset_json": [value], "parsed_data": parsed_data}
            elif isinstance(value, list):
                # Direct array data
                return {"dataset_json": value, "parsed_data": value}
            elif isinstance(value, dict):
                # Direct object data
                return {"dataset_json": [json.dumps(value)], "parsed_data": value}
        except (json.JSONDecodeError, TypeError):
            pass
        return None

    def _enhanced_unescape(self, json_string: str) -> str:
        """Enhanced unescaping that handles multiple levels of escaping."""
        # Handle multiple levels of escaping systematically
        # Level 1: \\\\\" -> \\\"
        json_string = json_string.replace('\\\\\"', '\\\"')
        # Level 2: \\\" -> \"
        json_string = json_string.replace('\\\"', '"')
        # Level 3: \" -> "
        json_string = json_string.replace('\\"', '"')
        # Handle other escapes
        json_string = json_string.replace('\\n', '\n').replace('\\t', '\t')
        return json_string

    def _has_only_metadata(self, instructions: str) -> bool:
        """Check if instructions contain only A2A metadata without actual data."""
        # Core A2A metadata patterns
        metadata_patterns = [
            'task', 'analysis_instruction', 'dataset_path', 'analysis_type', 'output_requirements',
            'total_records_in_file', 'sampled_records', 'sampling_method', 'data_source'
        ]
        
        # Check for metadata indicators
        metadata_count = sum(1 for pattern in metadata_patterns if f'"{pattern}"' in instructions)
        
        # Check for actual data arrays or large JSON structures (real datasets)
        has_data_arrays = (
            '[{' in instructions or  # Array of objects (real data)
            '{"' in instructions and instructions.count('"') > 20 or  # Large JSON structure
            any(pattern in instructions for pattern in ['"dataset_json"', '"db_data"', '"records"'])
        )
        
        # If we have metadata indicators but no actual data arrays, it's metadata-only
        is_metadata_only = metadata_count >= 2 and not has_data_arrays
        
        if is_metadata_only:
            logger.debug(f"Detected metadata-only content: {metadata_count} metadata patterns, no data arrays")
        
        return is_metadata_only
        
    async def _convert_json_to_temp_csv(self, json_data: dict) -> str:
        """Convert JSON data to temporary CSV file"""
        try:
            temp_dir = Path(getattr(self.config, "temp_datasets_path", "temp_datasets"))
            temp_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            temp_file = temp_dir / f"json_converted_dataset_{timestamp}.csv"
            
            # Helper to normalize input into Python object (list/dict)
            def _ensure_parsed(obj):
                # If bytes, decode
                if isinstance(obj, (bytes, bytearray)):
                    try:
                        obj = obj.decode('utf-8')
                    except Exception:
                        obj = obj.decode('latin-1')

                # If it's already a dict or list, return as-is
                if isinstance(obj, (dict, list)):
                    return obj

                # If it's a string, try to load JSON, otherwise return string
                if isinstance(obj, str):
                    try:
                        return json.loads(obj)
                    except Exception:
                        return obj

                # Unknown type - return as-is
                return obj

            # Handle different JSON structures
            if 'db_data' in json_data:
                # Handle A2A format with nested JSON string
                if 'parsed_data' in json_data:
                    # Use the pre-parsed data
                    parsed_data = json_data['parsed_data']
                else:
                    # Parse the JSON string or object
                    raw = json_data['db_data'][0]
                    parsed_data = _ensure_parsed(raw)

                if isinstance(parsed_data, list):
                    df = pd.DataFrame(parsed_data)
                elif isinstance(parsed_data, dict):
                    df = pd.DataFrame([parsed_data])
                else:
                    logger.error("db_data not in a tabular format; cannot convert to CSV")
                    return None

            elif 'dataset_json' in json_data:
                # Handle dataset_json format (your colleague's format)
                if 'parsed_data' in json_data:
                    # Use the pre-parsed data
                    parsed_data = json_data['parsed_data']
                    logger.info(f"Using pre-parsed dataset_json data: {type(parsed_data)} with {len(parsed_data) if isinstance(parsed_data, list) else 'N/A'} items")
                else:
                    # Parse the JSON string or object
                    raw = json_data['dataset_json'][0]
                    parsed_data = _ensure_parsed(raw)
                    logger.info(f"Parsed dataset_json data type: {type(parsed_data)}")

                if isinstance(parsed_data, list):
                    df = pd.DataFrame(parsed_data)
                    logger.info(f"Created DataFrame from dataset_json: {len(df)} rows, {len(df.columns)} columns")
                elif isinstance(parsed_data, dict):
                    df = pd.DataFrame([parsed_data])
                    logger.info(f"Created single-row DataFrame from dataset_json: {len(df.columns)} columns")
                else:
                    logger.error("dataset_json not in a tabular format; cannot convert to CSV")
                    return None

            elif 'records' in json_data:
                # Handle direct records format
                df = pd.DataFrame(json_data['records'])
                
            elif isinstance(json_data, list):
                # Handle direct array format
                df = pd.DataFrame(json_data)
                
            else:
                # Enhanced: Search for tabular data in complex JSON structures
                def find_tabular_data(obj, path=""):
                    """Recursively find the best tabular data in JSON structure"""
                    if isinstance(obj, list) and len(obj) > 0:
                        # Check if this is a list of records (dicts with same keys)
                        if isinstance(obj[0], dict):
                            first_keys = set(obj[0].keys())
                            # Verify it's actually tabular data (consistent structure)
                            if all(isinstance(item, dict) and set(item.keys()) == first_keys for item in obj[:5]):
                                logger.info(f"Found tabular data at path: {path} with {len(obj)} records")
                                return obj
                    
                    if isinstance(obj, dict):
                        for key, value in obj.items():
                            result = find_tabular_data(value, f"{path}.{key}")
                            if result is not None:
                                return result
                    
                    return None
                
                # Search for actual tabular data first
                tabular_data = find_tabular_data(json_data)
                
                if tabular_data:
                    # Found proper tabular data
                    df = pd.DataFrame(tabular_data)
                    logger.info(f"Extracted tabular data: {len(df)} records, {len(df.columns)} columns")
                else:
                    # Try to convert dict to DataFrame
                    try:
                        df = pd.DataFrame([json_data])
                        logger.warning(f"Using whole JSON structure as single row: {len(df.columns)} columns")
                    except Exception:
                        # Fallback: flatten the dict
                        flattened = {}
                        for key, value in json_data.items():
                            if isinstance(value, list) and len(value) > 0:
                                if isinstance(value[0], dict):
                                    # This looks like records
                                    df = pd.DataFrame(value)
                                    break
                                else:
                                    flattened[key] = value
                            else:
                                flattened[key] = [value]
                        else:
                            df = pd.DataFrame(flattened)
                            logger.warning(f"Using flattened structure: {len(df.columns)} columns")
            
            # Save to CSV
            df.to_csv(temp_file, index=False)
            
            logger.info(f"Converted JSON to CSV: {temp_file} ({len(df)} records, {len(df.columns)} columns)")
            return str(temp_file)
            
        except Exception as e:
            logger.error(f"Error converting JSON to CSV: {e}")
            return None
        
    def _enhance_instructions_with_file_reference(
        self, original_instructions: str, csv_path: str, json_data: dict = None
    ) -> str:
        """Enhance instructions to reference the converted CSV file"""
        import re
        import json as json_module
        
        # Extract the actual analysis instruction from the JSON structure
        clean_instructions = original_instructions
        
        try:
            # Try to parse the original instructions as JSON to extract the analysis_instruction
            if original_instructions.strip().startswith('{'):
                parsed = json_module.loads(original_instructions)
                if isinstance(parsed, dict):
                    # Extract the meaningful instruction parts
                    instruction_parts = []
                    
                    if 'task' in parsed:
                        instruction_parts.append(f"Task: {parsed['task']}")
                    
                    if 'analysis_instruction' in parsed:
                        instruction_parts.append(f"Instructions: {parsed['analysis_instruction']}")
                    
                    # Combine the meaningful parts
                    if instruction_parts:
                        clean_instructions = '\n\n'.join(instruction_parts)
                    else:
                        clean_instructions = "Perform comprehensive data analysis"
                        
        except Exception as e:
            logger.debug(f"Could not parse instructions as JSON, using original: {e}")
            # Fallback: Remove only the large data arrays, keep the rest
            # Remove large data arrays but preserve instruction text
            array_pattern = r'"data":\s*\[[^\]]*\]'
            clean_instructions = re.sub(array_pattern, '', clean_instructions, flags=re.DOTALL)
            
            # Clean up multiple spaces and newlines
            clean_instructions = re.sub(r'\s+', ' ', clean_instructions).strip()
        
        # Build enhanced instructions based on whether JSON conversion occurred
        if json_data:
            # JSON conversion occurred
            dataset_context = f"""DATASET CONTEXT:
- This dataset was automatically converted from JSON data embedded in your request
- The dataset contains structured data ready for comprehensive analysis
- Use the provided analysis tools to explore patterns, statistics, and insights

JSON DATA SUMMARY:
- Data keys detected: {list(json_data.keys())}
- Conversion timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"""
        else:
            # Direct CSV file analysis
            dataset_context = """DATASET CONTEXT:
- This dataset is ready for comprehensive analysis
- Use the provided analysis tools to explore patterns, statistics, and insights"""
        
        enhanced = f"""DATASET ANALYSIS REQUEST

DATASET FILE: {csv_path}

ANALYSIS INSTRUCTIONS: {clean_instructions}

{dataset_context}

Please proceed with your analysis using the available tools. The dataset is ready for immediate processing."""

        return enhanced

    async def _handle_instruction_a2a(
        self, message: Message, content_text: str, uploaded_files: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Handle initial instruction from orchestrator with intelligent workflow planning and file support
        
        Works with FilePart and DataPart only - no TextPart processing.
        Instructions and dataset info are extracted from DataPart.
        """

        # Create a Langfuse trace for the entire A2A workflow
        trace = None
        if self.langfuse_handler:
            try:
                langfuse_client = get_client()
                if langfuse_client:
                    trace = langfuse_client.trace(
                        name="A2A_Analysis_Workflow",
                        input=f"Processing {len(uploaded_files)} uploaded parts",
                        user_id=self.agent_id,
                        session_id=message.context_id,
                        tags=["a2a", "analysis", "instruction"],
                        metadata={
                            "agent_id": self.agent_id,
                            "message_id": message.message_id,
                            "context_id": message.context_id,
                        },
                    )
                    logger.info(f"Created Langfuse trace for A2A workflow: {trace.id}")
            except Exception as e:
                logger.warning(f"Failed to create Langfuse trace: {e}")

        try:
            # Extract instructions from DataPart instead of TextPart
            instructions = ""
            
            # Look for instructions in DataPart
            for uploaded_file in uploaded_files or []:
                if uploaded_file.get("type") == "data" and isinstance(uploaded_file.get("data"), dict):
                    data_obj = uploaded_file["data"]
                    # Extract instructions from known keys
                    instruction_keys = ["analysis_instruction", "instructions", "task", "instruction"]
                    for key in instruction_keys:
                        if key in data_obj and data_obj[key]:
                            instructions = data_obj[key]
                            logger.info(f"Found instructions in DataPart key '{key}': {instructions[:100]}...")
                            break
                    if instructions:
                        break
            
            # Fallback: use a default instruction if none found
            if not instructions:
                instructions = "Perform comprehensive data analysis"
                logger.info("No instructions found in DataPart, using default: comprehensive data analysis")
            
            # Debug: Log what we actually received
            logger.debug(f"Extracted instructions: {instructions[:200]}")
            logger.debug(f"Number of uploaded files: {len(uploaded_files or [])}")

            # DYNAMIC DATASET HANDLING - Handle all scenarios
            file_parts = get_file_parts(message.parts)
            dataset_path = None
            dataset_type = "unknown"
            
            if file_parts:
                # Scenario 1: File Upload
                uploaded_file = file_parts[0]  # Use first file
                
                # Create temporary file path for analysis
                temp_dir = Path("temp_uploads")
                temp_dir.mkdir(exist_ok=True)
                
                # Generate unique filename to avoid conflicts
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # FIXED: Handle FilePart attribute safely
                file_name = getattr(uploaded_file, 'name', None) or getattr(uploaded_file, 'filename', None) or f"upload_{timestamp}"
                file_extension = Path(file_name).suffix if '.' in file_name else '.csv'
                local_filename = f"uploaded_{timestamp}_{file_name}"
                dataset_path = str(temp_dir / local_filename)
                dataset_type = "file_upload"
                
                # Save uploaded file data locally
                try:
                    if hasattr(uploaded_file, 'bytes'):
                        # FileWithBytes
                        file_data = uploaded_file.bytes
                        # Handle case where bytes might be base64 encoded string
                        if isinstance(file_data, str):
                            try:
                                file_data = base64.b64decode(file_data)
                            except Exception:
                                file_data = file_data.encode('utf-8')
                        
                        with open(dataset_path, 'wb') as f:
                            f.write(file_data)
                            
                    elif hasattr(uploaded_file, 'data'):
                        # Alternative data attribute
                        file_data = uploaded_file.data
                        if isinstance(file_data, str):
                            file_data = file_data.encode('utf-8')
                        with open(dataset_path, 'wb') as f:
                            f.write(file_data)
                            
                    elif hasattr(uploaded_file, 'uri'):
                        # FileWithUri - would need to download
                        logger.warning(f"URI-based files not yet supported: {uploaded_file.uri}")
                        dataset_path = None
                        
                except Exception as file_error:
                    logger.error(f"Error saving uploaded file: {file_error}")
                    dataset_path = None
                
                logger.info(f"File upload saved as: {dataset_path}")
                
                # CRITICAL: Define dataset_info for file upload path
                dataset_info = {"type": "file_upload", "path": dataset_path}
                
            else:
                # Check for uploaded files first
                if uploaded_files:
                    # Use the first uploaded file descriptor as the dataset
                    uploaded_file = uploaded_files[0]

                    # If the uploaded_file is a typed DataPart-like dict with structured data
                    if isinstance(uploaded_file, dict) and "data" in uploaded_file:
                        try:
                            # FIRST: Check if DataPart contains a valid file path using priority-based extraction
                            dataset_info = self._extract_dataset_info(uploaded_files=uploaded_files)
                            
                            if dataset_info["type"] == "file_path":
                                # Use the detected file path directly
                                dataset_path = dataset_info["path"]
                                dataset_type = "datapart_file_path"
                                logger.info(f"Using DataPart file path: {dataset_path}")
                            else:
                                # No valid file path found, convert nested structured data to CSV
                                structured = uploaded_file.get("data")
                                dataset_path = await self._create_temp_dataset_from_structured_data(
                                    structured
                                )
                                dataset_type = "datapart_structured"
                                dataset_info = {"type": "structured", "path": dataset_path, "structured_data": structured}
                                logger.info(f"Using structured DataPart payload converted to: {dataset_path}")
                        except Exception as conv_err:
                            logger.error(f"Failed to process DataPart: {conv_err}")
                            dataset_info = {"type": "none", "path": None}
                            dataset_type = "unknown"

                    # If uploaded_file signals raw bytes content
                    elif isinstance(uploaded_file, dict) and uploaded_file.get("type") == "bytes":
                        # Save uploaded file content to temporary file
                        dataset_path = await self._create_temp_dataset_from_file_content(
                            uploaded_file["content"], uploaded_file.get("name", "uploaded_file")
                        )
                        dataset_type = "uploaded_file"
                        dataset_info = {"type": "uploaded_file", "path": dataset_path}  # Store for later use
                        logger.info(f"Using uploaded file: {uploaded_file.get('name')} -> {dataset_path}")

                    # If uploaded_file provides a URI (e.g., FileWithUri), attempt to handle or warn
                    elif isinstance(uploaded_file, dict) and uploaded_file.get("uri"):
                        logger.warning(f"URI-based file uploads not yet supported: {uploaded_file.get('uri')}")
                        dataset_info = {"type": "none", "path": None}
                        dataset_type = "unknown"

                    else:
                        # Fallback: try to detect structured payload inside the uploaded_file object
                        try:
                            # uploaded_file may be an object from SDK with attributes
                            if hasattr(uploaded_file, "data"):
                                structured = getattr(uploaded_file, "data")
                                dataset_path = await self._create_temp_dataset_from_structured_data(structured)
                                dataset_type = "datapart_structured"
                                dataset_info = {"type": "structured", "path": dataset_path, "structured_data": structured}
                                logger.info(f"Using structured uploaded_file converted to: {dataset_path}")
                            else:
                                logger.warning("Uploaded file format not recognized; no dataset available")
                                dataset_info = {"type": "none", "path": None}
                                dataset_type = "unknown"
                        except Exception as e:
                            logger.error(f"Error handling uploaded_file fallback: {e}")
                            dataset_info = {"type": "none", "path": None}
                            dataset_type = "unknown"
                else:
                    # Scenario 2 & 3: Try to extract from DataPart using enhanced function
                    dataset_info = self._extract_dataset_info(uploaded_files=uploaded_files)
                    dataset_type = dataset_info["type"]
                
                if dataset_type == "structured":
                    # Convert structured data to temporary file
                    dataset_path = await self._create_temp_dataset_from_structured_data(
                        dataset_info["structured_data"]
                    )
                    logger.info(f"Structured data converted to: {dataset_path}")
                    
                    # CRITICAL: Also enhance instructions to reference the CSV file
                    # This ensures the workflow planner gets enhanced instructions, not raw JSON
                    instructions = self._enhance_instructions_with_file_reference(
                        instructions, dataset_path, dataset_info["structured_data"]
                    )
                    logger.debug(f"Enhanced instructions for structured data: {instructions[:200]}...")
                
                elif dataset_type == "file_path":
                    # Use the extracted file path (could be from metadata or direct reference)
                    dataset_path = dataset_info["path"]
                    logger.info(f"Using dataset path: {dataset_path}")
                    
                    # If we have metadata (from structured JSON), extract analysis instructions
                    if "metadata" in dataset_info and dataset_info["metadata"]:
                        metadata = dataset_info["metadata"]
                        if "analysis_instruction" in metadata:
                            # Use the analysis instruction from metadata
                            instructions = metadata["analysis_instruction"]
                            logger.info(f"Using analysis instructions from metadata: {instructions[:100]}...")
                        elif "task" in metadata:
                            # Use task description as fallback
                            instructions = metadata.get("task", instructions)
                            logger.info(f"Using task description from metadata: {instructions[:100]}...")
                    
                elif dataset_type == "none":
                    # No dataset detected - return early with appropriate message
                    logger.warning("No dataset detected in message content")
                    no_dataset_message = """
# Analysis Report: No Dataset Detected

## Issue
No dataset was detected in the provided message. The analysis agent supports:

1. **File Uploads**: Attach CSV, Excel, or other data files via A2A protocol
2. **Structured Data**: Embed JSON data directly in the message
3. **File References**: Specify existing dataset file paths

## Expected Format Examples

### For Structured Data:
```
{"db_data": ["[{\"column1\": \"value1\", \"column2\": \"value2\"}]"]}
```

### For File References:
```
Dataset: my_data.csv
Instructions: Analyze this dataset...
```

## Next Steps
Please provide a dataset in one of the supported formats to proceed with analysis.
"""
                    
                    if trace:
                        trace.update(output={"status": "no_dataset", "message": no_dataset_message}, status_message="completed")
                    
                    return {
                        "id": str(uuid.uuid4()),
                        "result": {
                            "id": str(uuid.uuid4()),
                            "status": {"state": "completed"},
                            "analysis_result": no_dataset_message,
                            "session_id": message.context_id,
                            "dataset_info": {
                                "path": None,
                                "type": "none",
                                "uploaded_file": None,
                            },
                        },
                    }

            if not instructions:
                error_msg = "Missing analysis instructions"
                if trace:
                    trace.update(output={"error": error_msg}, status_message="failed")
                return self._create_error_response_a2a(message, error_msg)

            # CRITICAL FIX: Apply JSON conversion BEFORE workflow planning
            # BUT SKIP if we already extracted dataset_path from metadata OR if JSON conversion already happened
            logger.info("Checking if JSON workflow enhancement is needed...")
            
            # DEBUG: Show dataset_info content
            logger.info(f"DEBUG: dataset_info = {dataset_info}")
            logger.info(f"DEBUG: dataset_path = {dataset_path}")
            
            # CRITICAL FIX: Extract original CSV path from structured data if available
            original_csv_path = None
            if (dataset_info.get("type") == "structured" and 
                "structured_data" in dataset_info and 
                "dataset_path" in dataset_info["structured_data"]):
                original_csv_path = dataset_info["structured_data"]["dataset_path"]
                logger.info(f"DEBUG: Found original CSV path in structured data: {original_csv_path}")
            
            # STRONGER CSV DETECTION: Check both current path and original path from structured data
            is_direct_csv_file = (
                dataset_path and 
                dataset_path.endswith('.csv') and 
                Path(dataset_path).exists() and
                'temp_datasets' not in dataset_path and
                'structured_dataset_' not in dataset_path
            )
            
            is_original_csv_file = (
                original_csv_path and 
                original_csv_path.endswith('.csv') and 
                Path(original_csv_path).exists() and
                'temp_datasets' not in original_csv_path and
                'structured_dataset_' not in original_csv_path
            )
            
            # Skip JSON conversion if:
            # 1. We have a direct CSV file that exists on disk, OR
            # 2. We already have a file_path from metadata, OR  
            # 3. The dataset_path is already a converted CSV file (indicates JSON conversion already happened)
            already_converted = (
                dataset_path and 
                dataset_path.endswith('.csv') and 
                'structured_dataset_' in dataset_path and
                'temp_datasets' in dataset_path
            )
            
            has_file_path_metadata = (dataset_info.get("type") == "file_path" and 
                                    "metadata" in dataset_info and 
                                    dataset_info["metadata"])
            
            logger.info(f"DEBUG: is_direct_csv_file = {is_direct_csv_file}")
            logger.info(f"DEBUG: is_original_csv_file = {is_original_csv_file}")
            logger.info(f"DEBUG: already_converted = {already_converted}")
            logger.info(f"DEBUG: has_file_path_metadata = {has_file_path_metadata}")
            
            if (is_direct_csv_file or is_original_csv_file or has_file_path_metadata or already_converted):
                
                if is_direct_csv_file:
                    logger.info("Skipping JSON conversion - direct CSV file detected")
                elif is_original_csv_file:
                    logger.info("Skipping JSON conversion - original CSV file detected in structured data")
                elif already_converted:
                    logger.info("Skipping JSON conversion - dataset already converted to CSV in previous step")
                else:
                    logger.info("Skipping JSON conversion - already extracted dataset_path from metadata")
                final_instructions = instructions
                final_dataset_path = dataset_path
                enhanced_dataset_path = dataset_path  # Set this for consistency in error handling
                
                # CRITICAL: Use original CSV path for text analytics inspection
                original_path_for_inspection = original_csv_path if original_csv_path else dataset_path
                
            else:
                # Apply JSON conversion for other cases
                logger.info("Applying JSON workflow enhancement...")
                enhanced_instructions, enhanced_dataset_path = await self._enhance_instructions_with_json_conversion(
                    instructions, dataset_path
                )
                
                # Use enhanced instructions and dataset path for workflow planning
                final_instructions = enhanced_instructions
                final_dataset_path = enhanced_dataset_path
                
                # For JSON conversion case, original path might still be useful for inspection
                original_path_for_inspection = original_csv_path if original_csv_path else dataset_path
                
                if enhanced_dataset_path != dataset_path:
                    logger.info(f"JSON conversion applied: {dataset_path} → {enhanced_dataset_path}")
                    logger.info("Workflow planning will use converted CSV file")
                else:
                    logger.debug("No JSON conversion needed - using original instructions")

            # Update agent state
            session_id = message.context_id
            if not self.current_state:
                await self.start_session()
                self.current_state["session_id"] = session_id

            self.current_state["dataset_path"] = final_dataset_path
            self.current_state["current_instruction"] = final_instructions


            # Add workflow planning span
            planning_span = None
            if trace:
                planning_span = trace.span(
                    name="Workflow_Planning",
                    input={
                        "instructions": final_instructions,
                        "dataset_path": final_dataset_path,
                        "json_conversion_applied": enhanced_dataset_path != dataset_path,
                    },
                )

            # Dynamically analyze instruction and plan workflow using enhanced inputs
            # Pass original CSV path for text analytics inspection
            workflow_plan = await self._analyze_instruction_and_plan_workflow(
                final_instructions, final_dataset_path, original_path_for_inspection
            )
            self.current_state["workflow_plan"] = workflow_plan

            if planning_span:
                planning_span.update(output=workflow_plan)
                planning_span.end()

            # Add analysis execution span
            execution_span = None
            if trace:
                execution_span = trace.span(
                    name="Analysis_Execution",
                    input={
                        "workflow_plan": workflow_plan,
                        "instructions": instructions,
                    },
                )

            # Execute analysis using intelligent workflow plan with final processed inputs
            analysis_result = await self._execute_analysis_with_workflow(
                final_instructions, final_dataset_path, workflow_plan, trace
            )

            if execution_span:
                execution_span.update(
                    output={
                        "result_length": len(analysis_result),
                        "status": "completed",
                    }
                )
                execution_span.end()

            # Store results with workflow metadata using final processed inputs
            self.current_state["analysis_results"].append(
                {
                    "instruction": final_instructions,
                    "result": analysis_result,
                    "timestamp": datetime.now().isoformat(),
                    "workflow_plan": workflow_plan,
                    "dataset_info": {
                        "path": final_dataset_path,
                        "type": dataset_type,
                        "uploaded_file": (getattr(uploaded_file, 'name', None) or getattr(uploaded_file, 'filename', None)) if file_parts else None,
                        "json_conversion_applied": enhanced_dataset_path != dataset_path,
                    }
                }
            )

            # Finalize trace
            if trace:
                trace.update(
                    output={
                        "status": "completed",
                        "analysis_result_length": len(analysis_result),
                        "workflow_metadata": {
                            "planned_tools": workflow_plan.get("suggested_tools", []) if isinstance(workflow_plan, dict) else [],
                            "complexity_score": workflow_plan.get(
                                "complexity_score", 0.5
                            ) if isinstance(workflow_plan, dict) else 0.5,
                            "estimated_duration": workflow_plan.get(
                                "estimated_duration", 60
                            ) if isinstance(workflow_plan, dict) else 60,
                        },
                    },
                    status_message="success",
                )

            # Return success response
            return {
                "id": str(uuid.uuid4()),
                "result": {
                    "id": str(uuid.uuid4()),
                    "status": {"state": "completed"},
                    "analysis_result": analysis_result,
                    "session_id": session_id,
                    "dataset_info": {
                        "path": final_dataset_path,
                        "type": dataset_type,
                        "uploaded_file": (getattr(uploaded_file, 'name', None) or getattr(uploaded_file, 'filename', None)) if file_parts else None,
                        "json_conversion_applied": enhanced_dataset_path != dataset_path,
                    },
                    "workflow_metadata": {
                        "planned_tools": workflow_plan.get("suggested_tools", []) if isinstance(workflow_plan, dict) else [],
                        "complexity_score": workflow_plan.get("complexity_score", 0.5) if isinstance(workflow_plan, dict) else 0.5,
                        "estimated_duration": workflow_plan.get(
                            "estimated_duration", 60
                        ) if isinstance(workflow_plan, dict) else 60,
                    },
                },
            }

        except Exception as e:
            logger.error(f"Error handling instruction: {e}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            if trace:
                trace.update(output={"error": str(e)}, status_message="error")
            return self._create_error_response_a2a(message, str(e))

    async def _analyze_instruction_and_plan_workflow(
        self, instructions: str, dataset_path: str, original_dataset_path: str = None
    ) -> Dict[str, Any]:
        """
        Dynamically analyze orchestrator's instruction and plan intelligent workflow
        using available tools from MCP servers
        """
        try:
            # Get available tools dynamically
            available_tools = []
            tool_descriptions = {}

            for tool in self.tools:
                tool_info = {
                    "name": tool.name,
                    "description": getattr(
                        tool, "description", "No description available"
                    ),
                    "parameters": str(getattr(tool, "args_schema", {})),
                }
                available_tools.append(tool_info)
                tool_descriptions[tool.name] = tool_info["description"]

            # DEBUG: Check if text analytics tools are available to LLM
            text_tools = [t['name'] for t in available_tools if any(keyword in t['name'] for keyword in ['text', 'extract', 'merge'])]
            logger.info(f"DEBUG: Text analytics tools in available_tools: {text_tools}")
            
            # Use original dataset path for inspection if available (for text analytics), otherwise use current path
            inspection_path = original_dataset_path if original_dataset_path else dataset_path
            logger.info(f"DEBUG: Using dataset path for inspection: {inspection_path}")
            dataset_context = await self._quick_dataset_inspection(inspection_path)
            
            # DEBUG: Check if Message column analysis is being generated
            message_analysis = dataset_context.get('text_content_analysis', {}).get('Message', {})
            if message_analysis:
                logger.info(f"DEBUG: Message column detected with '{message_analysis.get('extraction_potential', 'unknown')}' extraction potential")
            else:
                logger.info("DEBUG: No Message column analysis found in dataset_context")
            
            # Check if no dataset is available
            if (dataset_path is None or 
                dataset_path == "No explicit dataset provided" or 
                dataset_context.get("file_type") == "Unknown" and dataset_context.get("error")):
                
                logger.info("No dataset available - returning no-dataset workflow plan")
                return {
                    "analysis_summary": "No dataset provided - cannot perform data analysis",
                    "suggested_tools": [],  # No tools needed
                    "execution_strategy": [],
                    "complexity_score": 0.0,
                    "estimated_duration": 5,
                    "expected_outputs": ["no_dataset_message"],
                    "reasoning": "No dataset was provided or detected in the message content",
                    "requires_dataset": True
                }

            # Use LLM to analyze instruction and create workflow plan
            planning_prompt = f"""
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.
You are a highly analytically intelligent expert data analysis workflow planner with deep domain expertise across multiple fields. 
Your ONLY task is to design a workflow plan — do not perform or simulate any analysis execution. 
The plan must be tailored to the CURRENT dataset’s structure and metadata. 
Do not assume columns or fields beyond what is provided.

ORCHESTRATOR INSTRUCTION:
{instructions}

DATASET CONTEXT:
{json.dumps(dataset_context, indent=2)}

AVAILABLE TOOLS FROM MCP SERVERS:
{json.dumps(available_tools, indent=2)}

WORKFLOW PLANNING METHODOLOGY:

STAGE 1 - INPUT INGESTION & INSTRUCTION ENHANCEMENT:
Understand the user's request, inferring the true intent behind it and consider the following: 
- If instructions are SPECIFIC, preserve their exact requirements.
- If instructions are GENERIC, derive what meaningful analysis can be done given the dataset contents

STAGE 2 - DATASET PROFILING:
Interpret the dataset structure, data types, and characteristics to identify what analysis is feasible, contextually and analytically intelligent.
Pay attention to any text content analysis that describes analysis opportunities and extraction potential.
CRITICAL: If text columns have high extraction potential (locations, entities, patterns), plan to use text analytics tools like 'analyze_text_content' and 'extract_structured_data_from_text' to unlock hidden insights.

STAGE 3 - QUERY PARSING & INTENT RECOGNITION:
Infer the business or analytical need behind the instruction, not just its literal words.
Go beyond basic summary statistics to discover meaningful patterns that drive decisions.
Consider all data sources - both structured columns and any text content with high extraction potential.
TEXT ANALYTICS INTELLIGENCE: If the dataset contains unstructured text with operational data, traffic information, location references, or temporal patterns, plan to extract this intelligence using text analytics tools.

STAGE 4 - DYNAMIC TOOL PLAN CREATION:
Select the most appropriate tools and sequence them logically to transform the dataset into interpretable and actionable insights.
Choose tools based on the data characteristics and analysis opportunities identified in the dataset context.
INTELLIGENT TOOL SELECTION: Include text analytics tools when text content analysis indicates high extraction potential - this unlocks valuable insights that standard statistical analysis cannot reveal.

At each stage, specify how insights will be inferred and how results will be interpreted, not just calculated. 
DO NOT run the tools or provide actual outputs — this step is planning only.

Your workflow plan should:
- Begin with validation and profiling of the data
- Progress through exploration, interpretation, and deeper inference
- Plan which tools will later be used to generate evidence, and describe how insights would be inferred from that evidence
- When instructions are generic, derive what meaningful analysis can be done given the dataset contents
- When instructions are specific, precisely honor the stated requirements while suggesting complementary analysis
- Produce outputs that are actionable and decision-ready, contextually and analytically intelligent
- Avoid assumptions about schema beyond the dataset context provided
- Stay strictly within a planning role, no execution
- LEVERAGE TEXT ANALYTICS: When text columns show high extraction potential, plan to use text analytics tools to extract structured insights like locations, entities, patterns, and operational intelligence

Return ONLY the workflow plan in structured JSON (no prose, no explanations outside JSON) for example:

{{
    "intent_analysis": "Interpretation of the user's real needs and intent",
    "data_compatibility": "Assessment of how well the dataset supports this request",
    "analysis_approach": "Best analytical methodology for this dataset and question",
    "suggested_tools": ["ordered_list_of_recommended_tools"],
    "execution_strategy": [
        {{"step": 1, "tool": "tool_name", "purpose": "why this tool is needed and what it will uncover", "expected_output": "evidence this step produces"}},
        {{"step": 2, "tool": "tool_name", "purpose": "how it builds on the previous output to interpret or infer patterns", "expected_output": "insight or evidence"}}
    ],
    "complexity_score": 0.7,
    "estimated_duration": 120,
    "planned_deliverables": ["concrete outputs including inferred insights and interpretations"],
    "planning_reasoning": "Explain why this workflow design will maximize insights and actionable outcomes",
    "success_criteria": "How to confirm the workflow delivered valid evidence and meaningful interpretations"
}}

Be thorough, insightful, precise, contextually and analytically intelligent. Always think in terms of evidence → interpretation → inferred insights → recommendations. 
REMEMBER: You are planning only, not executing any analysis.
"""



            # Get workflow plan from LLM with rate limiting and proper tracing
            llm_messages = [HumanMessage(content=planning_prompt)]

            # Add Langfuse tracing specifically for workflow planning
            config = {}
            if self.langfuse_handler:
                config["callbacks"] = [self.langfuse_handler]
                # Add run name for better tracing
                config["run_name"] = "Workflow_Planning_LLM_Call"
                config["tags"] = ["workflow_planning", "llm_call"]

            response = await self._rate_limited_llm_call(
                llm_messages, "workflow planning"
            )

            try:
                # Extract JSON from response
                response_content = response.content
                if "```json" in response_content:
                    json_start = response_content.find("```json") + 7
                    json_end = response_content.find("```", json_start)
                    json_str = response_content[json_start:json_end].strip()
                else:
                    json_str = response_content.strip()

                workflow_plan = json.loads(json_str)
                
                # CRITICAL: Always include dataset inspection results for React agent
                workflow_plan["dataset_inspection"] = dataset_context

                logger.info(
                    f"Generated dynamic workflow plan: {workflow_plan.get('analysis_summary', 'No summary')}"
                )
                logger.info(
                    f"Suggested tools: {workflow_plan.get('suggested_tools', [])}"
                )

                return workflow_plan

            except json.JSONDecodeError as e:
                logger.error(f"WORKFLOW PLANNING JSON PARSING FAILED: {e}")
                logger.error(f"Raw LLM response that failed to parse: {response_content[:500]}...")
                logger.warning("FALLBACK ACTIVATED: Using standard workflow plan due to LLM JSON parsing failure")
                # Return a fallback workflow plan instead of raising an error
                workflow_plan = {
                    "intent_analysis": "Analysis request detected",
                    "data_compatibility": "Using standard workflow",
                    "analysis_approach": "Comprehensive data analysis",
                    "suggested_tools": ["load_and_analyze_csv", "generate_eda_plots"],
                    "execution_strategy": [
                        {"step": 1, "tool": "load_and_analyze_csv", "purpose": "Load and analyze dataset", "expected_output": "Basic statistics and insights"},
                        {"step": 2, "tool": "generate_eda_plots", "purpose": "Create visualizations", "expected_output": "Charts and plots"}
                    ],
                    "complexity_score": 0.5,
                    "estimated_duration": 120,
                    "planned_deliverables": ["Data analysis report", "Visualizations"],
                    "planning_reasoning": "FALLBACK WORKFLOW: LLM workflow planning failed JSON parsing - using standard analysis workflow",
                    "success_criteria": "Basic analysis completed with visualizations",
                    "dataset_inspection": dataset_context,
                    "is_fallback_workflow": True,
                    "fallback_reason": "LLM JSON parsing error"
                }
                logger.info("FALLBACK WORKFLOW PLAN ACTIVATED: Standard analysis workflow will be used")
                return workflow_plan

        except Exception as e:
            logger.error(f"CRITICAL SYSTEM ERROR: Workflow planning completely failed: {e}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            logger.error("EMERGENCY FALLBACK ACTIVATED: All workflow planning systems failed")
            # Return emergency fallback workflow instead of raising error
            return {
                "intent_analysis": "Emergency analysis mode",
                "data_compatibility": "Using basic workflow",
                "analysis_approach": "Standard data analysis",
                "suggested_tools": ["load_and_analyze_csv", "generate_eda_plots"],
                "execution_strategy": [
                    {"step": 1, "tool": "load_and_analyze_csv", "purpose": "Load and analyze dataset", "expected_output": "Basic statistics"},
                    {"step": 2, "tool": "generate_eda_plots", "purpose": "Create visualizations", "expected_output": "Charts"}
                ],
                "complexity_score": 0.5,
                "estimated_duration": 120,
                "planned_deliverables": ["Analysis report", "Visualizations"],
                "planning_reasoning": "EMERGENCY FALLBACK WORKFLOW: Complete workflow planning system failure - using minimal analysis workflow",
                "success_criteria": "Basic analysis completed",
                "dataset_inspection": {},  # Empty fallback for emergency case
                "is_fallback_workflow": True,
                "fallback_reason": "Complete system error"
            }

    async def _quick_dataset_inspection(self, dataset_path: str) -> Dict[str, Any]:
        """Quick inspection of dataset to provide context for workflow planning"""
        try:
            if dataset_path.endswith(".csv"):
                # Read just a sample for planning
                df = pd.read_csv(dataset_path, nrows=100)
                
                # Enhanced text content analysis for intelligent planning
                text_content_analysis = {}
                text_columns = df.select_dtypes(include=["object"]).columns.tolist()[:5]
                
                for col in text_columns[:3]:  # Limit to first 3 text columns
                    if not df[col].dropna().empty:
                        text_samples = df[col].dropna().head(3).tolist()
                        avg_length = float(df[col].str.len().mean()) if not df[col].dropna().empty else 0
                        has_patterns = self._detect_text_patterns(df[col])
                        
                        # Generate intelligent analysis opportunities description
                        analysis_opportunities = []
                        if has_patterns:
                            analysis_opportunities.append("consistent structure detected - likely contains extractable data entities")
                        if avg_length > 50:
                            analysis_opportunities.append("substantial content - may contain multiple data points per entry")
                        if any(keyword in str(sample).lower() for sample in text_samples for keyword in ['time', 'date', 'location', 'type', 'status', 'incident', 'event', 'report']):
                            analysis_opportunities.append("operational data detected - potential for time/location/event extraction")
                        if has_patterns and avg_length > 30:
                            analysis_opportunities.append("rich structured information likely embedded - high value for data extraction")
                        
                        text_content_analysis[col] = {
                            "samples": text_samples,
                            "avg_length": avg_length,
                            "analysis_opportunities": analysis_opportunities,
                            "extraction_potential": "high" if (has_patterns and avg_length > 30) else "moderate" if has_patterns else "low",
                            "sample_count": len(text_samples)
                        }
                
                return {
                    "file_type": "CSV",
                    "sample_rows": int(len(df)),
                    "columns": df.columns.tolist()[:10],  # Limit for prompt size
                    "column_count": int(len(df.columns)),
                    "numeric_columns": df.select_dtypes(
                        include=["number"]
                    ).columns.tolist()[:5],
                    "text_columns": text_columns,
                    "text_content_analysis": text_content_analysis,  # NEW: Text content samples for LLM planning
                    "has_missing_values": bool(
                        df.isnull().any().any()
                    ),  # Explicitly convert to bool
                }
            else:
                return {
                    "file_type": "Unknown",
                    "note": "Limited inspection available for non-CSV files",
                }
        except Exception as e:
            logger.warning(f"Could not inspect dataset {dataset_path}: {e}")
            return {
                "file_type": "Unknown",
                "error": f"Inspection failed: {str(e)}",
            }

    def _detect_text_patterns(self, text_series) -> bool:
        """Simple detection of structured patterns in text - no domain assumptions"""
        try:
            sample = text_series.dropna().head(10)
            if len(sample) == 0:
                return False
            
            # Basic heuristics for structured vs free-form text
            avg_length = sample.str.len().mean()
            length_std = sample.str.len().std()
            
            # Structured text tends to have consistent length and common patterns
            length_consistency = length_std < avg_length * 0.5 if avg_length > 0 else False
            sufficient_samples = len(sample) >= 3
            reasonable_length = avg_length > 10  # Not just short codes
            
            return length_consistency and sufficient_samples and reasonable_length
        except Exception:
            return False

    async def _execute_analysis_with_workflow(
        self,
        instructions: str,
        dataset_path: str,
        workflow_plan: Dict[str, Any],
        parent_trace=None,
    ) -> str:
        """Execute analysis using React agent with intelligent workflow guidance"""
        try:
            # SAFETY CHECK: Ensure workflow_plan is a dictionary
            if not isinstance(workflow_plan, dict):
                logger.warning(f"Invalid workflow_plan type: {type(workflow_plan)}. Using fallback.")
                workflow_plan = {
                    "suggested_tools": ["load_and_analyze_csv", "generate_eda_plots"],
                    "analysis_summary": "Fallback analysis workflow",
                    "complexity_score": 0.5,
                    "estimated_duration": 120,
                    "is_fallback_workflow": True,
                    "fallback_reason": "Invalid workflow plan type"
                }
            
            # Check if we're using a fallback workflow and log it prominently
            if workflow_plan.get("is_fallback_workflow", False):
                fallback_reason = workflow_plan.get("fallback_reason", "unknown")
                logger.warning(f"EXECUTING FALLBACK WORKFLOW: Reason - {fallback_reason}")
                logger.info(f"Fallback workflow tools: {workflow_plan.get('suggested_tools', [])}")
            else:
                logger.info("EXECUTING INTELLIGENT WORKFLOW: Using LLM-generated workflow plan")
                logger.info(f"Planned tools: {workflow_plan.get('suggested_tools', [])}")
            
            # ENHANCED JSON WORKFLOW: Automatically detect and convert JSON in instructions
            enhanced_instructions, enhanced_dataset_path = await self._enhance_instructions_with_json_conversion(
                instructions, dataset_path
            )
            
            # Use enhanced instructions and dataset path for the rest of the workflow
            if enhanced_dataset_path != dataset_path:
                logger.info(f"JSON workflow enhancement applied: {enhanced_dataset_path}")
                dataset_path = enhanced_dataset_path
                instructions = enhanced_instructions
            # JSON conversion is now handled earlier in the workflow, so we use the inputs as-is
            logger.info(f"Executing analysis with dataset: {dataset_path}")
            logger.info(f"Instructions already enhanced and ready for execution")
            
            # Initialize workflow progress tracker
            planned_tools = workflow_plan.get("suggested_tools", [])
            progress_tracker = WorkflowProgressTracker(planned_tools)
            
            # Store tracker in agent state for persistence
            self.current_state["workflow_tracker"] = progress_tracker
            
            # Create workflow execution span
            workflow_span = None
            if parent_trace:
                workflow_span = parent_trace.span(
                    name="Workflow_Execution_Details",
                    input={
                        "instructions": instructions,
                        "dataset_path": dataset_path,
                        "suggested_tools": planned_tools,
                    },
                )

            # Get rich dataset inspection results from workflow planning
            dataset_inspection = workflow_plan.get("dataset_inspection", {})
            
            # Extract key workflow plan components directly (no reformatting needed)
            intent_analysis = workflow_plan.get("intent_analysis", "General data exploration and insights discovery")
            analysis_approach = workflow_plan.get("analysis_approach", "Comprehensive data analysis")
            planning_reasoning = workflow_plan.get("planning_reasoning", "Systematic analytical approach")
            execution_strategy = workflow_plan.get("execution_strategy", [])
            success_criteria = workflow_plan.get("success_criteria", "Meaningful insights generated")

            # Intelligent analysis prompt that preserves planning context
            safe_default_first_tool = planned_tools[0] if planned_tools else ('load_and_analyze_csv' if (dataset_path and str(dataset_path).lower().endswith('.csv')) else 'llm_dataset_detector')

            # Analysis Output Template: formatting instructions for fact-based analytical reporting
            analysis_output_template = f"""
Produce an analytically intelligent report in three sections: Descriptive Statistics, Analysis Results, and Executive Summary.

## Descriptive Statistics
- Dataset size (rows, columns)
- Column names and inferred types
- Missing values and duplicate counts
- Key but analytically intelligent and meaningful numeric summaries (mean, median, min, max) for relevant numeric columns except for numeric unique identifiers, latitude or longitude columns.

## Analysis Results
For EACH visualization produced, include either:
- A single-plot section:
  ### <Plot Title>
  ![Plot Title](<filename.png>)
  **Overview:** one sentence describing the plot.
  **Key Findings:** concise bullets with exact numeric evidence.
  **Actionable Insights:** bullets linking findings to recommended actions with numeric justification.
OR
- A grouped section (when multiple files are intentionally grouped):
  ### Group: <Group Title> (includes: ![Plot Title](<filename1.png>), ![Plot Title](<filename2.png>), ...)
  **Overview:** one sentence describing why these are grouped.
  **Key Findings:** consolidated bullets with file-specific evidence labeled in parentheses.
  **Actionable Insights:** consolidated recommendations with which file supports each.

If a plot is inferred (tool output truncated), mark it "[inferred]" and state what was inferred and why.

## Text-Extracted Insight
- Consolidate text-analysis findings; provide analytically intelligent and meaningfully representative examples and counts.

## Executive Summary
- Bullet points summarizing the most important findings and top recommended actions with expected impact estimates.

CRITICAL: Use exact numbers from analyses. Avoid unsupported comparisons or marketing language. Keep language precise and factual.
"""


            # Format execution strategy for clear presentation
            strategy_text = ""
            for i, step in enumerate(execution_strategy[:8], 1):
                tool_name = step.get('tool', 'unknown')
                purpose = step.get('purpose', 'Analysis component')
                expected_output = step.get('expected_output', 'Results')
                strategy_text += f"{i}. {tool_name}: {purpose}\n   Expected: {expected_output}\n"

            analysis_prompt = f"""
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.
You are also a highly analytically intelligent and precise expert data analyst conducting a comprehensive analysis.
You have access to sophisticated analytical tools and intelligent workflow planning to guide your analysis.

DATASET CONTEXT:
Dataset File Path: {dataset_path}
Dataset Overview: {dataset_inspection}

RESEARCH QUESTION:
{instructions}

WORKFLOW PLANNING INTELLIGENCE:
ANALYSIS INTENT: {intent_analysis}
ANALYTICAL APPROACH: {analysis_approach}
EXECUTION STRATEGY: {strategy_text}
PLANNING REASONING: {planning_reasoning}
SUCCESS CRITERIA: {success_criteria}

------------------------------------------------------------
INTELLIGENT VISUALIZATION GUIDELINES
------------------------------------------------------------
When generating visualizations, apply these principles contextually:
- **Spatial Data Intelligence**: Use Latitude/Longitude ONLY for scatterplots or spatial maps showing geographic distribution. Never use coordinates in bar charts, histograms, boxplots, or correlation heatmaps.
- **Spatial Context Enhancement**: Always color or group spatial scatterplots by relevant categories (Type, Severity, Time Period) to reveal meaningful patterns.
- **Coordinate Exclusion**: Exclude Latitude/Longitude from correlation analyses, statistical summaries, and all non-spatial visualizations.
- **Intelligent Fallbacks**: If coordinates are detected in a non-spatial context, switch to a spatial plot or exclude them intelligently.
- **Meaningful Correlations**: Avoid nonsensical relationships such as latitude vs longitude or correlations involving ID/index columns.

VISUALIZATION & ANALYSIS GUIDANCE (to include in analysis_prompt):

- Spatial rule: Use Latitude/Longitude ONLY for spatial scatterplots/maps. Do NOT use coordinates in bar/line/histogram/boxplot/correlation outputs for non-spatial analysis.
- Plot axis rule: Always ensure one axis is a quantitative metric (count, mean, rate); do not plot identifiers or near-constant columns.
- Correlation filter: Suppress/report only correlations with |r| >= 0.20 unless user requests exploratory output.
- Plot differentiation: If multiple files share a title but differ by category/slice (e.g., "Hourly Distribution - Roadwork", "Hourly Distribution - Heavy Traffic"), treat each file as a distinct visualization. You may group them, but list every filename and provide file-specific evidence when needed.
- Grouping guideline: Group only when plots share the exact metric and primary variable across slices. When grouping, list all filenames and attach findings to filenames in parentheses for specificity.
- Completeness check: Collect all .png filenames from tool outputs. Each filename must appear either as its own section or inside a grouped section. Do not drop files for perceived redundancy.
- Text integration: Immediately integrate text-analysis outputs into structured analysis; cross-reference entities/locations from text with numerical trends and label evidence.
- Precision formatting: Round percentages to 1 decimal place; include counts (n=...) for key findings.

Use these rules to guide analysis and visualization choices. The `analysis_output_template` strictly controls final formatting.

------------------------------------------------------------
EXECUTION GUIDELINES
------------------------------------------------------------
- Follow the analytical approach systematically with precision and reasoning.
- Consolidate and interpret results intelligently and meaningfully in context of the analysis intent.
- Connect findings directly to the research question and success criteria.
- Apply analytical reasoning at each stage to produce meaningful, data-driven insights.
- Do not include nonsense outputs of mathematical operations on numerical unique identifiers or latitude or longitude like mean/std/min/max.
------------------------------------------------------------
TEXT ANALYTICS INTEGRATION
------------------------------------------------------------
- When using text analytics tools (e.g., analyze_text_content, extract_structured_data_from_text), immediately integrate extracted insights into your current analytical context.
- Treat text-derived insights as first-class analytical data that deepen understanding of trends, patterns, and relationships in the structured dataset.
- Cross-reference textual findings (locations, entities, operational patterns) with numerical or categorical results to identify correlations, contradictions, or reinforcement.
- Text insights should enrich your interpretation of statistical findings, not appear as separate or unrelated data.

------------------------------------------------------------
COMPREHENSIVE PLOT COVERAGE
------------------------------------------------------------
- Create analysis sections for ALL generated plots; ensure every visualization receives explicit analytical discussion.
- Be especially vigilant with temporal analysis tools, which may generate multiple plots (hourly, daily, heatmaps, incident-type patterns). #maybe too specific.
- Treat each plot file as a distinct analytical slice, even if filenames or titles appear similar.
- You may group related plots for readability, but every individual filename must still be listed or referenced in the report.
- Do NOT infer missing plots unless tool output explicitly indicates them.

------------------------------------------------------------
VISUALIZATION COMPLETENESS CHECK
------------------------------------------------------------
Before finalizing the report:
1. Review ALL tool outputs referencing .png or visualization files.
2. Ensure each filename appears either as its own ### section or within a grouped section.
3. For temporal analyses, confirm that all relevant variations (hourly, daily, heatmap, incident-type) are represented.
4. Cross-check tool output mentions vs. your ### sections — counts must match exactly.

------------------------------------------------------------
GROUPING DECISION GUIDELINE
------------------------------------------------------------
- Group if: same plot type + same metric + different slices (e.g., hourly distributions across multiple incident types).
- Do not group if: plots represent different metrics, primary variables, or distinct insights.
- When in doubt, default to individual sections OR include a grouped section with explicit reference to each filename.

------------------------------------------------------------
PLOT DIFFERENTIATION RULE
------------------------------------------------------------
If multiple plots share similar names (e.g., all contain "Hourly Distribution") but differ by category, type, or filter condition 
(e.g., "Roadwork - Hourly Distribution" vs "Heavy Traffic - Hourly Distribution"), treat them as separate visualizations or grouped subplots 
that must still be explicitly listed in the report.

When grouping:
- Include every file name within the group title.
- Explicitly state key findings for each category represented.
- Never collapse category-specific hourly or temporal plots into one generic section unless all categories are identical.

------------------------------------------------------------
ANALYSIS OUTPUT TEMPLATE
------------------------------------------------------------
{analysis_output_template}

------------------------------------------------------------
EXECUTION REQUIREMENT
------------------------------------------------------------
After conducting the analysis using the planned tools, synthesize and write the final comprehensive report following the 
ANALYSIS OUTPUT TEMPLATE exactly. This is your final output — a complete, properly formatted analytical report.

STARTING INSTRUCTION:
Begin the analysis following the planned approach. Start with: {safe_default_first_tool}

Remember: You have intelligent workflow guidance above — apply it thoughtfully and analytically, not mechanically. 
End with a complete, formatted report using the template above.
"""


            # Execute React agent with progress tracking
            enhanced_result = await self._execute_react_single_run(
                analysis_prompt, dataset_path, progress_tracker, parent_trace
            )

            if workflow_span:
                workflow_span.update(
                    output={
                        "response_length": len(enhanced_result),
                        "tools_executed": len(progress_tracker.executed_tools),
                        "workflow_complete": progress_tracker.is_workflow_complete()[0],
                    }
                )
                workflow_span.end()

            # Save the complete analysis to a Markdown file
            await self._save_analysis_report_with_workflow(
                enhanced_result, dataset_path, instructions, workflow_plan
            )

            return enhanced_result

        except Exception as e:
            logger.error(f"Error in workflow execution: {e}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return f"Analysis failed: {str(e)}"

    async def _execute_react_single_run(self, initial_prompt: str, dataset_path: str,
                                       progress_tracker: WorkflowProgressTracker,
                                       parent_trace=None) -> str:
        """Single-run React executor: call the react agent once and produce final enhanced report.

        This function intentionally removes the multi-iteration loop and iteration history
        to simplify the workflow while preserving canonical block selection and tool-output
        appendix behavior.
        """

        messages = [HumanMessage(content=initial_prompt)]

        # Configure React agent execution
        config = {"recursion_limit": 50}
        if self.langfuse_handler:
            config.setdefault("callbacks", []).append(self.langfuse_handler)

        try:
            response = await self._rate_limited_react_execution(messages, config, "analysis_single_run")
        except Exception as e:
            logger.error(f"React execution failed: {e}")
            return f"Analysis failed: {e}"

        if not response or "messages" not in response:
            return "Analysis failed: No response from React agent"

        messages_list = response["messages"]

        # Record executed tools in the compact tracker
        executed_tools = self._extract_executed_tools_from_messages(messages_list)
        for t in executed_tools:
            progress_tracker.record_tool_execution(t)

        # INTELLIGENT EXTRACTION: Get the React agent's final comprehensive report
        # Since the LLM already has all context and was instructed to format using the template,
        # we simply extract its final response - no need for redundant synthesis
        final_analysis = self._extract_final_analysis_from_react_messages(messages_list)

        # Save pure messages_list for debugging if needed
        await self._save_messages_list_debug(messages_list, dataset_path)

        # The React agent already produced the final formatted report - return it directly
        logger.info(f"Single-run execution completed. Tools executed: {len(executed_tools)}")
        return final_analysis

    def _extract_final_analysis_from_react_messages(self, messages_list: List[Any]) -> str:
        """Extract the React agent's final analysis response intelligently"""
        try:
            # Look for the final AI message that contains the formatted report
            # The React agent was instructed to end with a complete formatted report
            for msg in reversed(messages_list):
                if (hasattr(msg, 'type') and getattr(msg, 'type') == 'ai' and 
                    hasattr(msg, 'content') and msg.content):
                    
                    content = self._normalize_message_content(msg.content)
                    
                    # Check if this looks like a final formatted report
                    has_sections = ("## Descriptive Statistics" in content or 
                                  "## Analysis Results" in content or 
                                  "## Executive Summary" in content)
                    
                    is_substantial = len(content.strip()) > 200
                    
                    if has_sections and is_substantial:
                        logger.info("Found React agent's final formatted report")
                        return content
                    elif is_substantial:
                        # Even if not perfectly formatted, use the most substantial final response
                        logger.info("Using React agent's final substantial response")
                        return content
            
            # Fallback: use the last AI message regardless
            for msg in reversed(messages_list):
                if (hasattr(msg, 'type') and getattr(msg, 'type') == 'ai' and 
                    hasattr(msg, 'content') and msg.content):
                    content = self._normalize_message_content(msg.content)
                    if len(content.strip()) > 50:
                        logger.warning("Using last AI message as fallback analysis")
                        return content
                        
            return "Analysis completed but no final report found in React agent messages."
            
        except Exception as e:
            logger.error(f"Error extracting final analysis: {e}")
            return "Analysis completed but error occurred extracting final report."

    async def _save_messages_list_debug(self, messages_list: List[Any], dataset_path: str):
        """Save the pure messages_list as a timestamped MD file for debugging and transparency"""
        try:
            # Create report directory if it doesn't exist
            report_dir = self.config.reports_path
            report_dir.mkdir(parents=True, exist_ok=True)

            # Generate timestamped filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            try:
                dataset_name = Path(dataset_path).stem if dataset_path else "no_dataset"
            except Exception:
                dataset_name = "no_dataset"
            
            debug_filename = f"messages_debug_{dataset_name}_{timestamp}.md"
            debug_path = report_dir / debug_filename

            # Build debug content
            debug_content = f"""# React Agent Messages Debug Log

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Dataset:** {dataset_path}
**Total Messages:** {len(messages_list)}

---

"""

            for i, msg in enumerate(messages_list, 1):
                msg_type = getattr(msg, 'type', 'unknown')
                msg_content = self._normalize_message_content(getattr(msg, 'content', ''))
                
                # Save complete message content without truncation
                debug_content += f"""## Message {i}: {msg_type.upper()}

**Content Length:** {len(msg_content)} characters

```
{msg_content}
```

---

"""

            # Save to file
            with open(debug_path, "w", encoding="utf-8") as f:
                f.write(debug_content)

            logger.info(f"Messages debug log saved to: {debug_path}")

        except Exception as e:
            logger.warning(f"Failed to save messages debug log: {e}")

    async def _save_analysis_report_with_workflow(
        self,
        analysis_result: str,
        dataset_path: str,
        instructions: str,
        workflow_plan: Dict[str, Any],
    ):
        """Save analysis report with workflow planning metadata"""
        try:
            # Create report directory if it doesn't exist
            report_dir = self.config.reports_path
            # Use parents=True to ensure nested directories are created on Windows
            report_dir.mkdir(parents=True, exist_ok=True)

            # Generate timestamped filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Defensive handling: dataset_path may be None or invalid
            try:
                dataset_name = Path(dataset_path).stem
            except Exception:
                dataset_name = "no_dataset"
            report_filename = f"analysis_report_{dataset_name}_{timestamp}.md"
            report_path = report_dir / report_filename

            # Create comprehensive Markdown report with workflow metadata
            # Defensive session id read
            session_id = None
            try:
                session_id = self.current_state.get("session_id") if self.current_state else None
            except Exception:
                session_id = None

            report_content = f"""{analysis_result}"""

            # Save to file
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(report_content)

            logger.info(
                f"Analysis report with workflow metadata saved to: {report_path}"
            )

        except Exception as e:
            logger.error(f"Error saving analysis report with workflow: {e}")
            # Don't degrade to simple report - this indicates a system problem that should be fixed
            raise

    def _create_error_response_a2a(
        self, original_message: Message, error: str
    ) -> Dict[str, Any]:
        """Create standardized error response using A2A format"""
        return {
            "id": str(uuid.uuid4()),
            "error": {
                "code": "agent_error",
                "message": error,
                "original_message_id": original_message.message_id,
                "context_id": original_message.context_id,
            },
        }

    def _extract_tool_outputs_from_messages(self, messages: List[Any]) -> List[Dict[str, Any]]:
        """Extract tool outputs from React agent messages for hybrid enhancement"""
        tool_outputs = []
        
        try:
            for message in messages:
                # Check for tool calls and their outputs
                if hasattr(message, 'tool_calls') and message.tool_calls:
                    for tool_call in message.tool_calls:
                        tool_name = tool_call.get('name', 'unknown_tool')
                        tool_outputs.append({
                            'tool': tool_name,
                            'call_id': tool_call.get('id', 'unknown'),
                            'args': tool_call.get('args', {}),
                        })
                
                # Check for tool messages (outputs)
                if hasattr(message, 'type') and message.type == 'tool':
                    tool_outputs.append({
                        'tool': getattr(message, 'name', 'unknown_tool'),
                        'content': getattr(message, 'content', ''),
                        'tool_call_id': getattr(message, 'tool_call_id', 'unknown'),
                    })
                
                # Alternative: Check for tool usage in content
                if hasattr(message, 'content') and isinstance(message.content, str):
                    content = message.content
                    # Look for MCP tool patterns in the content
                    # Look for MCP tool patterns or canonical analysis headings in the content
                    if ('perform_advanced_eda_on_csv' in content or 'analyze_csv_patterns' in content
                        or any(h in content for h in ['Key Findings', 'Evidence', 'Interpretation', 'Inferred Insights', 'Recommendations'])):
                        # Extract contiguous blocks that look like tool outputs or analysis blocks
                        lines = content.split('\n')
                        current_tool_output = []
                        in_tool_output = False

                        for line in lines:
                            if '## Comprehensive CSV Data Analysis' in line or '### Dataset Overview' in line or any(h in line for h in ['Key Findings', 'Evidence', 'Interpretation', 'Inferred Insights', 'Recommendations']):
                                in_tool_output = True
                                current_tool_output = [line]
                            elif in_tool_output and line.strip() and not line.startswith('#'):
                                current_tool_output.append(line)
                            elif in_tool_output and (line.startswith('##') or not line.strip()):
                                if current_tool_output:
                                    tool_outputs.append({
                                        'tool': 'perform_advanced_eda_on_csv',
                                        'content': '\n'.join(current_tool_output),
                                        'extracted_from': 'content_analysis',
                                    })
                                    current_tool_output = []
                                    in_tool_output = False
                        # If we didn't find structured sub-blocks but the content includes canonical headings, append whole content
                        if not any(item.get('extracted_from') == 'content_analysis' for item in tool_outputs) and any(h in content for h in ['Key Findings', 'Evidence', 'Interpretation', 'Inferred Insights', 'Recommendations']):
                            tool_outputs.append({
                                'tool': 'detected_analysis_block',
                                'content': content,
                                'extracted_from': 'full_content',
                            })
            
            logger.info(f"Extracted {len(tool_outputs)} tool outputs for hybrid enhancement")
            return tool_outputs
            
        except Exception as e:
            logger.warning(f"Error extracting tool outputs: {e}")
            return []

    def _enhance_report_with_tool_outputs(self, agent_summary: str, tool_outputs: List[Dict[str, Any]]) -> str:
        """
        DEPRECATED: Secondary enhancement approach removed for clean architecture.
        Report generation should be intelligent from the start, not enhanced after the fact.
        This method now simply returns the agent summary to maintain compatibility.
        """
        logger.info("Using primary analysis result without secondary enhancement - clean architecture approach")
        return agent_summary

    def _extract_executed_tools_from_messages(self, messages: List[Any]) -> List[str]:
        """Extract tool names that were executed from message sequence"""
        executed_tools = []
        
        try:
            for message in messages:
                # Check for tool calls in message
                if hasattr(message, 'tool_calls') and message.tool_calls:
                    for tool_call in message.tool_calls:
                        tool_name = tool_call.get('name', 'unknown_tool')
                        if tool_name not in executed_tools:
                            executed_tools.append(tool_name)
                
                # Alternative: check additional_kwargs for tool_calls 
                if hasattr(message, 'additional_kwargs') and 'tool_calls' in message.additional_kwargs:
                    tool_calls = message.additional_kwargs['tool_calls']
                    for tool_call in tool_calls:
                        if 'function' in tool_call and 'name' in tool_call['function']:
                            tool_name = tool_call['function']['name']
                            if tool_name not in executed_tools:
                                executed_tools.append(tool_name)
                
                # Check for tool messages (which indicate a tool was called)
                if hasattr(message, 'type') and message.type == 'tool':
                    tool_name = getattr(message, 'name', 'unknown_tool')
                    if tool_name not in executed_tools:
                        executed_tools.append(tool_name)
        
        except Exception as e:
            logger.warning(f"Error extracting executed tools: {e}")
        
        return executed_tools

    def _normalize_message_content(self, content: Any) -> str:
        """Normalize message content into a plain string for safe processing.

        Handles strings, lists of strings/dicts, dicts, bytes, and other types.
        """
        try:
            if content is None:
                return ""

            # Fast path for strings
            if isinstance(content, str):
                return content

            # Bytes -> decode
            if isinstance(content, (bytes, bytearray)):
                try:
                    return content.decode("utf-8")
                except Exception:
                    return content.decode("latin-1", errors="ignore")

            # Lists: join string elements, JSON-dump non-strings
            if isinstance(content, list):
                parts = []
                for item in content:
                    if isinstance(item, str):
                        parts.append(item)
                    else:
                        try:
                            parts.append(json.dumps(item))
                        except Exception:
                            parts.append(str(item))
                return "\n".join(parts)

            # Dicts and other objects: try JSON dump, fallback to str()
            try:
                return json.dumps(content)
            except Exception:
                return str(content)

        except Exception:
            # Last resort
            try:
                return str(content)
            except Exception:
                return ""

    def _message_preview(self, content: Any, max_chars: int = 500) -> str:
        """Return a safe, single-line preview of message content."""
        text = self._normalize_message_content(content)
        preview = text.replace("\n", " ")
        if len(preview) > max_chars:
            return preview[: max_chars - 3] + "..."
        return preview



    async def close_session(self):
        """Clean up session resources"""
        try:
            # Properly close MCP client connections
            if self.mcp_client:
                try:
                    # Close all MCP server connections with timeout
                    await asyncio.wait_for(self.mcp_client.close(), timeout=5.0)
                    logger.info("MCP client connections closed")
                except (
                    ConnectionError,
                    ConnectionResetError,
                    OSError,
                ) as conn_error:
                    logger.warning(f"Connection already closed or reset: {conn_error}")
                except asyncio.TimeoutError:
                    logger.warning("MCP client close timeout - forcing cleanup")
                except Exception as e:
                    logger.warning(f"Error closing MCP client: {e}")
                finally:
                    self.mcp_client = None

            # Flush Langfuse traces if available
            if self.langfuse_handler:
                try:
                    langfuse_client = get_client()
                    if langfuse_client:
                        langfuse_client.flush()
                        logger.info("Langfuse traces flushed")
                except Exception as e:
                    logger.warning(f"Failed to flush Langfuse traces: {e}")

            # Clear tools and agent references
            self.tools = []
            self.react_agent = None
            self.current_state = None

            logger.info("Session closed and resources cleaned up")

        except Exception as e:
            logger.error(f"Error during session cleanup: {e}")
            # Force cleanup even if there are errors
            self.mcp_client = None
            self.tools = []
            self.react_agent = None
            self.current_state = None

    async def cleanup(self):
        """Force cleanup all resources - for use during shutdown"""
        self._shutdown_requested = True
        try:
            await self.close_session()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

        # Clean up temporary uploaded files
        try:
            temp_dir = Path("temp_uploads")
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
                logger.info("Cleaned up temporary uploaded files")
        except Exception as e:
            logger.warning(f"Error cleaning up temp upload files: {e}")
            
        # Clean up temporary dataset files
        try:
            temp_datasets_dir = self.config.temp_datasets_path
            if temp_datasets_dir.exists():
                shutil.rmtree(temp_datasets_dir)
                logger.info("Cleaned up temporary dataset files")
        except Exception as e:
            logger.warning(f"Error cleaning up temp dataset files: {e}")

        # Force close any remaining connections with short timeout
        try:
            if hasattr(self, "mcp_client") and self.mcp_client:
                await asyncio.wait_for(self.mcp_client.close(), timeout=2.0)
        except (
            asyncio.TimeoutError,
            ConnectionError,
            ConnectionResetError,
            OSError,
        ):
            logger.info("MCP connections forcibly closed")
        except Exception as e:
            logger.warning(f"Force cleanup MCP client error: {e}")

        # Final cleanup
        self.mcp_client = None
        self.tools = []
        self.react_agent = None
        self.current_state = None

        logger.info("Forced cleanup completed")

    def request_shutdown(self):
        """Request graceful shutdown"""
        self._shutdown_requested = True
        logger.info("Shutdown requested")

# Factory function for A2A compatibility
async def create_analysis_agent(
    config_path: Optional[str] = None,
) -> LangGraphReactAnalysisAgent:
    """Create and initialize a LangGraph React Analysis Agent with proper resource management"""
    agent = LangGraphReactAnalysisAgent(config_path)

    try:
        # Initialize MCP tools
        success = await agent.initialize_mcp_tools()
        if not success:
            await agent.cleanup()
            raise RuntimeError("Failed to initialize MCP tools")

        return agent

    except Exception:
        # Cleanup on initialization failure
        await agent.cleanup()
        raise


if __name__ == "__main__":
    print("LangGraph React Analysis Agent")
    print("This is a A2A agent designed to work with orchestrators.")
    print("Import this module to use the agent programmatically via A2A protocol.")