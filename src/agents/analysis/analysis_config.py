"""
Configuration management for the Dynamic Analysis Agent
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


from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import Field, AliasChoices
from pydantic_settings import BaseSettings, SettingsConfigDict


class MCPServerConfig(BaseSettings):
    """Configuration for an MCP server."""

    name: str
    script_path: str
    transport_type: str = "subprocess"  # Changed from "transport" to "transport_type"
    description: Optional[str] = None
    enabled: bool = True


class AgentConfig(BaseSettings):
    """Main configuration for the Dynamic Analysis Agent."""

    # API Keys - Multiple LLM Support
    google_api_key: Optional[str] = Field(default=None, validation_alias=AliasChoices("GOOGLE_API_KEY"))
    gemini_api_key: Optional[str] = Field(default=None, validation_alias=AliasChoices("GEMINI_API_KEY3", "GEMINI_API_KEY"))
    openai_api_key: Optional[str] = Field(default=None, validation_alias=AliasChoices("OPENAI_API_KEY"))
    anthropic_api_key: Optional[str] = Field(default=None, validation_alias=AliasChoices("ANTHROPIC_API_KEY"))

    # LangFuse Configuration
    langfuse_secret_key: Optional[str] = Field(default=None, validation_alias=AliasChoices("LANGFUSE_SECRET_KEY"))
    langfuse_public_key: Optional[str] = Field(default=None, validation_alias=AliasChoices("LANGFUSE_PUBLIC_KEY"))
    langfuse_host: str = Field(
        default="https://cloud.langfuse.com", validation_alias=AliasChoices("LANGFUSE_HOST")
    )

    # Rate Limiting Configuration (Provider-Aware)
    rate_limit_rpm: int = Field(
        default=15, validation_alias=AliasChoices("RATE_LIMIT_RPM")
    )  # Conservative for Gemini free tier (30 RPM limit, use 15 for safety)
    rate_limit_rpd: int = Field(
        default=150, validation_alias=AliasChoices("RATE_LIMIT_RPD")
    )  # Well under 200 RPD limit for Gemini free tier
    enable_rate_limiting: bool = Field(
        default=True, validation_alias=AliasChoices("ENABLE_RATE_LIMITING")
    )  # Enabled by default to prevent quota exhaustion

    # LLM Configuration - Flexible Provider Support
    llm_provider: str = Field(
        default="anthropic", validation_alias=AliasChoices("LLM_PROVIDER")
    )  # anthropic, openai, gemini
    anthropic_model: str = Field(
        default="claude-3-5-haiku-20241022", validation_alias=AliasChoices("ANTHROPIC_MODEL")

    )
    openai_model: str = Field(default="gpt-4", validation_alias=AliasChoices("OPENAI_MODEL"))
    gemini_model: str = Field(default="gemini-2.0-flash-lite", validation_alias=AliasChoices("GEMINI_MODEL"))
    temperature: float = Field(default=0.1, validation_alias=AliasChoices("LLM_TEMPERATURE"))
    max_output_tokens: int = Field(default=4000, validation_alias=AliasChoices("LLM_MAX_TOKENS"))

    # FastMCP Configuration - Make it dynamic
    fastmcp_server_path: str = Field(
        default_factory=lambda: str(Path(__file__).parent / "fastmcp_server.py"),
        validation_alias=AliasChoices("FASTMCP_SERVER_PATH"),
    )

    # MCP Servers Configuration - Make it dynamic
    mcp_servers: List[Dict[str, Any]] = Field(default_factory=list)

    # File paths - Use relative paths from the analysis agent directory
    reports_dir: str = "reportDemo"
    uploads_dir: str = "reportDemo" 
    plots_dir: str = "reportDemo"
    temp_datasets_dir: str = "temp_datasets"
    
    @property
    def analysis_agent_dir(self) -> Path:
        """Get the analysis agent directory as the base for all relative paths."""
        return Path(__file__).parent
    
    @property
    def reports_path(self) -> Path:
        """Get the full path to the reports directory."""
        return self.analysis_agent_dir / self.reports_dir
    
    @property
    def plots_path(self) -> Path:
        """Get the full path to the plots directory."""
        return self.analysis_agent_dir / self.plots_dir
    
    @property
    def temp_datasets_path(self) -> Path:
        """Get the full path to the temp datasets directory."""
        return self.analysis_agent_dir / self.temp_datasets_dir

    # Session management
    session_timeout: int = 3600
    max_concurrent_sessions: int = 10

    # Pydantic v2 settings config (replaces class Config)
    model_config = SettingsConfigDict(
        # Keep .env at the project root (DAA/ in this extracted repo).
        env_file=str(Path(__file__).resolve().parents[2] / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        validate_by_alias=True,
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize MCP servers if not provided
        if not self.mcp_servers:
            self.mcp_servers = self._get_default_mcp_servers()

    def _get_default_mcp_servers(self) -> List[Dict[str, Any]]:
        """Get default MCP server configurations."""
        # Use relative path to current directory
        current_dir = Path(__file__).parent
        server_path = current_dir / "fastmcp_server.py"

        return [
            {
                "name": "analysis_server",
                "script_path": str(server_path),
                "transport_type": "subprocess",
                "description": "Main analysis server with data tools",
                "enabled": True,
            }
        ]

    @property
    def api_key(self) -> str:
        """Get the primary API key for the active LLM provider."""
        if self.llm_provider == "anthropic":
            key = self.anthropic_api_key
            if not key:
                raise ValueError(
                    "Anthropic API key required. Set ANTHROPIC_API_KEY environment variable"
                )
            return key
        elif self.llm_provider == "openai":
            key = self.openai_api_key
            if not key:
                raise ValueError(
                    "OpenAI API key required. Set OPENAI_API_KEY environment variable"
                )
            return key
        elif self.llm_provider == "gemini":
            key = self.google_api_key or self.gemini_api_key
            if not key:
                raise ValueError(
                    "Gemini API key required. Set GOOGLE_API_KEY or GEMINI_API_KEY environment variable"
                )
            return key
        else:
            raise ValueError(f"Unknown LLM provider: {self.llm_provider}")

    @classmethod
    def from_env(cls, env_file: Optional[str] = None) -> "AgentConfig":
        """Create AgentConfig from environment variables or env file."""
        if env_file:
            return cls(_env_file=env_file)
        return cls()

    def get_mcp_server_configs(self) -> List[MCPServerConfig]:
        """Get validated MCP server configurations."""
        configs = []
        for server_dict in self.mcp_servers:
            if server_dict.get("enabled", True):
                try:
                    config = MCPServerConfig(**server_dict)
                    # Validate script path exists
                    script_path = Path(config.script_path)
                    if not script_path.exists():
                        print(
                            f" Warning: MCP server script not found: {config.script_path}"
                        )
                        print(f"   Looking for file at: {script_path.absolute()}")
                        continue
                    configs.append(config)
                except Exception as e:
                    print(f" Warning: Failed to create MCP server config: {e}")
                    continue

        # If no configs found, log helpful information
        if not configs:
            current_dir = Path(__file__).parent
            server_path = current_dir / "fastmcp_server.py"
            print(" No valid MCP server configurations found.")
            print(f"   Expected fastmcp_server.py at: {server_path.absolute()}")
            print(f"   File exists: {server_path.exists()}")

        return configs

    def validate_paths(self) -> Dict[str, bool]:
        """Validate all configured paths exist."""
        validation_results = {}

        # Check FastMCP server path
        fastmcp_path = Path(self.fastmcp_server_path)
        validation_results["fastmcp_server"] = fastmcp_path.exists()

        # Check MCP server paths
        for i, server in enumerate(self.mcp_servers):
            server_path = Path(server["script_path"])
            validation_results[f"mcp_server_{i}"] = server_path.exists()

        # Check directories (create if they don't exist) relative to the analysis agent dir
        base_dir = self.analysis_agent_dir
        for dir_name in [self.reports_dir, self.uploads_dir, self.plots_dir]:
            dir_path = base_dir / dir_name
            if not dir_path.exists():
                try:
                    dir_path.mkdir(parents=True, exist_ok=True)
                    validation_results[dir_name] = True
                except Exception:
                    validation_results[dir_name] = False
            else:
                validation_results[dir_name] = True

        return validation_results


def get_analysis_config(env_file: Optional[str] = None) -> AgentConfig:
    """Compatibility helper for callers expecting get_analysis_config()."""
    return AgentConfig.from_env(env_file=env_file)