"""
Modular Analysis Pipeline package

Provides a minimal StateGraph-based analysis agent tailored for local LLMs.
"""

from .runner import create_analysis_pipeline_agent

__all__ = [
	"create_analysis_pipeline_agent",
]
