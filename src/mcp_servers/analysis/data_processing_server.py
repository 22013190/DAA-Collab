"""
Data Processing FastMCP Server
==============================

A complementary FastMCP server for data processing operations.
Works alongside the main analysis server to provide additional data manipulation capabilities.

Tools provided:
- data_cleaning: Clean and preprocess datasets
- data_transformation: Transform data formats and structures
- data_validation: Validate data quality and integrity
- data_export: Export processed data to various formats

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

import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# FastMCP imports
from fastmcp import FastMCP
from packages.simple_py_logger.src.logger import Logger

# LLM imports for intelligent parsing
try:
    import anthropic

    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    import openai

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Setup logging

parent_logger = Logger("MCP Data Processing Server")
logger = parent_logger.get_current_logger()

# Initialize FastMCP server
mcp = FastMCP("Data Processing Server")

# LLM Configuration
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic")  # anthropic, openai, or google
LLM_MODEL = os.getenv("LLM_MODEL", "claude-3-haiku-20240307")  # Cost-effective model
LLM_API_KEY = os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY")

# Parsing prompt templates
PARSING_PROMPT_TEMPLATE = """
You are an expert at parsing messages between AI agents. Extract structured information from the following message:

MESSAGE:
{content}

Extract and return ONLY a valid JSON object with this exact structure:
{{
  "dataset_info": {{
    "type": "file_path|structured_data|database_results|none",
    "source": "Description of where the data comes from",
    "path_or_data": "File path, embedded data, or data reference",
    "metadata": {{}}
  }},
  "instructions": {{
    "analysis_type": "Type of analysis requested",
    "requirements": ["List of specific tasks"],
    "output_format": "report|visualization|summary|custom",
    "priority": "high|medium|low"
  }},
  "context": {{
    "session_info": "Any session or workflow context",
    "dependencies": "Other agents or tools mentioned",
    "constraints": "Any limitations or special requirements"
  }},
  "confidence": 0.95
}}

IMPORTANT:
- Return ONLY valid JSON, no other text
- If no dataset is found, set type to "none"
- Extract instructions even from informal language
- Handle malformed JSON by fixing common errors
- Assign confidence score (0.0-1.0) based on parsing certainty
"""

DATASET_DETECTION_PROMPT = """
Analyze this message and detect any datasets or data sources mentioned:

MESSAGE:
{content}

Return ONLY a JSON object with this structure:
{{
  "type": "file_path|structured_data|database_results|none",
  "source": "Description of data source",
  "path_or_data": "File path or data reference",
  "metadata": {{
    "format": "csv|json|excel|database|other",
    "size_estimate": "estimated size if mentioned",
    "description": "brief description of the dataset"
  }},
  "confidence": 0.95
}}

Focus specifically on identifying:
- File paths and names
- Embedded data structures
- Database query results
- Data URLs or references
"""

INSTRUCTION_EXTRACTION_PROMPT = """
Extract analysis instructions and requirements from this message:

MESSAGE:
{content}

Return ONLY a JSON object with this structure:
{{
  "analysis_type": "comprehensive|statistical|visual|exploratory|predictive|custom",
  "requirements": ["specific task 1", "specific task 2"],
  "output_format": "report|visualization|summary|dashboard|raw_data",
  "priority": "high|medium|low",
  "constraints": ["any limitations mentioned"],
  "success_criteria": ["how to measure success"],
  "confidence": 0.95
}}

Extract instructions from any format:
- Formal JSON specifications
- Natural language requests
- Bullet point lists
- Mixed format descriptions
"""


# Helper function for LLM calls
async def call_llm_api(prompt: str, max_tokens: int = 1000) -> str:
    """
    Call LLM API with the given prompt.

    Args:
        prompt: The prompt to send to the LLM
        max_tokens: Maximum tokens for the response

    Returns:
        str: The LLM response

    Raises:
        Exception: If API call fails
    """
    try:
        if LLM_PROVIDER == "anthropic" and ANTHROPIC_AVAILABLE and LLM_API_KEY:
            client = anthropic.Anthropic(api_key=LLM_API_KEY)
            response = client.messages.create(
                model=LLM_MODEL,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text

        elif LLM_PROVIDER == "openai" and OPENAI_AVAILABLE and LLM_API_KEY:
            client = openai.OpenAI(api_key=LLM_API_KEY)
            response = client.chat.completions.create(
                model=LLM_MODEL,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content

        else:
            # Fallback: Return structured template for testing
            logger.warning("LLM API not available, using fallback parsing")
            return json.dumps(
                {
                    "dataset_info": {
                        "type": "none",
                        "source": "fallback",
                        "path_or_data": "",
                        "metadata": {},
                    },
                    "instructions": {
                        "analysis_type": "custom",
                        "requirements": ["fallback parsing"],
                        "output_format": "summary",
                        "priority": "medium",
                    },
                    "context": {
                        "session_info": "fallback",
                        "dependencies": "",
                        "constraints": "",
                    },
                    "confidence": 0.5,
                }
            )

    except Exception as e:
        logger.error(f"LLM API call failed: {e}")
        # Return fallback structure
        return json.dumps(
            {
                "dataset_info": {
                    "type": "none",
                    "source": "error",
                    "path_or_data": "",
                    "metadata": {},
                },
                "instructions": {
                    "analysis_type": "custom",
                    "requirements": ["parsing failed"],
                    "output_format": "summary",
                    "priority": "medium",
                },
                "context": {
                    "session_info": "error",
                    "dependencies": "",
                    "constraints": "",
                },
                "confidence": 0.1,
            }
        )


@mcp.tool()
async def llm_message_parser(content: str, format_hint: str = None) -> dict:
    """
    Parse any message format using LLM intelligence.

    This is the primary LLM-powered parsing tool that can handle any message format:
    - JSON structures (well-formed or malformed)
    - Natural language requests
    - Mixed format messages
    - A2A protocol messages
    - Database query results
    - Multi-language content

    Args:
        content: Raw message content to parse
        format_hint: Optional hint about expected format (json, natural_language, a2a, etc.)

    Returns:
        dict: Structured parsing results with dataset_info, instructions, context, and confidence
    """
    logger.info(
        f"LLM parsing message: {len(content)} characters, format_hint: {format_hint}"
    )

    try:
        # Enhance prompt with format hint if provided
        prompt = PARSING_PROMPT_TEMPLATE.format(content=content)
        if format_hint:
            prompt += f"\n\nFORMAT HINT: The message is likely in {format_hint} format."

        # Call LLM API
        response = await call_llm_api(prompt, max_tokens=1500)

        # Parse response as JSON
        try:
            result = json.loads(response)
            logger.info(
                f"LLM parsing successful, confidence: {result.get('confidence', 'unknown')}"
            )
            return result
        except json.JSONDecodeError:
            # Try to extract JSON from response if LLM added extra text
            import re

            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                logger.info(
                    f"LLM parsing successful after extraction, confidence: {result.get('confidence', 'unknown')}"
                )
                return result
            else:
                raise ValueError("LLM response was not valid JSON")

    except Exception as e:
        logger.error(f"LLM message parsing failed: {e}")
        # Return error structure
        return {
            "dataset_info": {
                "type": "none",
                "source": "parsing_error",
                "path_or_data": "",
                "metadata": {"error": str(e)},
            },
            "instructions": {
                "analysis_type": "custom",
                "requirements": ["manual_review_required"],
                "output_format": "summary",
                "priority": "medium",
            },
            "context": {
                "session_info": "error",
                "dependencies": "",
                "constraints": "parsing_failed",
            },
            "confidence": 0.1,
        }


@mcp.tool()
async def llm_dataset_detector(content: str) -> dict:
    """
    Detect datasets in any message format using LLM intelligence.

    Specialized tool for identifying data sources including:
    - File paths and names (CSV, JSON, Excel, etc.)
    - Embedded data structures
    - Database query results
    - Data URLs and references
    - Inline data samples

    Args:
        content: Message content to analyze for datasets

    Returns:
        dict: Dataset detection results with type, source, path_or_data, metadata, and confidence
    """
    logger.info(f"LLM dataset detection: {len(content)} characters")

    try:
        prompt = DATASET_DETECTION_PROMPT.format(content=content)
        response = await call_llm_api(prompt, max_tokens=800)

        # Parse response
        try:
            result = json.loads(response)
            logger.info(
                f"Dataset detection successful, type: {result.get('type')}, confidence: {result.get('confidence')}"
            )
            return result
        except json.JSONDecodeError:
            # Extract JSON from response
            import re

            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                logger.info(
                    f"Dataset detection successful after extraction, type: {result.get('type')}"
                )
                return result
            else:
                raise ValueError("LLM response was not valid JSON")

    except Exception as e:
        logger.error(f"LLM dataset detection failed: {e}")
        return {
            "type": "none",
            "source": "detection_error",
            "path_or_data": "",
            "metadata": {"error": str(e), "format": "unknown"},
            "confidence": 0.1,
        }


@mcp.tool()
async def llm_instruction_extractor(content: str) -> dict:
    """
    Extract analysis instructions from any format using LLM intelligence.

    Specialized tool for understanding and extracting analysis requirements from:
    - Formal JSON specifications
    - Natural language requests
    - Bullet point lists
    - Mixed format descriptions
    - Conversational messages

    Args:
        content: Message content with analysis instructions

    Returns:
        dict: Instruction extraction results with analysis_type, requirements, output_format, etc.
    """
    logger.info(f"LLM instruction extraction: {len(content)} characters")

    try:
        prompt = INSTRUCTION_EXTRACTION_PROMPT.format(content=content)
        response = await call_llm_api(prompt, max_tokens=1000)

        # Parse response
        try:
            result = json.loads(response)
            logger.info(
                f"Instruction extraction successful, type: {result.get('analysis_type')}, confidence: {result.get('confidence')}"
            )
            return result
        except json.JSONDecodeError:
            # Extract JSON from response
            import re

            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                logger.info(
                    f"Instruction extraction successful after extraction, type: {result.get('analysis_type')}"
                )
                return result
            else:
                raise ValueError("LLM response was not valid JSON")

    except Exception as e:
        logger.error(f"LLM instruction extraction failed: {e}")
        return {
            "analysis_type": "custom",
            "requirements": ["extraction_failed", "manual_review_required"],
            "output_format": "summary",
            "priority": "medium",
            "constraints": [f"parsing_error: {str(e)}"],
            "success_criteria": [],
            "confidence": 0.1,
        }


@mcp.tool()
def clean_dataset(
    file_path: str, remove_duplicates: bool = True, handle_missing: str = "drop"
) -> str:
    """
    Clean a dataset by removing duplicates and handling missing values.

    Args:
        file_path: Path to the dataset file (CSV, Excel, etc.)
        remove_duplicates: Whether to remove duplicate rows
        handle_missing: How to handle missing values ('drop', 'fill_mean', 'fill_median', 'fill_mode')

    Returns:
        Status message with cleaning summary
    """
    try:
        # Load dataset
        if file_path.endswith(".csv"):
            df = pd.read_csv(file_path)
        elif file_path.endswith((".xlsx", ".xls")):
            df = pd.read_excel(file_path)
        else:
            return "Error: Unsupported file format. Use CSV or Excel files."

        original_shape = df.shape
        logger.info(f"Original dataset shape: {original_shape}")

        # Remove duplicates if requested
        if remove_duplicates:
            df = df.drop_duplicates()
            logger.info(f"Removed {original_shape[0] - df.shape[0]} duplicate rows")

        # Handle missing values
        missing_before = df.isnull().sum().sum()

        if handle_missing == "drop":
            df = df.dropna()
        elif handle_missing == "fill_mean":
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        elif handle_missing == "fill_median":
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        elif handle_missing == "fill_mode":
            for col in df.columns:
                mode_val = df[col].mode()
                if len(mode_val) > 0:
                    df[col] = df[col].fillna(mode_val[0])

        missing_after = df.isnull().sum().sum()

        # Save cleaned dataset
        output_path = file_path.replace(".csv", "_cleaned.csv").replace(
            ".xlsx", "_cleaned.csv"
        )
        df.to_csv(output_path, index=False)

        summary = f"""
Dataset Cleaning Summary:
- Original shape: {original_shape}
- Final shape: {df.shape}
- Duplicates removed: {remove_duplicates}
- Missing values before: {missing_before}
- Missing values after: {missing_after}
- Missing value strategy: {handle_missing}
- Cleaned dataset saved to: {output_path}
"""

        logger.info("Dataset cleaning completed successfully")
        return summary

    except Exception as e:
        error_msg = f"Error cleaning dataset: {str(e)}"
        logger.error(error_msg)
        return error_msg


@mcp.tool()
def transform_data_types(file_path: str, column_types: str) -> str:
    """
    Transform data types of specific columns in a dataset.

    Args:
        file_path: Path to the dataset file
        column_types: JSON string mapping column names to desired types
                     e.g., '{"age": "int", "price": "float", "date": "datetime"}'

    Returns:
        Status message with transformation summary
    """
    try:
        # Load dataset
        if file_path.endswith(".csv"):
            df = pd.read_csv(file_path)
        else:
            return "Error: Only CSV files supported for type transformation"

        # Parse column types
        try:
            type_mapping = json.loads(column_types)
        except json.JSONDecodeError:
            return "Error: Invalid JSON format for column_types"

        transformations = []

        for col, dtype in type_mapping.items():
            if col not in df.columns:
                transformations.append(f"Warning: Column '{col}' not found in dataset")
                continue

            try:
                if dtype == "int":
                    df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
                elif dtype == "float":
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                elif dtype == "datetime":
                    df[col] = pd.to_datetime(df[col], errors="coerce")
                elif dtype == "string":
                    df[col] = df[col].astype(str)
                elif dtype == "category":
                    df[col] = df[col].astype("category")
                else:
                    transformations.append(
                        f"Warning: Unknown type '{dtype}' for column '{col}'"
                    )
                    continue

                transformations.append(f" Transformed '{col}' to {dtype}")

            except Exception as e:
                transformations.append(f"Error transforming '{col}': {str(e)}")

        # Save transformed dataset
        output_path = file_path.replace(".csv", "_transformed.csv")
        df.to_csv(output_path, index=False)

        summary = f"""
Data Type Transformation Summary:
{chr(10).join(transformations)}

Transformed dataset saved to: {output_path}
"""

        logger.info("Data type transformation completed")
        return summary

    except Exception as e:
        error_msg = f"Error transforming data types: {str(e)}"
        logger.error(error_msg)
        return error_msg


@mcp.tool()
def validate_data_quality(file_path: str) -> str:
    """
    Perform comprehensive data quality validation on a dataset.

    Args:
        file_path: Path to the dataset file

    Returns:
        Detailed data quality report
    """
    try:
        # Load dataset
        if file_path.endswith(".csv"):
            df = pd.read_csv(file_path)
        else:
            return "Error: Only CSV files supported for data validation"

        # Basic statistics
        total_rows = len(df)
        total_cols = len(df.columns)

        # Missing data analysis
        missing_data = df.isnull().sum()
        missing_percent = (missing_data / total_rows * 100).round(2)

        # Duplicate analysis
        duplicate_rows = df.duplicated().sum()

        # Data type analysis
        dtype_summary = df.dtypes.value_counts()

        # Outlier detection for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outlier_summary = []

        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            if outliers > 0:
                outlier_summary.append(
                    f"  - {col}: {outliers} outliers ({(outliers / total_rows * 100):.1f}%)"
                )

        # Generate report
        report = f"""
Data Quality Validation Report
==============================
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Dataset: {file_path}

BASIC STATISTICS:
- Total rows: {total_rows:,}
- Total columns: {total_cols}
- Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB

MISSING DATA ANALYSIS:
"""

        if missing_data.sum() == 0:
            report += "- No missing values found \n"
        else:
            report += f"- Total missing values: {missing_data.sum():,}\n"
            for col, missing in missing_data[missing_data > 0].items():
                report += f"  - {col}: {missing} ({missing_percent[col]}%)\n"

        report += f"""
DUPLICATE DATA:
- Duplicate rows: {duplicate_rows} ({(duplicate_rows / total_rows * 100):.1f}%)

DATA TYPES:
"""
        for dtype, count in dtype_summary.items():
            report += f"- {dtype}: {count} columns\n"

        if outlier_summary:
            report += "\nOUTLIER ANALYSIS:\n"
            report += "\n".join(outlier_summary)
        else:
            report += "\nOUTLIER ANALYSIS:\n- No significant outliers detected in numeric columns "

        report += """

RECOMMENDATIONS:
"""
        recommendations = []

        if duplicate_rows > 0:
            recommendations.append("- Remove duplicate rows to improve data quality")

        if missing_data.sum() > 0:
            high_missing_cols = missing_percent[missing_percent > 50].index.tolist()
            if high_missing_cols:
                recommendations.append(
                    f"- Consider removing columns with >50% missing: {', '.join(high_missing_cols)}"
                )
            else:
                recommendations.append(
                    "- Handle missing values using appropriate imputation methods"
                )

        if len(outlier_summary) > 0:
            recommendations.append(
                "- Investigate and handle outliers in numeric columns"
            )

        if not recommendations:
            recommendations.append("- Data quality looks good! ")

        report += "\n".join(recommendations)

        logger.info("Data quality validation completed")
        return report

    except Exception as e:
        error_msg = f"Error validating data quality: {str(e)}"
        logger.error(error_msg)
        return error_msg


@mcp.tool()
def export_processed_data(
    file_path: str, output_format: str = "csv", sheet_name: str = "Data"
) -> str:
    """
    Export processed data to various formats.

    Args:
        file_path: Path to the source dataset file
        output_format: Output format ('csv', 'excel', 'json', 'parquet')
        sheet_name: Sheet name for Excel export

    Returns:
        Export status message
    """
    try:
        # Load dataset
        if file_path.endswith(".csv"):
            df = pd.read_csv(file_path)
        else:
            return "Error: Source file must be CSV format"

        # Generate output filename
        base_name = Path(file_path).stem

        if output_format.lower() == "csv":
            output_path = f"{base_name}_exported.csv"
            df.to_csv(output_path, index=False)

        elif output_format.lower() == "excel":
            output_path = f"{base_name}_exported.xlsx"
            df.to_excel(output_path, sheet_name=sheet_name, index=False)

        elif output_format.lower() == "json":
            output_path = f"{base_name}_exported.json"
            df.to_json(output_path, orient="records", indent=2)

        elif output_format.lower() == "parquet":
            output_path = f"{base_name}_exported.parquet"
            df.to_parquet(output_path, index=False)

        else:
            return f"Error: Unsupported output format '{output_format}'. Use: csv, excel, json, parquet"

        summary = f"""
Data Export Summary:
- Source file: {file_path}
- Output format: {output_format}
- Output file: {output_path}
- Rows exported: {len(df):,}
- Columns exported: {len(df.columns)}
- Export completed successfully
"""

        logger.info(f"Data exported to {output_path}")
        return summary

    except Exception as e:
        error_msg = f"Error exporting data: {str(e)}"
        logger.error(error_msg)
        return error_msg


@mcp.tool()
def get_server_info() -> str:
    """
    Get information about this data processing server.

    Returns:
        Server information and available tools
    """
    info = """
Data Processing FastMCP Server
==============================
Status: Active
Server Type: Data Processing and Transformation

Available Tools:
1. llm_message_parser - Parse any message format using LLM intelligence
2. llm_dataset_detector - Detect datasets in any message format
3. llm_instruction_extractor - Extract analysis instructions from any format
4. clean_dataset - Remove duplicates and handle missing values
5. transform_data_types - Convert column data types
6. validate_data_quality - Comprehensive data quality analysis
7. export_processed_data - Export data to various formats
8. get_server_info - This information

Purpose:
This server provides specialized data processing and LLM-powered parsing capabilities.
The LLM tools enable intelligent parsing of any message format, while traditional tools
handle data cleaning and transformation tasks.

Usage:
- Use LLM tools for intelligent message and instruction parsing
- Use data tools for preprocessing before analysis
- Validate data quality to ensure reliable results
- Transform data types for optimal processing
- Export processed data in various formats
"""

    return info


if __name__ == "__main__":
    logger.info("Starting Data Processing FastMCP Server...")
    logger.info(
        "Available tools: llm_message_parser, llm_dataset_detector, llm_instruction_extractor, clean_dataset, transform_data_types, validate_data_quality, export_processed_data, get_server_info"
    )
    mcp.run()
