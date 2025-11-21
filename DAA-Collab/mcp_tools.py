import json
from functools import lru_cache
import pandas as pd
from pathlib import Path
from typing import Annotated, List, Union, Optional, Literal, Any, Dict
try:
    # pydantic Field used in some annotations
    from pydantic import Field
except Exception:
    # minimal placeholder for annotations (not used at runtime)
    def Field(*args, **kwargs):
        return None

# Provide a tiny fallback for `mcp.tool` so this file can be imported
try:
    import mcp
except Exception:
    class _mcp_fallback:
        @staticmethod
        def tool(*a, **k):
            def _dec(f):
                return f
            return _dec

    mcp = _mcp_fallback()



@lru_cache()
def load_csv_data_cached(file_path: str) -> pd.DataFrame:
    """Load CSV with caching and optimized type detection based on file path"""
    # simple cache key based on modification time
    def get_file_hash(fp: str) -> str:
        try:
            return str(Path(fp).stat().st_mtime)
        except Exception:
            return fp

    file_hash = get_file_hash(file_path)
    cache_key = f"{file_path}_{file_hash}"

    if cache_key not in _dataset_cache:
        # Load and perform expensive type detection only once
        df = load_csv_data_with_types(file_path)
        _dataset_cache[cache_key] = df

    return _dataset_cache[cache_key]


# Loads a CSV from disk using pandas. Tries to convert string columns to numeric if possible. Returns a cleaned DataFrame.
def load_csv_data(file_path: str) -> pd.DataFrame:
    """DEPRECATED: Use load_csv_data_cached() for better performance"""
    return load_csv_data_with_types(file_path)


# Minimal dataset cache container
try:
    _dataset_cache
except NameError:
    _dataset_cache = {}


def validate_file_and_columns(file_path: str) -> None:
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"File '{file_path}' not found")


def load_csv_data_with_types(file_path: str, auto_expand_json: bool = True) -> pd.DataFrame:
    # Minimal loader that reads CSV and attempts to parse numeric/datetime
    df = pd.read_csv(file_path)
    # Try to parse object columns as datetime where possible
    for c in df.select_dtypes(include=["object"]).columns:
        try:
            parsed = pd.to_datetime(df[c], errors="coerce")
            if parsed.notna().sum() > 0 and parsed.notna().sum() / len(parsed) > 0.5:
                df[c] = parsed
        except Exception:
            pass
    return df


def detect_column_types(df: pd.DataFrame) -> Dict[str, Any]:
    numeric = df.select_dtypes(include=["number"]).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime", "datetime64"]).columns.tolist()
    categorical = [c for c in df.columns if c not in numeric and c not in datetime_cols]
    return {"_metadata": {"numeric": numeric, "categorical": categorical, "datetime": datetime_cols},
            **{c: str(df[c].dtype) for c in df.columns}}


# Loads a CSV file and outputs:
# Basic metadata (rows, columns).
# Data types.
# Basic numeric stats (mean, std, min, max).
# Missing values summary.
# Returns a Markdown-formatted string report.
@mcp.tool()
def load_and_analyze_csv(
    file_path: Annotated[str, Field(description="Path to the CSV file to analyze")],
    auto_expand_json: Annotated[bool, Field(description="Automatically detect and expand JSON columns into separate columns for analysis")] = True,
) -> str:
    """
    Load a CSV file and provide a detailed summary of its structure and contents.

    WHEN TO USE THIS TOOL:
    - ALWAYS use this as the FIRST tool when starting any data analysis
    - Use when you need to understand what columns exist in the dataset
    - Use to identify data types (numeric, categorical, datetime)
    - Use to detect data quality issues (missing values, duplicates)

    WHAT THIS TOOL DOES:
    - Loads CSV and returns comprehensive dataset overview
    - Identifies all column names and their data types
    - Provides descriptive statistics for numeric columns (mean, std, min, max, quartiles)
    - Reports missing values per column with percentages
    - Detects and expands JSON columns automatically (if enabled)
    - Identifies high-cardinality categorical columns

    RETURNS: Markdown-formatted report with:
    - Dataset dimensions (rows, columns, memory usage)
    - Complete column list with detected types
    - Summary statistics for all numeric columns
    - Missing value analysis
    - Data quality warnings

    PREREQUISITES: None - this is always the first tool to use
    """
    try:
        # Validate file exists and columns if needed
        validate_file_and_columns(file_path)

        # Load the data using enhanced loader with JSON processing
        df = load_csv_data_with_types(file_path, auto_expand_json=auto_expand_json)

        # Get column type information
        column_types = detect_column_types(df)
        type_metadata = column_types.get("_metadata", {})
        types_by_column = {
            key: value for key, value in column_types.items() if key != "_metadata"
        }

        report = []
        report.append("## CSV Data Analysis")
        report.append(f"- File: {file_path}")
        report.append(f"- Total rows: {len(df):,}")
        report.append(f"- Total columns: {len(df.columns)}")
        report.append(
            f"- Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
        )
        report.append(f"- Columns: {', '.join(df.columns)}")

        # Enhanced data types with auto-detected types
        report.append("\n### Data Types:")
        for col in df.columns:
            dtype = df[col].dtype
            detected_type = types_by_column.get(col, "unknown")
            report.append(f"- {col}: {dtype} (detected as: {detected_type})")

        high_cardinality_cols = type_metadata.get("high_cardinality", [])
        if high_cardinality_cols:
            report.append(
                "High-cardinality categorical columns: "
                + ", ".join(high_cardinality_cols)
            )

        # Basic statistics for numeric columns
        numeric_cols = df.select_dtypes(include=["number"]).columns
        if len(numeric_cols) > 0:
            report.append("\n### Numeric Columns Summary:")
            for col in numeric_cols:
                stats = df[col].describe()
                report.append(
                    f"- {col}: mean={stats['mean']:.2f}, std={stats['std']:.2f}, "
                    f"min={stats['min']}, max={stats['max']}, "
                    f"25%={stats['25%']:.2f}, 75%={stats['75%']:.2f}"
                )

        # Missing values with percentages
        missing_data = df.isnull().sum()
        if missing_data.sum() > 0:
            report.append("\n### Missing Values:")
            for col, missing_count in missing_data.items():
                if missing_count > 0:
                    missing_pct = (missing_count / len(df)) * 100
                    report.append(
                        f"- {col}: {missing_count} missing values ({missing_pct:.1f}%)"
                    )

        # Data quality summary
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            report.append("\n### Data Quality:")
            report.append(f"- Duplicate rows: {duplicates}")

        # Append a machine-readable JSON footer to help downstream argument inference
        try:
            footer_payload = {
                "path": file_path,
                # Provide both a flat summary and nested metadata for flexibility
                "numeric_columns": type_metadata.get("numeric", []),
                # Server uses 'categorical' to denote text-like columns
                "text_columns": type_metadata.get("categorical", []),
                "datetime_columns": type_metadata.get("datetime", []),
                "column_types": column_types,  # includes _metadata with lists
            }
            report.append("\n<!--output_json:" + json.dumps(footer_payload) + "-->")
        except Exception:
            # If footer generation fails, proceed without blocking
            pass

        return "\n".join(report)

    except Exception as e:
        return f"Error analyzing CSV: {str(e)}"


# Full EDA pipeline with:
# Descriptive stats.
# Correlation matrix.
# Missing/duplicate data summary.
# Optional KMeans clustering on numeric columns.
# Optional Linear Regression if valid x_column and y_column are given.
# Returns Markdown report summarizing the above.


@mcp.tool()
def perform_advanced_eda_on_csv(
    file_path: Annotated[str, Field(description="Path to the CSV file to analyze")],
    auto_detect_analysis: Annotated[bool, Field(description="Automatically detect and perform appropriate analysis based on column types - enhances analysis quality with intelligent pattern detection")] = False,
    auto_expand_json: Annotated[bool, Field(description="Automatically detect and expand JSON columns into separate columns for comprehensive analysis")] = True,
    n_clusters: Annotated[
        int,
        Field(
            description="Number of clusters for KMeans (default=min(3, len(df)//10))"
        ),
    ] = None,
    random_state: Annotated[int, Field(description="Random state for clustering")] = 0,
    n_init: Annotated[
        Union[str, int], Field(description="n_init parameter for KMeans")
    ] = "auto",
    x_column: Annotated[
        str, Field(description="Independent variable column for regression")
    ] = None,
    y_column: Annotated[
        str, Field(description="Dependent variable column for regression")
    ] = None,
) -> str:
    """
    Perform comprehensive exploratory data analysis (EDA) on a CSV dataset.

    WHEN TO USE THIS TOOL:
    - Use AFTER load_and_analyze_csv to get deeper statistical insights
    - Use when you need correlation analysis between numeric columns
    - Use when you want to identify patterns in temporal data (with auto_detect_analysis=True)
    - Use when you need categorical distribution analysis (with auto_detect_analysis=True)
    - Use for geographical analysis if lat/lon columns exist (with auto_detect_analysis=True)

    WHAT THIS TOOL DOES:
    Core analysis (always performed):
    - Descriptive statistics for all columns (numeric and categorical)
    - Correlation matrix for numeric columns with significance filtering
    - Missing data patterns and duplicate detection
    - Clustering analysis (KMeans) on numeric data
    - Regression analysis if x_column and y_column specified

    Enhanced intelligent analysis (when auto_detect_analysis=True):
    - TEMPORAL: Creates hourly/daily/weekly patterns if datetime columns found
    - CATEGORICAL: Distribution analysis for incident/event type columns  
    - GEOGRAPHICAL: Spatial analysis if latitude/longitude columns detected

    JSON handling (when auto_expand_json=True):
    - Detects columns containing JSON strings (handles various formats)
    - Automatically expands JSON into separate columns
    - Supports standard JSON, escaped JSON, and malformed JSON cleaning

    RETURNS: Markdown report with:
    - Complete descriptive statistics
    - Filtered correlation matrix (only meaningful correlations shown)
    - Data quality assessment
    - Clustering insights (if applicable)
    - Temporal/categorical/spatial patterns (if auto_detect_analysis=True)

    PREREQUISITES: Dataset must exist and be loadable
    RECOMMENDED PARAMETERS: Set auto_detect_analysis=True for comprehensive insights
    """
    try:
        if isinstance(n_init, str):
            if n_init.lower() == "auto":
                n_init = "auto"  # Keep as string
            elif n_init.isdigit():
                n_init = int(n_init)  # Convert string numbers to int
            else:
                raise ValueError("n_init must be 'auto' or an integer")
        elif isinstance(n_init, int):
            if n_init < 1:
                raise ValueError("n_init as integer must be >= 1")
        else:
            raise ValueError("n_init must be 'auto' or an integer")

        if not Path(file_path).exists():
            return f"Error: File '{file_path}' does not exist."

        # Use enhanced data loading with JSON processing
        df = load_csv_data_with_types(file_path, auto_expand_json=auto_expand_json)

        # Get column type information
        column_types = detect_column_types(df)

        eda_report = ["## Comprehensive CSV Data Analysis"]
        eda_report.append("\n### Dataset Overview:")
        eda_report.append(f"- File: {file_path}")
        eda_report.append(f"- Total records: {len(df)}")
        eda_report.append(f"- Columns: {', '.join(df.columns)}")

        eda_report.append("\n### Descriptive Statistics:")
        eda_report.append(df.describe(include="all").to_string())

        numeric_cols = df.select_dtypes(include=["number"]).columns
        if len(numeric_cols) > 1:
            eda_report.append("\n\n### Correlation Analysis:")
            # Use optimized correlation analysis for large datasets
            if len(numeric_cols) > 20:
                corr_df = optimize_correlation_analysis(df[numeric_cols])
                eda_report.append("(Showing top correlated features for performance)")
                corr = corr_df.corr()
            else:
                corr = df[numeric_cols].corr()
            
            # Use semantic filter to show only meaningful correlations
            meaningful_corr_text = filter_correlation_matrix(corr)
            eda_report.append(meaningful_corr_text)
            
            # Only show full matrix for small datasets with meaningful correlations
            if len(numeric_cols) <= 5 and any(abs(corr.iloc[i, j]) >= 0.2 and is_meaningful_correlation(corr.columns[i], corr.columns[j]) 
                                             for i in range(len(corr)) for j in range(i+1, len(corr))):
                eda_report.append("\n**Full Correlation Matrix:**")
                eda_report.append(corr.round(3).to_string())

        eda_report.append("\n\n### Data Quality:")
        missing_data = df.isnull().sum()
        if missing_data.sum() > 0:
            eda_report.append("Missing Values:")
            for col, missing_count in missing_data.items():
                if missing_count > 0:
                    pct = (missing_count / len(df)) * 100
                    eda_report.append(f"- {col}: {missing_count} ({pct:.1f}%)")
        else:
            eda_report.append("No missing values found.")
        eda_report.append(f"Duplicate rows: {df.duplicated().sum()}")

        if len(numeric_cols) >= 2:
            eda_report.append("\n\n## Advanced Analysis")

            clean_numeric_df = df[numeric_cols].dropna()
            if len(clean_numeric_df) > 2:
                try:
                    cluster_count = n_clusters or min(3, len(clean_numeric_df) // 10)
                    scaler = StandardScaler()
                    scaled_data = scaler.fit_transform(clean_numeric_df)
                    kmeans = KMeans(
                        n_clusters=cluster_count,
                        random_state=random_state,
                        n_init=n_init,
                    )
                    clusters = kmeans.fit_predict(scaled_data)

                    df_with_clusters = clean_numeric_df.copy()
                    df_with_clusters["cluster"] = clusters

                    eda_report.append(
                        f"\n### KMeans Clustering ({cluster_count} clusters):"
                    )
                    cluster_summary = (
                        df_with_clusters.groupby("cluster")[numeric_cols]
                        .agg(["mean", "count"])
                        .round(2)
                    )
                    eda_report.append(cluster_summary.to_string())

                except Exception as e:
                    eda_report.append(f"\nClustering analysis failed: {e}")

            # Regression
            col1 = x_column or (numeric_cols[0] if len(numeric_cols) > 1 else None)
            col2 = y_column or (numeric_cols[1] if len(numeric_cols) > 1 else None)

            if col1 and col2 and col1 in df.columns and col2 in df.columns:
                try:
                    if pd.api.types.is_numeric_dtype(
                        df[col1]
                    ) and pd.api.types.is_numeric_dtype(df[col2]):
                        clean_data = df[[col1, col2]].dropna()
                        if len(clean_data) > 1:
                            model = LinearRegression()
                            X = clean_data[[col1]].values
                            y = clean_data[col2].values
                            model.fit(X, y)
                            r2 = model.score(X, y)
                            eda_report.append(
                                f"\n### Linear Regression: {col2} ~ {col1}"
                            )
                            eda_report.append(f"- R² Score: {r2:.4f}")
                            eda_report.append(f"- Coefficient: {model.coef_[0]:.4f}")
                            eda_report.append(f"- Intercept: {model.intercept_:.4f}")
                        else:
                            eda_report.append(
                                f"\nRegression skipped: not enough data for {col1} and {col2}."
                            )
                    else:
                        eda_report.append(
                            f"\nRegression skipped: {col1} or {col2} is not numeric."
                        )
                except Exception as e:
                    eda_report.append(f"\nRegression analysis failed: {e}")
            else:
                eda_report.append(
                    "\nRegression skipped: no valid column pair selected."
                )

        # INTELLIGENT ANALYSIS: Auto-detect common patterns and add specialized analysis
        if auto_detect_analysis:
            eda_report.append("\n## Intelligent Pattern Detection")
            
            # 1. TEMPORAL ANALYSIS: Auto-detect datetime columns
            datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
            if not datetime_cols:
                # Try to detect datetime-like string columns
                for col in df.columns:
                    if any(word in col.lower() for word in ['time', 'date', 'timestamp', 'created', 'occurred']):
                        try:
                            pd.to_datetime(df[col].head(100))  # Test with sample
                            datetime_cols.append(col)
                            break
                        except:
                            continue
            
            if datetime_cols:
                eda_report.append(f"\n### 📅 TEMPORAL ANALYSIS DETECTED")
                datetime_col = datetime_cols[0]
                eda_report.append(f"**Temporal column identified**: {datetime_col}")
                
                try:
                    # Convert to datetime if needed
                    if df[datetime_col].dtype == 'object':
                        df[datetime_col] = pd.to_datetime(df[datetime_col])
                    
                    # Extract temporal components
                    df['_hour'] = df[datetime_col].dt.hour
                    df['_day_of_week'] = df[datetime_col].dt.day_name()
                    df['_month'] = df[datetime_col].dt.month
                    
                    # Hourly distribution
                    hourly_counts = df['_hour'].value_counts().sort_index()
                    peak_hour = hourly_counts.idxmax()
                    peak_count = hourly_counts.max()
                    eda_report.append(f"- **Peak Hour**: {peak_hour}:00 with {peak_count:,} records ({peak_count/len(df)*100:.1f}%)")
                    
                    # Daily distribution
                    daily_counts = df['_day_of_week'].value_counts()
                    peak_day = daily_counts.idxmax()
                    peak_day_count = daily_counts.max()
                    eda_report.append(f"- **Peak Day**: {peak_day} with {peak_day_count:,} records ({peak_day_count/len(df)*100:.1f}%)")
                    
                    # Date range
                    date_range = df[datetime_col].max() - df[datetime_col].min()
                    eda_report.append(f"- **Date Range**: {df[datetime_col].min().strftime('%Y-%m-%d')} to {df[datetime_col].max().strftime('%Y-%m-%d')} ({date_range.days} days)")
                    
                    # Optional: Save temporal analysis files (only for detailed analysis workflows)
                    # Note: CSV export removed to prevent automatic file generation
                    # Users can use dedicated export tools if CSV output is needed
                    
                    # Clean up temporary columns
                    df.drop(['_hour', '_day_of_week', '_month'], axis=1, inplace=True, errors='ignore')
                    
                except Exception as e:
                    eda_report.append(f"- Temporal analysis failed: {e}")
            
            # 2. CATEGORICAL ANALYSIS: Auto-detect incident/event type columns
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            category_col = None
            
            for col in categorical_cols:
                col_lower = col.lower()
                if any(word in col_lower for word in ['type', 'category', 'class', 'kind', 'incident', 'event']):
                    unique_ratio = df[col].nunique() / len(df)
                    if 0.001 < unique_ratio < 0.5:  # Good categorical ratio
                        category_col = col
                        break
            
            if category_col:
                eda_report.append(f"\n### 📊 CATEGORICAL ANALYSIS DETECTED")
                eda_report.append(f"**Category column identified**: {category_col}")
                
                try:
                    type_counts = df[category_col].value_counts()
                    total = len(df)
                    
                    eda_report.append(f"- **Total categories**: {len(type_counts)}")
                    eda_report.append(f"- **Most common**: {type_counts.index[0]} ({type_counts.iloc[0]:,} records, {type_counts.iloc[0]/total*100:.1f}%)")
                    if len(type_counts) > 1:
                        eda_report.append(f"- **Second most**: {type_counts.index[1]} ({type_counts.iloc[1]:,} records, {type_counts.iloc[1]/total*100:.1f}%)")
                    
                    # Distribution analysis
                    eda_report.append(f"\n**Full distribution:**")
                    for category, count in type_counts.head(10).items():
                        percentage = count / total * 100
                        eda_report.append(f"  - {category}: {count:,} ({percentage:.1f}%)")
                    
                    if len(type_counts) > 10:
                        eda_report.append(f"  - ... and {len(type_counts) - 10} more categories")
                        
                except Exception as e:
                    eda_report.append(f"- Categorical analysis failed: {e}")
            
            # 3. GEOGRAPHICAL ANALYSIS: Auto-detect lat/lon columns
            geo_cols = []
            for col in df.columns:
                col_lower = col.lower()
                if any(word in col_lower for word in ['lat', 'longitude', 'lng', 'coord']):
                    if pd.api.types.is_numeric_dtype(df[col]):
                        geo_cols.append(col)
            
            if len(geo_cols) >= 2:
                eda_report.append(f"\n### 🗺️ GEOGRAPHICAL ANALYSIS DETECTED")
                eda_report.append(f"**Geographic columns identified**: {', '.join(geo_cols[:2])}")
                
                try:
                    lat_col, lon_col = geo_cols[0], geo_cols[1]
                    # Basic geographic statistics
                    lat_range = df[lat_col].max() - df[lat_col].min()
                    lon_range = df[lon_col].max() - df[lon_col].min()
                    center_lat = df[lat_col].mean()
                    center_lon = df[lon_col].mean()
                    
                    eda_report.append(f"- **Geographic center**: ({center_lat:.4f}, {center_lon:.4f})")
                    eda_report.append(f"- **Latitude range**: {lat_range:.4f} degrees")
                    eda_report.append(f"- **Longitude range**: {lon_range:.4f} degrees")
                    eda_report.append(f"- **Spatial spread**: {'Wide' if lat_range > 1 or lon_range > 1 else 'Localized'}")
                    
                    # Check for actual spatial clustering or patterns (meaningful analysis)
                    if lat_range > 0.001 and lon_range > 0.001:  # Only if there's actual variation
                        eda_report.append(f"- **Data distribution**: Geographic points show spatial variation suitable for mapping")
                    else:
                        eda_report.append(f"- **Data distribution**: Points are highly concentrated (minimal spatial variation)")
                    
                except Exception as e:
                    eda_report.append(f"- Geographic analysis failed: {e}")

        # Append a machine-readable JSON footer so the orchestrator can capture schema/profile
        try:
            type_metadata = column_types.get("_metadata", {})
            footer_payload = {
                "path": file_path,
                "numeric_columns": type_metadata.get("numeric", []),
                "text_columns": type_metadata.get("categorical", []),
                "datetime_columns": type_metadata.get("datetime", []),
                "column_types": column_types,
            }
            eda_report.append("\n<!--output_json:" + json.dumps(footer_payload) + "-->")
        except Exception:
            pass

        return "\n".join(eda_report)

    except Exception as e:
        return f"Error performing EDA: {str(e)}"


@mcp.tool()
def generate_basic_plot(
    file_path: Annotated[str, Field(description="Path to the CSV file to plot")],
    plot_type: Annotated[
        Literal["histogram", "scatterplot", "boxplot", "pairplot", "bar_chart"],
        Field(description="One of: histogram, scatterplot, boxplot, pairplot, bar_chart"),
    ],
    x_column: Annotated[
        str,
        Field(description="For histogram/boxplot: numeric column. For scatterplot/bar_chart: x-axis column.")
    ] = None,
    y_column: Annotated[
        str,
        Field(description="For scatterplot/bar_chart: y-axis column (numeric for scatterplot).")
    ] = None,
    hue_column: Annotated[
        str,
        Field(description="Optional grouping/category for boxplot or bar_chart (acts as x categories for boxplot)")
    ] = None,
    columns_for_pairplot: Annotated[
        List[str],
        Field(description="Numeric columns for pairplot (max ~5 recommended).")
    ] = None,
    title: Annotated[str, Field(description="Title of the plot")] = "Generated Plot",
    output_dir: Annotated[
        str,
        Field(description="Directory to save the plot image. Relative paths resolved from current working directory.")
    ] = ".",
) -> str:
    """
    Generate a standard plot: histogram, scatterplot, boxplot, pairplot, or bar_chart.

    Behavior:
    - histogram: requires numeric x_column
    - scatterplot: requires numeric x_column and y_column
    - boxplot: x_column is the numeric value plotted on Y; hue_column (if provided) is used as categorical X
    - pairplot: requires columns_for_pairplot (numeric); max ~5 recommended
    - bar_chart:
        • If y_column is missing or non-numeric, plots counts of x_column categories
        • If y_column is numeric, plots bar heights of y grouped by x (no aggregation)

    Returns:
    - Markdown with inline image, a "Saved to:" absolute path, an "Artifact:" filename, and a short summary.
    """
    # Local imports (safe if pasted into this file)
    from pathlib import Path
    from datetime import datetime
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    def cleanup_plot():
        try:
            plt.close("all")
        except Exception:
            pass

    def _save_and_return(_title: str, _plot_type: str, _desc: str) -> str:
        try:
            plt.tight_layout()
        except Exception:
            pass

        plot_file_path = None
        try:
            if not Path(output_dir).is_absolute():
                # Resolve relative to current working directory
                project_root = Path.cwd()
                output_path = project_root / output_dir
            else:
                output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            sanitized_title = "".join(c for c in _title if c.isalnum() or c in (" ", ".", "_")).rstrip()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_file_name = f"{sanitized_title.replace(' ', '_').replace('.', '')}_{_plot_type}_{timestamp}.png"
            plot_file_path = output_path / plot_file_name
            plt.savefig(plot_file_path, dpi=150, bbox_inches="tight")
        finally:
            cleanup_plot()

        if plot_file_path:
            return (
                f"![{_title}]({plot_file_path.name})\n"
                f"Saved to: {plot_file_path}\n"
                f"Artifact: {plot_file_path.name}\n"
                f"Summary: {_desc}"
            )
        return f"Plot generated but not saved to disk.\nSummary: {_desc}"

    # 1) Validate file
    # debug: show resolved path and cwd
    try:
        resolved = Path(file_path).resolve()
    except Exception:
        resolved = Path(file_path)
    print(f"[mcp_tools] generate_basic_plot: checking file {resolved} cwd={Path.cwd()}")
    if not Path(file_path).exists():
        return f"Error: File '{file_path}' does not exist."\

    # 2) Load data: prefer cached loader if available in this module
    try:
        df = load_csv_data_cached(file_path)  # type: ignore[name-defined]
    except Exception:
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            return f"Error: Failed to read CSV '{file_path}': {e}"

    # 3) Switch on plot type
    try:
        # HISTOGRAM
        if plot_type == "histogram":
            if not x_column:
                return "Error: 'x_column' is required for histogram."
            if x_column not in df.columns:
                return f"Error: Column '{x_column}' not found."
            if not pd.api.types.is_numeric_dtype(df[x_column]):
                return f"Error: Column '{x_column}' is not numeric for histogram."

            sns.histplot(df[x_column].dropna(), kde=True)
            plt.title(f"Histogram of {x_column}" if title == "Generated Plot" else title)
            plt.xlabel(x_column)
            plt.ylabel("Frequency")
            return _save_and_return(title if title != "Generated Plot" else f"Histogram of {x_column}", "histogram", f"Histogram of {x_column}.")

        # SCATTERPLOT
        if plot_type == "scatterplot":
            if not x_column or not y_column:
                return "Error: 'x_column' and 'y_column' are required for scatterplot."
            if x_column not in df.columns or y_column not in df.columns:
                return f"Error: One or both columns ('{x_column}', '{y_column}') not found."
            if not (pd.api.types.is_numeric_dtype(df[x_column]) and pd.api.types.is_numeric_dtype(df[y_column])):
                return "Error: Both 'x_column' and 'y_column' must be numeric for scatterplot."

            sns.scatterplot(
                x=df[x_column],
                y=df[y_column],
                hue=df[hue_column] if hue_column and hue_column in df.columns else None,
            )
            ttl = title if title != "Generated Plot" else f"Scatter Plot of {y_column} vs {x_column}"
            plt.title(ttl)
            plt.xlabel(x_column)
            plt.ylabel(y_column)
            return _save_and_return(ttl, "scatterplot", f"Scatter of {y_column} vs {x_column}.")

        # BOX PLOT
        if plot_type == "boxplot":
            if not x_column:
                return "Error: 'x_column' (numeric) is required for boxplot."
            if x_column not in df.columns:
                return f"Error: Column '{x_column}' not found."
            if not pd.api.types.is_numeric_dtype(df[x_column]):
                return f"Error: Column '{x_column}' is not numeric for boxplot."

            # x=category (if hue_column provided), y=value is x_column
            cat = df[hue_column] if hue_column and hue_column in df.columns else None
            sns.boxplot(x=cat, y=df[x_column])
            plt.title(title if title != "Generated Plot" else f"Box Plot of {x_column}")
            if hue_column and hue_column in df.columns:
                plt.xlabel(hue_column)
            plt.ylabel(x_column)
            return _save_and_return(
                title if title != "Generated Plot" else f"Box Plot of {x_column}",
                "boxplot",
                f"Box plot of {x_column}" + (f" grouped by {hue_column}." if hue_column and hue_column in df.columns else "."),
            )

        # PAIRPLOT
        if plot_type == "pairplot":
            if not columns_for_pairplot:
                return "Error: 'columns_for_pairplot' is required for pairplot."
            if len(columns_for_pairplot) > 5:
                return "Warning: Pairplot with more than 5 columns can be very slow. Please select fewer columns."

            subset_df = df[columns_for_pairplot].select_dtypes(include=["number"])
            if subset_df.empty:
                return "Error: No numeric columns found in the specified list for pairplot."

            # seaborn.pairplot creates its own figure; tight_layout/save handled after
            g = sns.pairplot(
                subset_df.dropna(),
                hue=hue_column if hue_column and hue_column in df.columns else None,
            )
            g.fig.suptitle(title if title != "Generated Plot" else "Pair Plot", y=1.02)
            return _save_and_return(title if title != "Generated Plot" else "Pair Plot", "pairplot", "Pairwise relationships across selected numeric variables.")

        # BAR CHART
        if plot_type == "bar_chart":
            if not x_column:
                return "Error: 'x_column' is required for bar_chart."
            if x_column not in df.columns:
                return f"Error: Column '{x_column}' not found."

            use_counts = True
            if y_column and (y_column in df.columns) and pd.api.types.is_numeric_dtype(df[y_column]):
                use_counts = False

            if use_counts:
                # Frequency of x categories
                counts = df[x_column].astype(str).value_counts()
                plt.figure(figsize=(12, 6))
                sns.barplot(x=counts.index, y=counts.values, color="steelblue", alpha=0.85)
                plt.title(title if title != "Generated Plot" else f"Counts by {x_column}")
                plt.xlabel(x_column)
                plt.ylabel("Count")
                plt.xticks(rotation=45, ha="right")
                return _save_and_return(
                    title if title != "Generated Plot" else f"Counts by {x_column}",
                    "bar_chart",
                    f"Counts by {x_column}.",
                )
            else:
                # Direct bars of y over x (no aggregation)
                plt.figure(figsize=(12, 6))
                sns.barplot(x=df[x_column].astype(str), y=df[y_column], hue=df[hue_column] if hue_column and hue_column in df.columns else None)
                plt.title(title if title != "Generated Plot" else f"{y_column} by {x_column}")
                plt.xlabel(x_column)
                plt.ylabel(y_column)
                plt.xticks(rotation=45, ha="right")
                return _save_and_return(
                    title if title != "Generated Plot" else f"{y_column} by {x_column}",
                    "bar_chart",
                    f"Bar chart of {y_column} by {x_column}.",
                )

        return f"Error: Unknown plot_type '{plot_type}'."

    except Exception as e:
        cleanup_plot()
        return f"Error generating plot: {str(e)}"