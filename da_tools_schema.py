# 2. Tools - Functions with metadata for the model
tools = [
    {
        "type": "function",
        "name": "inspect_data",
        "description": (
            "Get an overview of the dataset: number of rows and columns, each column's "
            "data type and number of unique values, missing values per column, duplicate "
            "rows, constant columns, ID-like columns, and the first 5 rows. "
            "Call this FIRST before any other analysis, to learn the exact column names."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "describe_variables",
        "description": (
            "Summary statistics for specific columns. Numeric columns: n, mean, std, min, "
            "quartiles, max, skewness, and count of IQR outliers. Categorical columns: n, "
            "number of unique values, most frequent value and its frequency. "
            "Use for questions about the typical values, spread, or shape of variables."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "columns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Exact column names to describe, e.g. ['Area', 'Diameter'].",
                },
            },
            "required": ["columns"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "plot_distribution",
        "description": (
            "Plot a histogram with a density curve for ONE numeric column and save it as "
            "a PNG file. Returns the file path and n. Use when the user asks to see, plot, "
            "or visualize a variable's distribution or skewness."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "column": {
                    "type": "string",
                    "description": "Exact name of a numeric column.",
                },
            },
            "required": ["column"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "calculate_correlation",
        "description": (
            "Correlation between TWO numeric columns. Returns r, p-value, n, rows dropped "
            "for missing values, and a 95% confidence interval. If group_by is given, "
            "returns a separate result for each group (e.g. each Day). Use 'pearson' for "
            "linear relationships and 'spearman' for monotonic or skewed data. "
            "Do not use with categorical columns."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "col1": {"type": "string", "description": "First numeric column."},
                "col2": {"type": "string", "description": "Second numeric column."},
                "method": {
                    "type": "string",
                    "enum": ["pearson", "spearman"],
                    "description": "Correlation method.",
                },
                "group_by": {
                    "type": ["string", "null"],
                    "description": "Optional column to split the data by. Use null for no grouping.",
                },
            },
            "required": ["col1", "col2", "method", "group_by"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "correlation_matrix",
        "description": (
            "Compute correlations for ALL pairs of numeric columns at once. Returns the "
            "strongest pairs sorted by |r|, each with n, raw p-value, and FDR-adjusted "
            "p-value, plus pairs with |r| > 0.8 flagged as possibly redundant. "
            "Use for questions like 'which variables are most related' or to check "
            "multicollinearity. Prefer the FDR-adjusted p-value when judging significance."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "columns": {
                    "type": ["array", "null"],
                    "items": {"type": "string"},
                    "description": "Columns to include. Use null to include all numeric columns.",
                },
                "method": {
                    "type": "string",
                    "enum": ["pearson", "spearman"],
                    "description": "Correlation method.",
                },
                "save_heatmap": {
                    "type": "boolean",
                    "description": "True to also save a heatmap PNG of the correlation matrix.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "How many of the strongest pairs to return. Use 10 if unsure.",
                },
            },
            "required": ["columns", "method", "save_heatmap", "top_k"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "groupby_analysis",
        "description": (
            "Compare a numeric variable across the groups of a categorical column. For "
            "each group returns n, mean, std, median, and the 95% confidence interval of "
            "the mean (t-distribution). Warns about groups with n < 5. Use for questions "
            "like 'how does X differ between conditions/days/cell lines'. This is "
            "descriptive only and does not run a significance test."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "group_col": {
                    "type": "string",
                    "description": "Categorical column that defines the groups, e.g. 'Day'.",
                },
                "target_col": {
                    "type": "string",
                    "description": "Numeric column to summarize, e.g. 'SM_Index'.",
                },
            },
            "required": ["group_col", "target_col"],
            "additionalProperties": False,
        },
        "strict": True,
    },
]