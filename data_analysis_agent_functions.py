# This file contains data analysis agent functions that can be called by the agent loop in data_analysis_agent.py.
# This file covers exploratory data analysis (EDA) and data visualization functions.

# The functions in this file are:
# 1. inspect_data: Inspect the DataFrame for basic information, missing values, and data types.
# 2. describe variables
# 3. plot distribution
# 4. calculate correlations
# 5. correlation matrix
# 6. groyupby analysis

import matplotlib
import pandas as pd
import numpy as np
matplotlib.use('Agg')  # Use a non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns
import os
# from sklearn.preprocessing import StandardScaler, MinMaxScaler
# from sklearn.decomposition import PCA
from scipy.stats import pearsonr, spearmanr
from itertools import combinations
from statsmodels.stats.multitest import multipletests
from scipy import stats


# ---------- Helper: make results JSON-safe ----------
def to_jsonable(obj, sig=4):
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v, sig) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v, sig) for v in obj]
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float):
        return None if not np.isfinite(obj) else float(f"{obj:.{sig}g}")
    if isinstance(obj, (str, int, bool)) or obj is None:
        return obj
    return str(obj) 

# BOnnet-Wright approximation for confidence intervals of correlation coefficients
def _corr_ci(r, n, method):
    if n < 4 or abs(r) >= 1:
        return [None, None]
    z = np.arctanh(r)
    se = 1 / np.sqrt(n - 3)
    if method == "spearman":                      # Bonett–Wright approximation
        se = np.sqrt((1 + r**2 / 2) / (n - 3))
    return [float(np.tanh(z - 1.96 * se)), float(np.tanh(z + 1.96 * se))]


# 1. Inspect DataFrame: 
# Information collected from this function
# Returns number of rows and columns, with column names, data type of each column, and number of unique values.
# Includes a missing value count and percentage of missing values for each column.
# Includes a duplicate rows, and a sample of the first 5 rows of the DataFrame.
# Flags constant columns, like one value columns, and ID-lie columns (a different value in every row) 
def inspect_data(df: pd.DataFrame) -> dict:
    
    n, d = df.shape
    # Get column names, data types, and number of unique values
    column_info = {col: {"dtype": str(df[col].dtype), "unique": df[col].nunique()} for col in df.columns}
    # Get missing value count and percentage
    missing_info = {col: {"missing_count": df[col].isnull().sum(), "missing_percentage": df[col].isnull().mean() * 100} for col in df.columns}
    # Get duplicate rows count
    duplicate_count = df.duplicated().sum()
    # Get a sample of the first 5 rows
    sample_rows = df.head().to_dict(orient="records")
    # Flag constant columns
    constant_columns = [col for col in df.columns if df[col].nunique() == 1]
    id_like_columns = [
        c for c in df.columns
        if df[c].nunique() == len(df)
        and (pd.api.types.is_integer_dtype(df[c]) or pd.api.types.is_string_dtype(df[c]))
    ]
    return {
        "num_rows": n,
        "num_columns": d,
        "column_info": column_info,
        "missing_info": missing_info,
        "duplicate_count": duplicate_count,
        "sample_rows": sample_rows,
        "constant_columns": constant_columns,
        "id_like_columns": id_like_columns
    }

# 2. Describe Variables:
# For numeric columns, returns n, mean, std, min, 25/50.75th quantiles , max, skewness, and the # of IQR outliers.
# For categorical columns, returns n, unique, top, and freq.
# If a column doesnt exisit, returns an error that lists the available columns.
def describe_variables(df: pd.DataFrame, columns: list) -> dict:
    description = {}
    for col in columns:
        if col not in df.columns:
            description[col] = {"error": f"Column '{col}' does not exist. Available columns: {list(df.columns)}"}
            continue

        if pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_bool_dtype(df[col]):
            desc = df[col].describe()
            iqr = desc['75%'] - desc['25%']
            outliers = ((df[col] < (desc['25%'] - 1.5 * iqr)) | (df[col] > (desc['75%'] + 1.5 * iqr))).sum()
            description[col] = {
                "n": int(desc['count']),
                "mean": float(desc['mean']),
                "std": float(desc['std']),
                "min": float(desc['min']),
                "25%": float(desc['25%']),
                "50%": float(desc['50%']),
                "75%": float(desc['75%']),
                "max": float(desc['max']),
                "skewness": float(df[col].skew()),
                "iqr_outliers": int(outliers)
            }
        else:
            desc = df[col].describe()
            description[col] = {
                "n": int(desc['count']),
                "unique": int(desc['unique']),
                "top": str(desc['top']),
                "freq": int(desc['freq'])
            }
    return description

# 3. Plot Distribution:
# Save a PNH file to a plots/ folder and returns the {"saved_to": "plots/filename.png"}.
# Always include axis labels and titles
# If column is not numeric, return an error message.
def plot_distribution(df: pd.DataFrame, column: str) -> dict:
    if column not in df.columns:
        return {"error": f"Column '{column}' does not exist. Available columns: {list(df.columns)}"}
    if not pd.api.types.is_numeric_dtype(df[column]):
        return {"error": f"Column '{column}' is not numeric. Cannot plot distribution."}
    
    plt.figure(figsize=(10, 6))
    sns.histplot(df[column].dropna(), kde=True, bins=30)
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    
    # filename = f'plots/{column}_distribution.png'
    os.makedirs("plots", exist_ok=True)
    safe = "".join(ch if ch.isalnum() else "_" for ch in column)
    path = f"plots/{safe}_distribution.png"
    plt.savefig(path)
    plt.close()

    return {"saved_to": path, "n": int(df[column].notna().sum())}

# 4. Calculate Correlations:
# Returns r, p-values, n and a 95% confidence interval
# If group_by is given, returns one result per group. This answers questions like 'Does it differe between two groups?'
# Safegaurds: 1. Return an error if either column is categorical. 2. Drop rows with missing values pairwise, and report how many were dropped. 
# 3. Add a warning if n < 10


def calculate_correlation(df: pd.DataFrame, col1: str, col2: str, method: str = 'pearson', group_by: str = None) -> dict:
    # Return an error if the method isnt pearson or spearman
    if method not in ['pearson', 'spearman']:
        return {"error": f"Method '{method}' is not supported. Use 'pearson' or 'spearman'."}

    if col1 not in df.columns or col2 not in df.columns:
        return {"error": f"One or both columns '{col1}', '{col2}' do not exist. Available columns: {list(df.columns)}"}
    if not pd.api.types.is_numeric_dtype(df[col1]) or not pd.api.types.is_numeric_dtype(df[col2]):
        return {"error": f"One or both columns '{col1}', '{col2}' are not numeric. Cannot calculate correlation."}
    # Return an error if group_by is given but isn’t in df.columns
    if group_by and group_by not in df.columns:
        return {"error": f"Group by column '{group_by}' does not exist. Available columns: {list(df.columns)}"}

    
    
    if group_by and group_by in df.columns:
        results = {}
        for group, group_df in df.groupby(group_by):
            n_before = len(group_df)
            group_df = group_df[[col1, col2]].dropna()
            n_after = len(group_df)
            n_dropped = n_before - n_after
            
            if len(group_df) < 10:
                warning = "Warning: Sample size is less than 10."
            else:
                warning = None
            # Before calling pearsonr or spearmanr, return an error if n < 3, or skip that group with a note
            if len(group_df) < 3:
                return {"error": f"Group '{group}' has fewer than 3 observations. Cannot calculate correlation."}
            if method == 'pearson':
                r, p_value = pearsonr(group_df[col1], group_df[col2])
            elif method == 'spearman':
                r, p_value = spearmanr(group_df[col1], group_df[col2])
            else:
                return {"error": f"Method '{method}' is not supported. Use 'pearson' or 'spearman'."}
            
            ci_low, ci_high = _corr_ci(r, len(group_df), method)
            
            results[group] = {
                "n": len(group_df),
                "n_dropped": n_dropped,
                "r": r,
                "p_value": p_value,
                "95%_ci": [ci_low, ci_high],
                "warning": warning
            }
        return results
    else:
        n_before = len(df)
        df_clean = df[[col1, col2]].dropna()
        n_after = len(df_clean)
        n_dropped = n_before - n_after

        if len(df_clean) < 3:
            return {"error": f"Only {len(df_clean)} complete rows. Need at least 3 to calculate correlation."}
        if len(df_clean) < 10:
            warning = "Warning: Sample size is less than 10."
        else:
            warning = None

        if method == 'pearson':
            r, p_value = pearsonr(df_clean[col1], df_clean[col2])
        elif method == 'spearman':
            r, p_value = spearmanr(df_clean[col1], df_clean[col2])
        else:
            return {"error": f"Method '{method}' is not supported. Use 'pearson' or 'spearman'."}

        ci_low, ci_high = _corr_ci(r, len(df_clean), method)

        return {
            "n": len(df_clean),
            "n_dropped": n_dropped,
            "r": r,
            "p_value": p_value,
            "95%_ci": [ci_low, ci_high],
            "warning": warning
        }

# 5. Correlation Matrix:
# COmputes all pairwise correlations and returns the top pairs sorted by |r|, each with its raw p-value and FDR adjusted p-value.
# Flags pairs with |r| > 0.8 as possible redundant, but doesn't remove anything.
# Optionaly saves a heatmap.

def correlation_matrix(df: pd.DataFrame, columns: list = None, method: str = 'pearson',
                       save_heatmap: bool = False, top_k: int = 10) -> dict:
    if method not in ("pearson", "spearman"):
        return {"error": f"Method '{method}' is not supported. Use 'pearson' or 'spearman'."}

    # Choose columns
    if columns:
        missing = [c for c in columns if c not in df.columns]
        if missing:
            return {"error": f"Columns {missing} do not exist. Available columns: {list(df.columns)}"}
        numeric_df = df[columns].select_dtypes(include=[np.number])
    else:
        numeric_df = df.select_dtypes(include=[np.number])

    cols = list(numeric_df.columns)
    if len(cols) < 2:
        return {"error": "Need at least 2 numeric columns."}

    # One test per unique pair, with aligned rows (pairwise deletion)
    func = pearsonr if method == "pearson" else spearmanr
    pairs = []
    for a, b in combinations(cols, 2):
        sub = numeric_df[[a, b]].dropna()
        if len(sub) < 3:
            continue
        r, p = func(sub[a], sub[b])
        pairs.append({"pair": [a, b], "n": len(sub), "r": float(r), "p_value": float(p)})

    if not pairs:
        return {"error": "No column pair had at least 3 complete rows."}

    # FDR correction over unique pairs only
    _, fdr, _, _ = multipletests([x["p_value"] for x in pairs], method="fdr_bh")
    for x, q in zip(pairs, fdr):
        x["fdr_adjusted_p_value"] = float(q)
        x["redundant_flag"] = abs(x["r"]) > 0.8

    pairs.sort(key=lambda x: abs(x["r"]), reverse=True)

    result = {
        "method": method,
        "num_pairs_tested": len(pairs),
        "top_pairs": pairs[:top_k],
        "redundant_pairs": [x for x in pairs if x["redundant_flag"]],
    }

    if save_heatmap:
        os.makedirs("plots", exist_ok=True)
        plt.figure(figsize=(12, 10))
        sns.heatmap(numeric_df.corr(method=method), annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
        plt.title(f"Correlation Matrix Heatmap ({method})")
        plt.tight_layout()
        plt.savefig("plots/correlation_matrix_heatmap.png")
        plt.close()
        result["heatmap_saved_to"] = "plots/correlation_matrix_heatmap.png"

    return result

# 6. groupby analysis:
# Returns n, mean, std, median, and the 95% confience interval of the mean for each group. 
# Safegaurd: warn if any group has n < 5.

def groupby_analysis(df: pd.DataFrame, group_col: str, target_col: str) -> dict:
    if group_col not in df.columns or target_col not in df.columns:
        return {"error": f"One or both columns '{group_col}', '{target_col}' do not exist. Available columns: {list(df.columns)}"}
    if not pd.api.types.is_numeric_dtype(df[target_col]):
        return {"error": f"Target column '{target_col}' is not numeric. Cannot perform groupby analysis."}
    
    grouped = df.groupby(group_col)[target_col].agg(['count', 'mean', 'std', 'median'])
    t_crit = stats.t.ppf(0.975, grouped['count'] - 1)
    se = grouped['std'] / np.sqrt(grouped['count'])
    grouped['95%_ci_low'] = grouped['mean'] - t_crit * se
    grouped['95%_ci_high'] = grouped['mean'] + t_crit * se
    
    warnings = {}
    for group, row in grouped.iterrows():
        if row['count'] < 5:
            warnings[group] = "Warning: Sample size is less than 5."
    
    result = grouped.reset_index().to_dict(orient='records')
    
    
    return {
        "groupby_analysis": result,
        "n_rows_missing_group_label": int(df[group_col].isna().sum()),
        "warnings": warnings
    }
