from typing import Literal
import pandas as pd
from dotenv import load_dotenv
from langchain.tools import tool
import data_analysis_agent_functions as daf

from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
import re



load_dotenv()

# DF = pd.read_csv("data/test_data.csv")      # use this line if running this python file
DF = None  # the DataFrame is passed in by the user in the dashboard

def set_data(df: pd.DataFrame):
    global DF
    DF = df


# Replaces call_function: adds the DataFrame, catches errors, makes the result JSON-safe
def _run(fn, **kwargs):
    if DF is None:
        return {"error": "No dataset loaded. Ask the user to upload a CSV file."}
    try:
        return daf.to_jsonable(fn(DF, **kwargs))
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


@tool
def inspect_data() -> dict:
    """Get an overview of the dataset: number of rows and columns, each column's
    data type and number of unique values, missing values per column, duplicate
    rows, constant columns, ID-like columns, and the first 5 rows.
    Call this FIRST before any other analysis, to learn the exact column names."""
    return _run(daf.inspect_data)


@tool(parse_docstring=True)
def calculate_correlation(
    col1: str,
    col2: str,
    method: Literal["pearson", "spearman"],
    group_by: str | None = None,
) -> dict:
    """Correlation between TWO numeric columns. Returns r, p-value, n, rows dropped
    for missing values, and a 95% confidence interval. If group_by is given,
    returns a separate result for each group. Do not use with categorical columns.

    Args:
        col1: First numeric column.
        col2: Second numeric column.
        method: 'pearson' for linear relationships, 'spearman' for monotonic or skewed data.
        group_by: Optional column to split the data by, e.g. 'species'.
    """
    return _run(daf.calculate_correlation, col1=col1, col2=col2,
                method=method, group_by=group_by)


@tool(parse_docstring=True)
def groupby_analysis(group_col: str, target_col: str) -> dict:
    """Compare a numeric variable across the groups of a categorical column. For
    each group returns n, mean, std, median, and the 95% confidence interval of
    the mean. Warns about groups with n < 5. Descriptive only, no significance test.

    Args:
        group_col: Categorical column that defines the groups, e.g. 'species'.
        target_col: Numeric column to summarize, e.g. 'body_mass_g'.
    """
    return _run(daf.groupby_analysis, group_col=group_col, target_col=target_col)

@tool(parse_docstring=True)
def describe_variables(columns: list[str]) -> dict:
    """Summary statistics for specific columns. Numeric columns: n, mean, std, min,
    quartiles, max, skewness, and count of IQR outliers. Categorical columns: n,
    number of unique values, most frequent value and its frequency.
    Use for questions about the typical values, spread, or shape of variables.

    Args:
        columns: Exact column names to describe, e.g. ['body_mass_g', 'flipper_length_mm'].
    """
    return _run(daf.describe_variables, columns=columns)


@tool(parse_docstring=True)
def plot_distribution(column: str) -> dict:
    """Plot a histogram with a density curve for ONE numeric column and save it as
    a PNG file. Returns the file path and n. Use when the user asks to see, plot,
    or visualize a variable's distribution or skewness.

    Args:
        column: Exact name of a numeric column.
    """
    return _run(daf.plot_distribution, column=column)


@tool(parse_docstring=True)
def correlation_matrix(
    columns: list[str] | None = None,
    method: Literal["pearson", "spearman"] = "pearson",
    save_heatmap: bool = False,
    top_k: int = 10,
) -> dict:
    """Compute correlations for ALL pairs of numeric columns at once. Returns the
    strongest pairs sorted by |r|, each with n, raw p-value, and FDR-adjusted
    p-value, plus pairs with |r| > 0.8 flagged as possibly redundant. Use for
    questions like 'which variables are most related' or to check multicollinearity.
    Prefer the FDR-adjusted p-value when judging significance.

    Args:
        columns: Columns to include. Leave empty to include all numeric columns.
        method: 'pearson' for linear relationships, 'spearman' for monotonic or skewed data.
        save_heatmap: True to also save a heatmap PNG of the correlation matrix.
        top_k: How many of the strongest pairs to return.
    """
    return _run(daf.correlation_matrix, columns=columns, method=method,
                save_heatmap=save_heatmap, top_k=top_k)
    


# print(describe_variables.invoke({"columns": ["body_mass_g", "species"]}))
# print(plot_distribution.invoke({"column": "body_mass_g"}))
# print(correlation_matrix.invoke({}))          # all defaults


INSTRUCTIONS = """You are a careful data analyst working with a tabular dataset.

Rules:
- Call inspect_data first if you don't yet know the exact column names.
- Use only numbers returned by tools. Never calculate or estimate statistics yourself.
- If a tool returns an error, read it, fix your arguments, and try again.
- When reporting a statistic, include n and the confidence interval when available.
- When several correlations are tested, judge significance by the FDR-adjusted p-value.
- Correlation is not causation; say so if the user asks what 'causes' something.
- Mention sample-size warnings and the paths of any saved plots.
- Answer concisely and directly."""
   # copy your INSTRUCTIONS string from data_analysis_agent.py unchanged

tools = [
    inspect_data, describe_variables, plot_distribution,
    calculate_correlation, correlation_matrix, groupby_analysis,
]

model = init_chat_model("openai:gpt-6-astra")
agent = create_agent(model=model, tools=tools, system_prompt=INSTRUCTIONS, checkpointer=InMemorySaver())


def run_agent(user_prompt, thread_id='default'):
    config = {"configurable": {"thread_id": thread_id, "recursion_limit": 20}}  # about the same as the max_steps=10
    result = agent.invoke(
        {"messages": [{"role": "user", "content": user_prompt}]},
        config
    )
    msgs = result["messages"]
    last_user = max(i for i,m in enumerate(msgs) if m.type == "human") # find the last user message
    this_turn = msgs[last_user+1:] # all messages after that, which may include multiple tool calls and assistant messages

    tool_log, plots = [], []
    for m in this_turn:
        for call in getattr(m, "tool_calls", None) or []:
            tool_log.append(f"{call['name']}({call['args']})")
        if m.type == 'tool':
            plots += re.findall(r"plots/[\w\-]+\.png", str(m.content))  # find any plot paths in the tool's output

    return result["messages"][-1].text, tool_log, plots


if __name__ == "__main__":
    set_data(pd.read_csv("data/test_data.csv"))
    tid = "test-1"
    print(run_agent("Which variable is most strongly correlated with body mass?", tid)[0])
    print(run_agent("Now show how that variable differs by species.", tid)[0])

    