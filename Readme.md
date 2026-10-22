# Data Analysis Agent

A small LLM agent that answers questions about a tabular dataset in plain English.

I built this to understand how tool-calling agents actually work, so there's no agent framework here. It's the OpenAI Responses API, a loop of about 40 lines, and six analysis functions that I wrote and tested myself. The model decides *which* analysis to run. Python does the actual math.

## Why I built it this way

My background is in ML and statistical modeling, and I wanted to learn agentic AI from the ground up instead of starting with LangChain or LangGraph. Writing the loop by hand made the moving parts clear to me:

- the **functions** are ordinary Python,
- the **tool definitions** (JSON Schema) are what the model reads,
- the **model** never runs anything; it only asks for a function call with arguments,
- my **code** runs the function and sends the result back, and this repeats until the model has an answer.

One design decision shaped everything else: **the LLM chooses the analysis, but it never computes statistics itself.** Every number in an answer has to come from a tool, and the statistical rules (confidence intervals, multiple-testing correction, sample-size checks) are written into the functions rather than left to the prompt.

## How it works

```
question -> model picks tool(s) -> my code runs them -> results go back to the model
                ^                                                  |
                |__________________ repeat until done _____________|
```

The dataset is loaded once by my code. The model never sees the raw data; it only sees column names and tool results. It passes column names as arguments, and the dispatcher supplies the DataFrame.

## Tools

| Tool | What it does |
|---|---|
| `inspect_data` | Rows/columns, dtypes, missing values, duplicates, constant and ID-like columns, first 5 rows |
| `describe_variables` | Summary stats and skewness for numeric columns; counts and top value for categorical ones |
| `plot_distribution` | Histogram with a density curve, saved as a PNG |
| `calculate_correlation` | Pearson or Spearman r with n, p-value, and 95% CI, optionally split by group |
| `correlation_matrix` | All pairwise correlations, ranked by \|r\|, with FDR-adjusted p-values and redundancy flags |
| `groupby_analysis` | n, mean, std, median, and a t-based 95% CI of the mean for each group |

## Statistical safeguards

These live in the code, not in the prompt:

- 95% confidence intervals for correlations use the Fisher z-transform (with the Bonett–Wright adjustment for Spearman).
- Group mean confidence intervals use the t-distribution, not 1.96, since some groups are small.
- `correlation_matrix` applies Benjamini–Hochberg FDR correction across all pairs, so testing many pairs doesn't produce false "significant" results.
- Missing values are dropped pairwise, and the number dropped is reported.
- Every result includes `n`, and small samples trigger a warning.
- Bad requests (wrong column names, categorical columns in a correlation) return an error message instead of crashing. The model reads the error and corrects itself.

## Example

I tested it on the [Palmer Penguins](https://allisonhorst.github.io/palmerpenguins/) dataset.

**Question:** *"Find the variable most strongly correlated with body mass, then show how that variable differs by species."*

The log shows it working in two rounds: first a correlation matrix, then a follow-up that depends on the first result.

```
[step 0] inspect_data({})
[step 1] correlation_matrix({'columns': [...], 'method': 'pearson', ...})
[step 2] calculate_correlation({'col1': 'flipper_length_mm', 'col2': 'body_mass_g', ...})
[step 2] groupby_analysis({'group_col': 'species', 'target_col': 'flipper_length_mm'})
```

It reported flipper length as the strongest correlate (r = 0.871, 95% CI 0.843–0.895, n = 342), showed Gentoo penguins have the longest flippers, and pointed out that the pooled correlation may partly reflect differences between species.

The behavior I cared most about was how it handled requests it *shouldn't* fulfill:

- Asked for a Pearson correlation between `species` and body mass, it explained why that isn't valid for a categorical variable and suggested a group comparison instead.
- Asked what *causes* penguins to be heavier, it reported the associations and said plainly that the data can't establish causes.

## Setup

```bash
git clone https://github.com/Shenbageshwaran/data-analysis-agent.git
cd data-analysis-agent
python -m venv .venv
.venv\Scripts\activate          # Windows; on macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add your own OpenAI API key. Then:

```bash
python data_analysis_agent.py
```

Change the question at the bottom of `data_analysis_agent.py` to ask something else. Plots are saved to `plots/`.

## Project structure

```
data_analysis_agent.py            # agent loop, dispatcher, system instructions
data_analysis_agent_functions.py  # the six analysis tools + JSON helper
da_tools_schema.py                # JSON Schema tool definitions
data/test_data.csv                # Palmer Penguins
examples/agent.py                 # my first toy agent (add + fake weather)
```

`examples/agent.py` is the two-function agent I wrote first, to understand tool calling before working with real data. I've kept it because it's the simplest version of the same loop.

## Limitations

- Each question starts fresh. There's no conversation memory yet, so "now do the same for island" won't work.
- There are no significance tests between groups yet. The agent says so instead of making up a p-value.
- `plot_distribution` can't split by group, so a per-species histogram isn't possible yet.
- When asked to do something it has no tool for (like training a model), it sometimes writes code for you instead of just saying it can't.

## What's next

- `compare_groups`: t-test / Mann–Whitney / ANOVA / Kruskal–Wallis, with effect sizes
- grouped plots (box, violin, per-group histograms)
- an evaluation script that runs a fixed set of questions and records which tools were called and whether the answer was right
- conversation memory
- a LangGraph version of the same agent, to compare with the hand-written loop

## Data

Palmer Penguins data from Horst, Hill & Gorman (2020), collected by Dr. Kristen Gorman at Palmer Station, Antarctica. Released under CC0.
