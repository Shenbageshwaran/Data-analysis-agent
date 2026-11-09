# Data Analysis Agent

An LLM agent that answers questions about a CSV dataset. The user uploads a csv file, and asks a question, particularly regarding exploratory data analysis.
The agent decides which functions to execute from the available tools (functions). 

I started this project to learn how tool-calling agents work. The first version has no framework, just the OpenAI Responses API and a short loop. Once that worked, I rebuilt it with LangChain and added a Streamlit dashboard and conversation memory.

## Dashboard Preview
![1.](image.png)
![2.](image-1.png)

## How it works
1. The user uploads the csv file, and asks a question to the model.
2. The model reads the question, and decides which analysis to run from the available tools.
3. My code runs the selected tool on the dataset and sends the result back to the model.
4. The user can ask follow-up questions, because the app saves the conversation for the current session.

One important thing is that the model never sees the raw data, only column names and tool results. Every number in the the model's response must come from the pre-defined tools.

## Tools

| Tool | Description |
|---|---|
| `inspect_data` | Rows, columns, data types, missing values, duplicates |
| `describe_variables` | Summary stats for numeric and categorical columns |
| `plot_distribution` | Histogram with a density curve |
| `calculate_correlation` | Pearson or Spearman correlation, optionally by group |
| `correlation_matrix` | All pairwise correlations, ranked, with FDR correction |
| `groupby_analysis` | Mean, std, median, and 95% CI for each group |

## Statistics

I put the statistical checks inside the functions instead of the prompt, so the agent can't skip them.

- Correlations always come with n and a 95% confidence interval.
- `correlation_matrix` uses Benjamini-Hochberg FDR correction, since it tests many pairs at once.
- Group means use t-based confidence intervals because some groups are small.
- Small samples get a warning.
- Bad requests, like a wrong column name, return an error message. The model reads it and fixes its call.


## Example Q/A discussion with the model

I tested it on the Palmer Penguins dataset.

**Question:** "Find the variable most strongly correlated with body mass, then show how it differs by species."

It ran a correlation matrix and found flipper length (r = 0.871, 95% CI 0.843 to 0.895, n = 342). Then it compared flipper length across species. It also pointed out that the overall correlation could partly come from differences between species, which I thought was a good catch.

When I asked what *causes* penguins to be heavier, it said the data only shows associations, not causes. That was the behavior I wanted.

## How to run locally
```bash
git clone https://github.com/Shenbageshwaran/data-analysis-agent.git
cd data-analysis-agent
python -m venv .venv
.venv\Scripts\activate          # on macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```
Create a new file named .env, and add your OPENAI API key to the file.
Example code in the .env file:    OPENAI_API_KEY=your-key-here

To run the dashboard: 
```bash
streamlit run app.py
```
To run the first version in the terminal:

```bash
python data_analysis_agent.py
```
# Files

- `app.py`: Streamlit dashboard
- `data_analysis_agent_langchain_v2.py`: LangChain agent with memory
- `data_analysis_agent.py`: first version with the hand-written loop
- `data_analysis_agent_functions.py`: the six analysis tools
- `da_tools_schema.py`: tool definitions for the first version
- `data/test_data.csv`: Palmer Penguins data
- `examples/agent.py`: my first toy agent, kept as the simplest example

## Limitations

- Memory only lasts while the app is running. Restarting it clears the chat.
- No significance tests between groups yet. I'm planning a `compare_groups` tool next.
- It can't make plots split by group yet.

## AI use disclosure
I used Claude to help with parts of the code and this README. The analysis functions and the design decisions are my own.

## Data
The Palmer Penguins data is from Horst, Hill & Gorman (2020), collected by Dr. Kristen Gorman at Palmer Station, Antarctica (CC0).