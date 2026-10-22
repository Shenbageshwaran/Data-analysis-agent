# This agent imports funcctions from data_analysis_agent_functions.py and uses them to answer questions about data analysis. 
# This is version 1.0.
# This agent is designed to do EDA and report stats, summaries, distribution and correlation visualizations.

import pandas as pd
from dotenv import load_dotenv
import json
from openai import OpenAI
from data_analysis_agent_functions import (
    to_jsonable, inspect_data, describe_variables, plot_distribution,
    calculate_correlation, correlation_matrix, groupby_analysis,
)
from da_tools_schema import tools

load_dotenv()

client = OpenAI()
MODEL="gpt-6-astra"


# Load the data 
DF = pd.read_csv("data/test_data.csv")      # loaded once, by your code

# Map tool names to Python functions
FUNCTIONS = {
    "inspect_data": inspect_data,
    "describe_variables": describe_variables,
    "plot_distribution": plot_distribution,
    "calculate_correlation": calculate_correlation,
    "correlation_matrix": correlation_matrix,
    "groupby_analysis": groupby_analysis,
}


# 3. Route a requested call to the right function
def call_function(name, args):
    if name not in FUNCTIONS:
        return {"error": f"Unknown function: {name}"}
    try:
        return FUNCTIONS[name](DF, **args)  # df comes from you, the rest from the model
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


# 4. System prompt: how the analyst should behave
INSTRUCTIONS = """You are a careful data analyst working with one tabular dataset.

Rules:
- Call inspect_data first if you don't yet know the exact column names.
- Use only numbers returned by tools. Never calculate or estimate statistics yourself.
- If a tool returns an error, read it, fix your arguments, and try again.
- When reporting a statistic, include n and the confidence interval when available.
- When several correlations are tested, judge significance by the FDR-adjusted p-value.
- Correlation is not causation; say so if the user asks what 'causes' something.
- Mention sample-size warnings and the paths of any saved plots.
- Answer concisely and directly."""


# 5. Agent loop
def run_agent(user_prompt, max_steps=10):
    input_list = [{"role": "user", "content": user_prompt}]

    for step in range(max_steps):
        response = client.responses.create(
            model=MODEL,
            instructions=INSTRUCTIONS,
            input=input_list,
            tools=tools,
        )
        input_list += response.output          # keep the model's turn in history

        calls = [item for item in response.output if item.type == "function_call"]
        if not calls:
            return response.output_text        # no more tool requests means it's done

        for call in calls:
            args = json.loads(call.arguments)
            result = to_jsonable(call_function(call.name, args))
            output = json.dumps(result)

            print(f"[step {step}] {call.name}({args}) -> {output[:300]}"
                  + ("..." if len(output) > 300 else ""))

            input_list.append({
                "type": "function_call_output",
                "call_id": call.call_id,
                "output": output,
            })

    return "Stopped: max steps reached."

# 6. Ask questions
if __name__ == "__main__":
    print(run_agent("Find the variable most strongly correlated with body mass, then show how it differs by species."))