from dotenv import load_dotenv
import json
from openai import OpenAI

load_dotenv()

client = OpenAI()
MODEL="gpt-6-astra"

# 1. Functions - Pure Python
def add(a: float, b: float) -> float:
    return a + b

def get_weather(city: str) -> dict:
    # Placeholder for a function that would fetch weather data
    return {"city": city, "temp_c": 25, "condition": "sunny"}

# 2. Tools - Functions with metadata for the model
tools = [{
    "type": "function",
    "name": "add",
    "description": "Adds two numbers together and returns the sum.",
    "parameters": {
        "type": "object",
        "properties": {
            "a": {"type": "number", "description": "The first number to add."},
            "b": {"type": "number", "description": "The second number to add."}
        },
        "required": ["a", "b"],
        "additionalProperties": False,
    }, "strict": True, 
    },
    {
        "type": "function",
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name, e.g. Auburn, AL."}
            },
            "required": ["city"],
            "additionalProperties": False,
        }, "strict": True, 
    }
]

# 3. Route a requested call to the right function

FUNCTIONS = {"add": add, "get_weather": get_weather}

def call_function(name, args):
    if name not in FUNCTIONS:
        return {"error": f"Function '{name}' not found."}
    return FUNCTIONS[name](**args)

# 4. Agent loop - continuously process messages and call functions as needed

def run_agent(user_prompt, max_steps=5):
    input_list = [{"role": "user", "content": user_prompt}]
    
    for step in range(max_steps):
        print(step)
        response = client.responses.create(model=MODEL, input=input_list, tools=tools)
        # print('response:')
        # print(response.model_dump_json(indent=2))  # Print the model's response
        # print(response.output)

        input_list += response.output # Keep the model's turn in history
        # print('input_list:')
        # print(input_list)
        calls = [item for item in response.output if item.type == 'function_call']

        if not calls:
            # No function calls, return the model's response
            return response.output_text

        for call in calls:
            args = json.loads(call.arguments)
            result = call_function(call.name, args)
            print(f"[step {step}] {call.name}({args}) -> {result}")

            input_list.append({"type": "function_call_output", "call_id": call.call_id, "output": json.dumps(result)})

    return "Stopped: Max steps reached"

# This sentence is an example question to the agent.
# It will call the add function and the get_weather function in sequence.
# The model will handle the function calls and return the results in a single response.
# print(run_agent("What's 17.5 + 24.5, and what's the weather in Paris?"))
# print(run_agent("What is 25 + 35, and explain what a neural network is."))
print(run_agent("Add 2 and 3, and then add 5 to it"))