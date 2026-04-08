from google.colab import userdata
api_key = userdata.get('OPENAI')
from utils import *

class MAS():
    def __init__(self, model_list):
        self.llm = LLM(model_list)

    def call_llm(self, prompt):
        response, _ = generate_and_extract_code(self.llm, prompt)
        return response

    def forward(self, query):

        # 1️⃣ math agent
        math_prompt = (
            "You are a math expert.\n"
            "Solve the following problem step by step:\n"
            f"{query}"
        )
        math_output = self.call_llm(math_prompt)

        # 2️⃣ feedback agent
        feedback_prompt = (
            "You are a strict reviewer.\n"
            "Check the correctness of the solution.\n"
            f"Problem: {query}\n"
            f"Solution: {math_output}\n"
            "If wrong, explain why."
        )
        feedback_output = self.call_llm(feedback_prompt)

        # 3️⃣ refine agent
        refine_prompt = (
            "You are a math expert.\n"
            "Given the problem, solution, and feedback,\n"
            "provide the correct final answer.\n\n"
            f"Problem: {query}\n"
            f"Solution: {math_output}\n"
            f"Feedback: {feedback_output}\n"
            "Give only the final answer."
        )
        final_output = self.call_llm(refine_prompt)

        return final_output

# Initialize the multi-agent system
mas = MAS([ ("gpt-4o-mini", "https://api.openai.com/v1", api_key)
,  ("gpt-4o-mini", "https://api.openai.com/v1", api_key)
,  ("gpt-4o-mini", "https://api.openai.com/v1", api_key)
])


# Define the task info
taskInfo = {"task": "field_extension_degree", "fields": ["Q(sqrt(2), sqrt(3), sqrt(18))", "Q"]}

# Run the multi-agent system
output = mas.forward(taskInfo)

# Print the output
if output is not None:
    print("The degree for the field extension Q(sqrt(2), sqrt(3), sqrt(18)) over Q is:", output)
else:
    print("Failed to find the degree for the field extension Q(sqrt(2), sqrt(3), sqrt(18)) over Q.")
# This code defines a multi-agent system that uses a large language model to generate a Python function to find the degree for the field extension Q(sqrt(2), sqrt(3), sqrt(18)) over Q. The function is then executed using the `execute_code` function, and the output is printed to the console.

# Note that this code assumes that the `utils` library has been implemented correctly and provides the necessary functions for generating and executing code. The `LLM` class is also assumed to be implemented correctly and provides the necessary functionality for interacting with the large language model.