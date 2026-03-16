from utils import LLM, execute_code, test_code_get_feedback, get_function_signature, get_test_cases, extract_code_solution, generate_and_extract_code
# import dotenv
# import os

# api_key = os.environ.get("OPENAI")
# dotenv.load_dotenv('.env')

from google.colab import userdata
api_key = userdata.get('OPENAI')

class MAS():
    def __init__(self, model_list):
        self.llm = LLM(model_list)

    def forward(self, taskInfo):
        # Define the task
        task = "Find the degree for the field extension Q(sqrt(2), sqrt(3), sqrt(18)) over Q"
        print("\n", taskInfo, "\n")
        # Get the function signature
        function_signature = get_function_signature(self.llm, taskInfo)

        # Get the test cases
        test_cases = get_test_cases(self.llm, taskInfo, function_signature)

        # Generate the code
        prompt = f"Write a Python function to find the degree for the field extension Q(sqrt(2), sqrt(3), sqrt(18)) over Q. The function should take no arguments."
        response, code = generate_and_extract_code(self.llm, prompt, temperature=0.7)
        print("\n", response, "\n")
        print("\n", code, "\n")

        # Execute the code
        try:
            output = execute_code(code)
        except Exception as e:
            print(f"Error executing code: {e}")
            return None

        # Return the output
        return output

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