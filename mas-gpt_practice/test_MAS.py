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
        function_signature = get_function_signature(self.llm, task)

        # Get the test cases
        test_cases = get_test_cases(self.llm, task, function_signature)

        # Generate the code
        # prompt = f"Write a Python function to find the degree for the field extension Q(sqrt(2), sqrt(3), sqrt(18)) over Q. The function should take no arguments."
        # prompt = """
        # Write a Python function to find the degree for the field extension Q(sqrt(2), sqrt(3), sqrt(18)) over Q.
        # The function should take no arguments and return the degree as an integer.
        
        # Return only Python code and do not include any explanation.
        # The Python code should be able to take the query as input and return the answer as output.
        # The code should be able to run without any errors.
        # Please make sure the code is well-structured and easy to understand.
        
        # Wrap the entire code inside the following tags exactly:
        
        # <Code Solution>
        # # your python code here
        # </Code Solution>
        # """
        
        # prompt = """
        # Write Python code that solves the following task:
        
        # Task:
        # Find the degree of the field extension Q(sqrt(2), sqrt(3), sqrt(18)) over Q.
        
        # Requirements:
        # 2. The function must take no arguments.
        # 3. The function must return the degree as an integer.
        # 4. Do not print anything.
        # 5. Return only valid Python code.
        # 6. Do not include any explanation, comments, or markdown.
        # 7. Wrap the entire code exactly in the following tags:
        
        # <Code Solution>
        # # your python code here
        # </Code Solution>
        # """
        
        prompt = """
        You are a helpful AI assistant tasked with extracting the final answer from a provided solution.
        **Input:**
        1. A problem statement, prefixed with ”===Problem: <problem>”.
        2. A solution to the problem, prefixed with ”===Solution:<solution>”.
        **Problem and Solution:**
        ===Problem: {query}
        ===Solution: {response}
        **Instructions:**
        - Carefully analyze the solution and extract the final answer in reply: ”The answer is <answer extracted> in reply”.
        - If the solution does not contain a final answer (e.g., only reasoning, code without execution, or incomplete information), respond with: ”The reply doesn’t contain an answer.”
        - Ensure that the extracted answer is exactly as presented in the solution. Do not infer or use external knowledge. Do not execute the code yourself.
        - Remember, Never execute the code yourself! Never doing any computation yourself! Just extract and output the existing answer!
        """
                
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