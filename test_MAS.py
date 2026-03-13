from utils import LLM, execute_code, test_code_get_feedback, get_function_signature, get_test_cases, extract_code_solution, generate_and_extract_code

class MAS():
    def __init__(self, model_list):
        self.llm = LLM(model_list)

    def forward(self, taskInfo):
        # Define the task
        task = "Find the degree for the field extension Q(sqrt(2), sqrt(3), sqrt(18)) over Q"

        # Get the function signature
        function_signature = get_function_signature(self.llm, taskInfo)

        # Get the test cases
        test_cases = get_test_cases(self.llm, taskInfo, function_signature)

        # Generate the code
        prompt = f"Write a Python function to find the degree for the field extension Q(sqrt(2), sqrt(3), sqrt(18)) over Q. The function should take no arguments."
        response, code = generate_and_extract_code(self.llm, prompt, temperature=0.7)

        # Execute the code
        try:
            output = execute_code(code)
        except Exception as e:
            print(f"Error executing code: {e}")
            return None

        # Return the output
        return output

# Initialize the multi-agent system
mas = MAS(["model1", "model2", "model3"])

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

# Here is the code for the `LLM` class:

class LLM():

    def __init__(self, model_list):
        self.model_list = model_list
    
    @retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(5), retry_error_callback=handle_retry_error)
    def call_llm(self, prompt, temperature=0.5):
        model_name, model_url, api_key = random.choice(self.model_list)
        llm = openai.OpenAI(base_url=f"{model_url}", api_key=api_key)
        try:
            completion = llm.chat.completions.create(
                model=f"{model_name}",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                stop=['<|eot_id|>'],
                temperature=temperature,
                max_tokens=2048,
                timeout=600
            )
            raw_response = completion.choices[0].message.content
            # remove the think part for reasoning models such as deepseek-r1
            final_response = re.sub(r'<think>.*?</think>', '', raw_response, flags=re.DOTALL).strip()
            return final_response
        except Exception as e:
            logging.error(f"[Request Error] {e}")
            raise e
        
# And here is the code for the `get_function_signature` function:

# Generate the function signature based on the task info
# This is a placeholder function and should be implemented correctly
def get_function_signature(llm, taskInfo):
    """
    Generate a Python function signature based on the problem description.

    Args:
        taskInfo (str): The problem description.

    Returns:
        str: The function signature with an appropriate docstring.
    """
    # Generates an instruction prompt by formatting the FUNCTION_SIGNATURE_DESIGNER_PROMPT with the task information
    function_signature_designer_instruction = FUNCTION_SIGNATURE_DESIGNER_PROMPT.format(taskInfo)
    # Calls the large language model (LLM) with the generated instruction
    answer = llm.call_llm(function_signature_designer_instruction)
    # Parses the LLM's response into a dictionary
    answer_dict = parse_to_json(answer)
    # Extracts and returns the function signature from the response
    if answer_dict and "function" in answer_dict.keys():
        return answer_dict["function"]
    return ""


# And here is the code for the `get_test_cases` function:

def get_test_cases(llm, taskInfo, function_signature):
    # Generate the test cases based on the function signature and task info
    # This is a placeholder function and should be implemented correctly
    pass

# And here is the code for the `generate_and_extract_code` function:

def generate_and_extract_code(llm, prompt, temperature=None):
    # Generate the code using the large language model
    # This is a placeholder function and should be implemented correctly
    pass

# And here is the code for the `execute_code` function:

def execute_code(code):
    # Execute the generated code
    # This is a placeholder function and should be implemented correctly
    pass