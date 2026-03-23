from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
# Load model directly

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
# model_id = "meta-llama/CodeLlama-13b-hf" # chat_template 지원 안함
# model_id = "Qwen/Qwen2.5-Coder-14B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
    device_map="auto")


SYSTEM_PROMPT = """You are an expert in generating multi-agent systems. You need to generate a proper multi-agent system described in Python code to solve the user query. The Python code should be able to take the user query as input and return the answer as output. The code should be able to run without any errors. Please make sure the code is well-structured and easy to understand.

## Here is an code template for multi-agent system:
```python
from utils import *

class MAS():
    def __init__(self, model_list):
        self.llm = LLM(model_list)

    def forward(self, taskInfo):
        pass
        return
```

## The utils library contains the following elements that you can use in your code:
- `LLM(model_list)`: a class that represents a large language model with the given model list.
- `execute_code(code)`: a function that executes the given code and returns the output.
- `test_code_get_feedback(code, test_cases)`: a function that tests the given code with the test cases and returns the feedback.
- `get_function_signature(llm, taskInfo)`: a function that returns the generated function signature for the given task.
- `get_test_cases(llm, taskInfo, function_signature)`: a function that returns the generated test cases for the given task and function signature.
- `extract_code_solution(solution)`: a function that returns the code by extracting (wrapped within <Code Solution> and </Code Solution>) from the given solution.
- `generate_and_extract_code(llm, prompt, temperature=None)`: a function that returns the generated response and the extracted code from the response.

## Here is an code template for LLM:
```python
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
```
## Here is an code template for execute_code:
```python
def execute_code(code, timeout=30):
    # Execute Python code and capture standard output and `output` variable, execute code in the specified path, and clean up the directory after execution.

    # Args:
    #     code (str): Python code to be executed.
    #     timeout (int): Maximum execution time (in seconds) for the code.

    # Returns:
    #     str: String containing the print output and `output` variable value of the code execution.

    if not code:
        return "Empty code. No output."
    
    # create a temporary directory
    temp_dir = tempfile.mkdtemp()

    # use Manager().dict() to safely share results between processes
    manager = multiprocessing.Manager()
    result_dict = manager.dict()

    # create and start a subprocess
    p = multiprocessing.Process(target=run_code_in_process, args=(code, temp_dir, result_dict))
    p.start()

    # wait for the process to end, or timeout
    p.join(timeout)

    final_result = ""

    # check if the process is still running (i.e. timeout)
    if p.is_alive():
        # force terminate the process!
        p.terminate()
        # wait for termination to complete
        p.join()
        final_result = "Execution Time Out"
    else:
        # process ended normally
        if result_dict.get("error"):
            final_result = f"Get the following error during code execution:\n{result_dict.get('error')}"
        else:
            final_result = f"Final output: {result_dict.get('output', 'None')}\nPrint during execution:\n{result_dict.get('stdout', '')}"

    # clean up the temporary directory
    try:
        shutil.rmtree(temp_dir)
    except Exception as e:
        print(f"[Warning] Error cleaning temp directory: {e}")

    return final_result
```
## Here is an code template for test_code_get_feedback:
```python
def test_code_get_feedback(code, test_cases, timeout=20):
    # Test the given code against a list of test cases in a specified directory with a time limit and provide feedback.

    # Args:
    #     code (str): The Python code to be tested, typically a function definition.
    #     test_cases (list of str): A list of test cases, where each test case is an assert statement represented as a string.
    #     timeout (int): Maximum time (in seconds) allowed for testing all test cases.

    # Returns:
    #     tuple: A tuple containing:
    #         - int: The number of test cases that passed.
    #         - str: A detailed, LLM-friendly feedback string.
    if not code:
        return 0, "Empty code! This might be due to the code not being provided in the correct format (wrapped with triple backticks ```), causing extraction to fail."
    if not test_cases:
        return 0, "No test case provided!"

    # Create a unique temporary directory for this specific run
    temp_dir_path = tempfile.mkdtemp(prefix="test_workspace_")
    
    queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=_execute_tests_in_process, args=(code, test_cases, temp_dir_path, queue))
    
    process.start()
    process.join(timeout)

    feedback = ""
    passed_count = 0

    if process.is_alive():
        process.terminate()
        process.join()
        passed_count = 0
        feedback = "Execution Time Out: The testing process took too long and was terminated."
    else:
        try:
            passed_count, feedback = queue.get_nowait()
        except multiprocessing.queues.Empty:
            passed_count = 0
            feedback = "Execution process finished unexpectedly without providing feedback."

    # Reliably clean up the temporary directory
    try:
        shutil.rmtree(temp_dir_path)
    except Exception as e:
        print(f"[Warning] Error cleaning temp directory: {e}")
    
    return passed_count, feedback

```
## Here is an code template for get_function_signature:
```python
def get_function_signature(llm, taskInfo):

    # Generate a Python function signature based on the problem description.

    # Args:
    #     taskInfo (str): The problem description.

    # Returns:
    #     str: The function signature with an appropriate docstring.

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
```
## Here is an code template for get_test_cases:
```python
def get_test_cases(llm, taskInfo, function_signature):

    # Generate test cases based on the problem description and function signature.

    # Args:
    #     taskInfo (str): The problem description.
    #     function_signature (str): The Python function signature.

    # Returns:
    #     list: A list of test cases combining basic, edge, and large-scale scenarios.

    # Generates an instruction prompt by formatting the TEST_DESIGNER_PROMPT with the task information and function signature
    test_designer_instruction = TEST_DESIGNER_PROMPT.format(problem=taskInfo, function=function_signature)
    # Calls the LLM with the generated instruction
    answer = llm.call_llm(test_designer_instruction, temperature=0.3)
    # Parses the LLM's response into a dictionary
    answer_dict = parse_to_json(answer)
    # Combines and returns the basic, edge, and large-scale test cases from the response
    if answer_dict and "basic" in answer_dict.keys() and "edge" in answer_dict.keys() and "large scale" in answer_dict.keys():
        return answer_dict["basic"] + answer_dict["edge"] + answer_dict["large scale"]
    # return an empty list if parse fails
    return []
```
## Here is an code template for extract_code_solution:
```python
def extract_code_solution(solution):

    # Extract the code solution from the provided solution string.

    # Args:
    #     solution (str): The solution string containing the code snippet.

    # Returns:
    #     str: The extracted code snippet.

    # Extract the code snippet enclosed by custom tags
    code_pattern = r"<Code Solution>\s*(.*?)\s*</Code Solution>"
    match = re.search(code_pattern, solution, re.DOTALL)
    if match:
        code = match.group(1)
        # Remove code block tags if present
        code = re.sub(r"^```(?:\w+)?\n?|```$", "", code, flags=re.MULTILINE).strip()
        if code:
            return code
        return ""
    return ""
```
## Here is an code template for generate_and_extract_code:
```python
def generate_and_extract_code(llm, prompt, temperature=None, max_attempts=3):

        # Generate a response from the LLM and extract the contained code with retry logic.

        # This function attempts to generate a response from the LLM containing a code snippet.
        # It first extracts the portion of the response wrapped within custom tags (e.g., <Code Solution>). 
        # Then remove possible code block tags (e.g., ```python).
        # Returns both the full response and the extracted code. 
        # If no valid code is found after multiple attempts, it returns the last response and an empty string for the code.

        # Args:
        #     prompt (str): The instruction to send to the LLM to generate a response with code.
        #     temperature (float, optional): Sampling temperature for the LLM, controlling randomness in the output.
        #     max_attempts (int): Maximum number of attempts to fetch a response with valid code. Default is 3.
            
        # Returns:
        #     tuple:
        #         str: The full LLM response.
        #         str: The extracted code snippet, or an empty string if no valid code is detected.

        attempts = 0  # Track the number of attempts
        tag_pattern = r"<Code Solution>\s*(.*?)\s*</Code Solution>" # Regular expression pattern to extract content within custom tags
        
        while attempts < max_attempts:
            # Generate response using the LLM
            if temperature:
                llm_response = llm.call_llm(prompt, temperature=temperature)
            else:
                llm_response = llm.call_llm(prompt)
                
            code = extract_code_solution(llm_response)
            if code:
                return llm_response, code
            
            attempts += 1  # Increment attempts and retry if no valid code is detected
        
        # Return the last LLM response and an empty code snippet after exhausting all attempts
        return llm_response, ""
```
If you need to use other functions, you need to implement them by yourself in your returned code."""

Query = """Here is the user query:
Find the degree for the given field extension Q(sqrt(2), sqrt(3), sqrt(18)) over Q.
"""



messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
     {"role": "user", "content": "Generate Python code for a multi-agent system that solves: Find the degree for the field extension Q(sqrt(2), sqrt(3), sqrt(18)) over Q."}
]
inputs = tokenizer.apply_chat_template(
	messages,
	add_generation_prompt=True,
	tokenize=True,
	return_dict=True,
	return_tensors="pt",
).to(model.device)

outputs = model.generate(
    **inputs, # inputs가 dict 형식으로 반환되기 때문에
    max_new_tokens=2000,
    pad_token_id=tokenizer.eos_token_id,  # pad token이 없는 모델이 있을 수 있기때문
    do_sample=False # 가장 확률이 높은 토큰만 사용
    )
generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]
print("generated token count:", len(generated_ids))
print(tokenizer.decode(generated_ids, skip_special_tokens=True))