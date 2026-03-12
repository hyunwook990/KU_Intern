# MAS-GPT practice
- 논문 출처: https://arxiv.org/abs/2503.03686

### 2026.03.09
- 논문에서 나온 내용중 구현에 참고할 내용들을 확인
1. Following this framework, we first re-implement several existing MAS methods (e.g., Multi-Agent Debate (Duet al., 2024), Self-Consistency (Wang et al., 2024b), Self-Refine (Madaan et al., 2024)) to align with our unified code representation.
2. To further expand the diversity of MAS candidates, we also manually design some MAS systems, resulting in a base MAS pool comprising over 40 unique MAS designs
3. Importantly, these 40+ MAS do not directly correspond to the exact number of MAS in the training dataset; rather, they serve as foundations that evolve during the query-MAS pair refinement process.
- github `template.py`에 MAS pyhton code 생성 System prompt로 보이는 것을 발견
- 데이터 생성: `Llama-3-70B-Instruct` 사용
- MAS-GPT backbone: `Qwen2.5-Coder-32B-Instruct` 사용 (to leveraging instruction-following, coding capabilities)
- 단, 자원부족으로 인해 실습은 `meta-llama/Meta-Llama-3.1-8B-Instruct` 사용
출처: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
### 2026.03.10
- MAS-GPT github 주소에 `template.py` 기반으로 간단한 MAS python snippet 생성 파이프라인 구축
### 2026.03.12
- `max_new_tokens`의 크기가 충분하지 않아 출력이 중간에 끊기는 현상 발생 `40 -> 200 -> 2000`으로 변경
- `max_new_tokens = 200`
```python
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
        response, code = generate_and_extract_code(self.ll
```
- `max_new_tokens = 2000` (`generated tokens: 875`)
```python
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

class LLM:
    def __init__(self, model_list):
        self.model_list = model_list

    def generate_code(self, prompt, temperature=None):
        # Generate code using the large language model
        # This is a placeholder function and should be implemented correctly
        pass

    def execute_code(self, code):
        # Execute the generated code
        # This is a placeholder function and should be implemented correctly
        pass

# And here is the code for the `get_function_signature` function:

def get_function_signature(llm, taskInfo):
    # Generate the function signature based on the task info
    # This is a placeholder function and should be implemented correctly
    pass

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

# Note that these functions are placeholders and should be implemented correctly to provide the necessary functionality for the multi-agent system.
```