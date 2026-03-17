# MAS-GPT practice
- 논문 출처: https://arxiv.org/abs/2503.03686
---
# 2026.03.09
- 논문에서 나온 내용중 구현에 참고할 내용들을 확인
1. Following this framework, we first re-implement several existing MAS methods (e.g., Multi-Agent Debate (Duet al., 2024), Self-Consistency (Wang et al., 2024b), Self-Refine (Madaan et al., 2024)) to align with our unified code representation.
2. To further expand the diversity of MAS candidates, we also manually design some MAS systems, resulting in a base MAS pool comprising over 40 unique MAS designs
3. Importantly, these 40+ MAS do not directly correspond to the exact number of MAS in the training dataset; rather, they serve as foundations that evolve during the query-MAS pair refinement process.
- github `template.py`에 MAS pyhton code 생성 System prompt로 보이는 것을 발견
- 데이터 생성: `Llama-3-70B-Instruct` 사용
- MAS-GPT backbone: `Qwen2.5-Coder-32B-Instruct` 사용 (to leveraging instruction-following, coding capabilities)
- 단, 자원부족으로 인해 실습은 `meta-llama/Meta-Llama-3.1-8B-Instruct` 사용
출처: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
---
# 2026.03.10
- MAS-GPT github 주소에 `template.py` 기반으로 간단한 MAS python snippet 생성 파이프라인 구축
---
# 2026.03.12
- `max_new_tokens`의 크기가 충분하지 않아 출력이 중간에 끊기는 현상 발생
`40 -> 200 -> 2000`으로 변경

<detils>
<summary> `max_new_tokens = 200` </summary>
<div markdown="1">

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

</div>
</details>

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
### 다음에 할 일
- 다른 모델 사용해보기 (코딩 특화): `Qwen/Qwen2.5-Coder-14B-Instruct`
## 2026.03.13
- `Qwen/Qwen2.5-Coder-14B-Instruct`모델로 코드 실행 시 아래와 같은 오류 발생
```python
# Google Colab T4로 런타임 유형 설정, 14B 모델이 VRAM에 전부 올라가지 않아서 CPU에 분산하여 로드한다는 경고 메세지
WARNING:accelerate.big_modeling:Some parameters are on the meta device because they were offloaded to the disk and cpu.
# do_sample = False라서 아래의 설정들을 사용하지 못한다는 경고 메세지
The following generation flags are not valid and may be ignored: ['temperature', 'top_p', 'top_k']. Set 'TRANSFORMERS_VERBOSITY=info' for more details.
```
---
## 2026.03.15
- Colab에서 `Qwen/Qwen2.5-Coder-14B-Instruct`모델로 코드 실행 시 5시간동안 출력이 나오지 않아 8B 모델로 진행할 예정.
- 8B 모델이 출력한 MAS 코드의 함수와 utils의 함수의 형태가 다른 것을 확인, 필요한 함수들만 `system prompt`로 제공하여 출력을 확인할 예정.
- OOM(Out Of Memory) 오류 발생 -> 프롬프트를 너무 길게 제공해서 발생한 듯 함.
---
## 2026.03.16
- RAG를 활용하여 utils의 함수들을 제공하는 방법을 사용할 예정.
### 오류 확인
- `test_MAS.py`파일 실행 후, `output = Empty code. No output.` 출력 확인.
- `execute_code` 함수에 code가 제공되지 않은 것으로 확인.
- `response`에 답변이 나오고 `code=""`로 return 되었기에 이런 문제가 발생함. -> `extract_code_solution` 함수에서 제대로 처리가 안되는 듯함.
## 2026.03.17
### 오류 확인
- `generate_and_extract_code`의 `llm_response`가 아래와 같이 출력되며, `extract_code_solution`에서 코드 추출을 제대로 수행하지 못하는 듯함.
 To find the degree of the field extension \( \mathbb{Q}(\sqrt{2}, \sqrt{3}, \sqrt{18}) \) over \( \mathbb{Q} \), we need to determine the minimal polynomials of the elements involved and how they contribute to the degree of the extension.

1. **Identify the elements**: 
   - We have \( \sqrt{2} \), \( \sqrt{3} \), and \( \sqrt{18} \). Notably, \( \sqrt{18} = \sqrt{9 \cdot 2} = 3\sqrt{2} \).

2. **Field extension construction**: 
   - Start with \( \mathbb{Q} \).
   - First, extend to \( \mathbb{Q}(\sqrt{2}) \).
   - Then extend to \( \mathbb{Q}(\sqrt{2}, \sqrt{3}) \).
   - Finally, check if \( \sqrt{18} \) introduces a new element or if it's already in the field.

3. **Calculate the degree of each extension**:
   - The degree of \( \mathbb{Q}(\sqrt{2}) \) over \( \mathbb{Q} \) is 2, since the minimal polynomial of \( \sqrt{2} \) over \( \mathbb{Q} \) is \( x^2 - 2 \).
   - Next, \( \mathbb{Q}(\sqrt{2}, \sqrt{3}) \) is obtained by adjoining \( \sqrt{3} \) to \( \mathbb{Q}(\sqrt{2}) \). The minimal polynomial of \( \sqrt{3} \) over \( \mathbb{Q}(\sqrt{2}) \) is \( x^2 - 3 \), which is irreducible in \( \mathbb{Q}(\sqrt{2}) \), giving us a degree of 2 for this extension as well.
   - Thus, the degree of \( \mathbb{Q}(\sqrt{2}, \sqrt{3}) \) over \( \mathbb{Q} \) is \( 2 \times 2 = 4 \).

4. **Check the contribution of \( \sqrt{18} \)**:
   - Since \( \sqrt{18} = 3\sqrt{2} \) is already in \( \mathbb{Q}(\sqrt{2}) \), it does not increase the degree of the field extension.

5. **Final degree calculation**:
   - Therefore, the degree of the extension \( \mathbb{Q}(\sqrt{2}, \sqrt{3}, \sqrt{18}) \) over \( \mathbb{Q} \) is 4.

Now, we can implement this logic in a Python function:

```python
def field_extension_degree():
    # The degree of Q(sqrt(2), sqrt(3), sqrt(18)) over Q
    degree_sqrt2 = 2  # [Q(sqrt(2)):Q]
    degree_sqrt3 = 2  # [Q(sqrt(2), sqrt(3)):Q(sqrt(2))]
    degree_sqrt18 = 1 # [Q(sqrt(2), sqrt(3), sqrt(18)):Q(sqrt(2), sqrt(3))]

    total_degree = degree_sqrt2 * degree_sqrt3 * degree_sqrt18
    return total_degree

# Call the function and print the result
print(field_extension_degree())
```

When you run this function, it will return the degree of the field extension \( \mathbb{Q}(\sqrt{2}, \sqrt{3}, \sqrt{18}) \) over \( \mathbb{Q} \), which is 4.