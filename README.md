# MAS-GPT practice
- 출처: ttps://arxiv.org/abs/2503.03686

### 2026.03.09
- 논문에서 나온 내용중 구현에 참고할 내용들을 확인
1. Following this framework, we first re-implement several existing MAS methods (e.g., Multi-Agent Debate (Duet al., 2024), Self-Consistency (Wang et al., 2024b), Self-Refine (Madaan et al., 2024)) to align with our unified code representation.
2. To further expand the diversity of MAS candidates, we also manually design some MAS systems, resulting in a base MAS pool comprising over 40 unique MAS designs
3. Importantly, these 40+ MAS do not directly correspond to the exact number of MAS in the training dataset; rather, they serve as foundations that evolve during the query-MAS pair refinement process.
- github `template.py`에 MAS pyhton code 생성 System prompt로 보이는 것을 발견
- 데이터 생성: Llama-3-70B-Instruct 사용
- MAS-GPT backbone: Qwen2.5-Coder-32B-Instruct 사용 (to leveraging instruction-following, coding capabilities)
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