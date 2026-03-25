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

<details>
<summary> max_new_tokens = 200 출력 </summary>
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

<details>
<summary> max_new_tokens = 2000 (generated tokens: 875) 출력 </summary>
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
</div>
</details>

### 다음에 할 일
- 다른 모델 사용해보기 (코딩 특화): `Qwen/Qwen2.5-Coder-14B-Instruct`
# 2026.03.13
- `Qwen/Qwen2.5-Coder-14B-Instruct`모델로 코드 실행 시 아래와 같은 오류 발생
```python
# Google Colab T4로 런타임 유형 설정, 14B 모델이 VRAM에 전부 올라가지 않아서 CPU에 분산하여 로드한다는 경고 메세지
WARNING:accelerate.big_modeling:Some parameters are on the meta device because they were offloaded to the disk and cpu.
# do_sample = False라서 아래의 설정들을 사용하지 못한다는 경고 메세지
The following generation flags are not valid and may be ignored: ['temperature', 'top_p', 'top_k']. Set 'TRANSFORMERS_VERBOSITY=info' for more details.
```
---
# 2026.03.15
- Colab에서 `Qwen/Qwen2.5-Coder-14B-Instruct`모델로 코드 실행 시 5시간동안 출력이 나오지 않아 8B 모델로 진행할 예정.
- 8B 모델이 출력한 MAS 코드의 함수와 utils의 함수의 형태가 다른 것을 확인, 필요한 함수들만 `system prompt`로 제공하여 출력을 확인할 예정.
- OOM(Out Of Memory) 오류 발생 -> 프롬프트를 너무 길게 제공해서 발생한 듯 함.
---
# 2026.03.16
- RAG를 활용하여 utils의 함수들을 제공하는 방법을 사용할 예정.
### 오류 확인
- `test_MAS.py`파일 실행 후, `output = Empty code. No output.` 출력 확인.
- `execute_code` 함수에 code가 제공되지 않은 것으로 확인.
- `response`에 답변이 나오고 `code=""`로 return 되었기에 이런 문제가 발생함. -> `extract_code_solution` 함수에서 제대로 처리가 안되는 듯함.
# 2026.03.17
### 오류 확인
- `generate_and_extract_code`의 `llm_response`가 아래와 같이 출력되며, `extract_code_solution`에서 코드 추출을 제대로 수행하지 못하는 듯함.

<details>
<summary> llm_response 출력 </summary>
<div>
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

</div>
</details>

### 문제점
- 
1. 현재 논문에서 사용한 모델보다 작은 모델을 사용하여 MAS snippet을 생성했기 때문에 Multi Agent System python code를 제대로 구현하지 못하는 것 같다.
- 입력을 사용자 쿼리로 받고 출력을 정답으로 하는 코드를 제공하라고 prompt에 작성되어있는데 제대로 수행하지 못하는 것 같다.
2. `utils`의 함수들을 모두 파악하지 못해서 LLM이 제대로 된 코드를 제공하지 못하는 것 같다.
- 문제가 단순해서인지 몰라서 사용하지 못하는 것인지 모르겠지만 다양한 함수를 사용하지 않는 것 같다.

# 2026.03.23
- 논문 github에 있는 `template.py`의 형식을 똑같이 따라하기위해 instruction 수정 후, MAS python 코드 재출력 시도
<details>
<summary>MAS 코드 출력</summary>
<div>
generated token count: 653
It seems like you haven't provided any specific query or problem to solve using a multi-agent system. However, I can provide a basic example of a multi-agent system in Python that can be used to solve a variety of problems.

Let's consider a simple problem where we have multiple agents that need to work together to find the shortest path in a grid world. We can use a multi-agent system to model this problem.

```python
from utils import *
import random

class Agent():
    def __init__(self, x, y):
        self.x = x
        self.y = y

class GridWorld():
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.agents = []

    def add_agent(self, agent):
        self.agents.append(agent)

    def get_neighbors(self, x, y):
        neighbors = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.width and 0 <= ny < self.height:
                neighbors.append((nx, ny))
        return neighbors

    def find_shortest_path(self, start, end):
        queue = [(start, [start])]
        visited = set()
        while queue:
            (x, y), path = queue.pop(0)
            if (x, y) == end:
                return path
            for nx, ny in self.get_neighbors(x, y):
                if (nx, ny) not in visited:
                    visited.add((nx, ny))
                    queue.append(((nx, ny), path + [(nx, ny)]))
        return None

class MAS():
    def __init__(self, grid_world):
        self.grid_world = grid_world

    def forward(self, taskInfo):
        # Find the shortest path for each agent
        for agent in self.grid_world.agents:
            start = (agent.x, agent.y)
            end = (random.randint(0, self.grid_world.width - 1), random.randint(0, self.grid_world.height - 1))
            path = self.grid_world.find_shortest_path(start, end)
            print(f"Agent at ({agent.x}, {agent.y}) found path: {path}")

# Create a grid world
grid_world = GridWorld(10, 10)

# Add agents to the grid world
for i in range(5):
    agent = Agent(random.randint(0, 9), random.randint(0, 9))
    grid_world.add_agent(agent)

# Create a multi-agent system
mas = MAS(grid_world)

# Run the multi-agent system
mas.forward("Find the shortest path for each agent")
```

This code defines a simple multi-agent system where each agent is a point in a grid world. The `MAS` class represents the multi-agent system, and it uses the `GridWorld` class to find the shortest path for each agent. The `Agent` class represents an individual agent.

You can modify this code to fit your specific problem and requirements.
</div>
</details>

# 2026.03.25
- MAS 코드가 원하는 수준으로 나오지 않는 이유를 찾기 위해 다음 두가지를 수행할 예정이다.
1. ChatGPT에 프롬프트를 제공, 출력되는 코드를 확인
<details>
<summary>출력</summary>
<div>

```python
from utils import *

class PlannerAgent:
    def __init__(self, llm):
        self.llm = llm

    def make_function_signature(self, taskInfo):
        return get_function_signature(self.llm, taskInfo)


class CoderAgent:
    def __init__(self, llm):
        self.llm = llm

    def make_code(self, taskInfo, function_signature):
        prompt = (
            "You are a careful Python mathematician.\n\n"
            "Solve the following user task by writing correct Python code.\n\n"
            "User task:\n"
            f"{taskInfo}\n\n"
            "Required function signature:\n"
            f"{function_signature}\n\n"
            "Requirements:\n"
            "1. Return only one complete Python solution wrapped in "
            "<Code Solution> and </Code Solution>.\n"
            "2. The code must be executable without errors.\n"
            "3. The code should define the required function.\n"
            "4. The code should compute the answer for the math problem exactly.\n"
            "5. Include a small runnable snippet at the bottom that prints the final answer.\n"
        )
        response, code = generate_and_extract_code(self.llm, prompt, temperature=0.2)
        return response, code


class TesterAgent:
    def __init__(self, llm):
        self.llm = llm

    def make_test_cases(self, taskInfo, function_signature):
        return get_test_cases(self.llm, taskInfo, function_signature)

    def test_code(self, code, test_cases):
        return test_code_get_feedback(code, test_cases)


class DebugAgent:
    def __init__(self, llm):
        self.llm = llm

    def fix_code(self, taskInfo, function_signature, old_code, feedback):
        prompt = (
            "You are fixing Python code.\n\n"
            "User task:\n"
            f"{taskInfo}\n\n"
            "Required function signature:\n"
            f"{function_signature}\n\n"
            "Previous code:\n"
            f"{old_code}\n\n"
            "Test feedback:\n"
            f"{feedback}\n\n"
            "Please repair the code so that:\n"
            "1. It satisfies the task.\n"
            "2. It matches the required function signature.\n"
            "3. It executes without errors.\n"
            "4. It prints the final answer for the given task.\n\n"
            "Return only the corrected code wrapped in "
            "<Code Solution> and </Code Solution>.\n"
        )
        response, code = generate_and_extract_code(self.llm, prompt, temperature=0.1)
        return response, code


class SolverAgent:
    def run_code(self, code):
        return execute_code(code)


class MAS:
    def __init__(self, model_list):
        self.llm = LLM(model_list)
        self.planner = PlannerAgent(self.llm)
        self.coder = CoderAgent(self.llm)
        self.tester = TesterAgent(self.llm)
        self.debugger = DebugAgent(self.llm)
        self.solver = SolverAgent()

    def forward(self, taskInfo):
        function_signature = self.planner.make_function_signature(taskInfo)
        test_cases = self.tester.make_test_cases(taskInfo, function_signature)
        _, code = self.coder.make_code(taskInfo, function_signature)
        feedback = self.tester.test_code(code, test_cases)

        max_retry = 2
        retry = 0

        while retry < max_retry:
            feedback_str = str(feedback).lower()

            passed = (
                ("pass" in feedback_str or "all tests passed" in feedback_str)
                and ("fail" not in feedback_str)
                and ("error" not in feedback_str)
            )

            if passed:
                break

            _, code = self.debugger.fix_code(
                taskInfo=taskInfo,
                function_signature=function_signature,
                old_code=code,
                feedback=feedback,
            )
            feedback = self.tester.test_code(code, test_cases)
            retry += 1

        final_output = self.solver.run_code(code)
        return final_output
```
</div>
</details>

2. `instruction`에 논문에 나온 코드를 제공, one-shot 효과를 기대

<details>
<summary>출력</summary>
<div>
generated token count: 653
It seems like you haven't provided any specific query or problem to solve using a multi-agent system. However, I can provide a basic example of a multi-agent system in Python that can be used to solve a variety of problems.

Let's consider a simple problem where we have multiple agents that need to work together to find the shortest path in a grid world. We can use a multi-agent system to model this problem.

```python
from utils import *
import random

class Agent():
    def __init__(self, x, y):
        self.x = x
        self.y = y

class GridWorld():
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.agents = []

    def add_agent(self, agent):
        self.agents.append(agent)

    def get_neighbors(self, x, y):
        neighbors = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.width and 0 <= ny < self.height:
                neighbors.append((nx, ny))
        return neighbors

    def find_shortest_path(self, start, end):
        queue = [(start, [start])]
        visited = set()
        while queue:
            (x, y), path = queue.pop(0)
            if (x, y) == end:
                return path
            for nx, ny in self.get_neighbors(x, y):
                if (nx, ny) not in visited:
                    visited.add((nx, ny))
                    queue.append(((nx, ny), path + [(nx, ny)]))
        return None

class MAS():
    def __init__(self, grid_world):
        self.grid_world = grid_world

    def forward(self, taskInfo):
        # Find the shortest path for each agent
        for agent in self.grid_world.agents:
            start = (agent.x, agent.y)
            end = (random.randint(0, self.grid_world.width - 1), random.randint(0, self.grid_world.height - 1))
            path = self.grid_world.find_shortest_path(start, end)
            print(f"Agent at ({agent.x}, {agent.y}) found path: {path}")

# Create a grid world
grid_world = GridWorld(10, 10)

# Add agents to the grid world
for i in range(5):
    agent = Agent(random.randint(0, 9), random.randint(0, 9))
    grid_world.add_agent(agent)

# Create a multi-agent system
mas = MAS(grid_world)

# Run the multi-agent system
mas.forward("Find the shortest path for each agent")
```

This code defines a simple multi-agent system where each agent is a point in a grid world. The `MAS` class represents the multi-agent system, and it uses the `GridWorld` class to find the shortest path for each agent. The `Agent` class represents an individual agent.

You can modify this code to fit your specific problem and requirements.
</div>
</details>

## 문제
- 1번의 경우 제공한 프롬프트에 작성된 `utils`의 `generate_and_extract_code`와 `execute_code`함수를 보고 코딩 문제로 판단하여 코드 에이전트를 제공하게 되면서 오류가 발생


<details>
<summary>이후 수정하여 받은 MAS python snippet</summary>
<div>

```python

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
```

</div>
</details>

- 현재까지의 출력에서 코딩 에이전트가 들어가게되면 출력이 잘 나오지 않았음, 코딩문제 MAS를 생성하려면 주의가 필요함