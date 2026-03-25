from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
# Load model directly

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
# model_id = "meta-llama/CodeLlama-13b-hf" # chat_template 지원 안함
# model_id = "Qwen/Qwen2.5-Coder-14B-Instruct" # Colab 추론시간이 너무 길다.

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
    device_map="auto")

def form_instruction(query):
    instruction = "I will give a user query and a code template for multi-agent system. You need to generate a proper multi-agent system described in Python code to solve the query. The Python code should be able to take the query as input and return the answer as output. The code should be able to run without any errors. Please make sure the code is well-structured and easy to understand.\n\n"
    instruction += f"Here is the user query: {query}\n\n"
    instruction += "Here is the code template:\n\n"
    instruction += '''```python
from utils import *

class MAS():
    def __init__(self, model_list):
        self.llm = LLM(model_list)

    def forward(query):
      math_agent = f'You are a math expert. Solve this question: {query}
      math_output = call_llm(math_agent)

      feedback_agent = f'Given {query} and {math_output}, provide feedback
      feedback_output = call_llm(feedback_agent)

      refine_agent = f'Given {query}, {math_output} and {feedback_output}, provide the final answer

      return call_llm(refine_agent)
```'''
#     instruction += '''```python
# from utils import *

# class MAS():
#     def __init__(self, model_list):
#         self.llm = LLM(model_list)

#     def forward(self, taskInfo):
#         pass
#         return
# ```'''

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

If you need to use other functions, you need to implement them by yourself in your returned code."""

Query = str(input())

instruction = form_instruction(Query)



messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
     {"role": "user", "content": instruction}
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