import itertools
from google.colab import userdata
from utils import *
api_key = userdata.get('OPENAI')

class MAS():
    def __init__(self, model_list):
        self.llm = LLM(model_list)

    def forward(self, query):
        math_agent = f'You are a math expert. Solve this question step by step: {query}.'
        math_output = self.llm.call_llm(math_agent)
        print(math_output)

        feedback_agent = f'Given {query} and {math_output}, provide feedback'
        feedback_output = self.llm.call_llm(feedback_agent)
        print(feedback_output)

        refine_agent = f'Given {query}, {math_output} and {feedback_output}, provide the correct final answer, Give only the final answer'
        refine_output = self.llm.call_llm(refine_agent)
        return refine_output

# Test the MAS
mas = MAS([ ("gpt-4o-mini", "https://api.openai.com/v1", api_key)
,  ("gpt-4o-mini", "https://api.openai.com/v1", api_key)
,  ("gpt-4o-mini", "https://api.openai.com/v1", api_key)
])
query = "Let p = (1, 2, 5, 4)(2, 3) in S_5 . Find the index of <p> in S_5."
print(mas.forward(query))