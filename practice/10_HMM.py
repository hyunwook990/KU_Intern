from google.colab import drive
drive.mount("/gdrive", force_remount=True)

# pip install hmmlearn

# 확률 설정
import numpy as np
from hmmlearn import hmm

states = ["Rainy", "Cloudy", "Sunny"]
n_states = len(states)

observations = ["Boots", "Shoes"]
n_observations = len(observations)

# 시작 확률
start_probability = np.array([0.2, 0.5, 0.3])

# 전이 확률
transition_probability = np.array([
    [0.4, 0.3, 0.3],
    [0.2, 0.6, 0.2],
    [0.1, 0.1, 0.8]
])

# 관측 확률
emission_probability = np.array([
    [0.8, 0.2],
    [0.5, 0.5],
    [0.1, 0.9]
])

# 모델 구성 및 디코딩
# 모델 만들기
model = hmm.MultinomialHMM(n_componets=n_states)
model.startprob_ = start_probability
model.transmat_ = transition_probability
model.emissionprob_ = emission_probability

# 관측 입겨
input = [0, 0, 1, 1]

# HMM 모델 호출
hmm_input = np.atleast_2d(input).T
logprob, sequence = model.decode(hmm_input)

print("INPUT:", ",".join(map(lambda x: observations[x], input)))
print("OUTPUT:", ",".join(map(lambda x: states[x], sequence)))
print("PROB:", logprob)