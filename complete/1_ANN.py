import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score

def load_dataset(file, device):
    data = np.loadtxt(file)
    print(type(data))
    print("DATA", data)
    
    input_features = data[:, 0:-1]
    print("INPUT_FEATURES=", input_features)
    
    labels = np.reshape(data[:,-1], (4,1))
    print("LABELS=", labels)
    
    input_features = torch.tensor(input_features, dtype=torch.float).to(device)
    labels = torch.tensor(labels, dtype=torch.float).to(device)
    
    return (input_features, labels)

def tensor2list(input_tensor):
    return input_tensor.cpu().detach().numpy().tolist()

# GPU 사용 가능 여부 확인
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
    
input_features, labels = load_dataset("")

model = nn.Sequential(
    nn.Linear(2, 2, bias=True),
    nn.Sigmoid(),
    nn.Linear(2, 1, bias=True),
    nn.Sigmoid()
).to(device)

# 노드가 더 많은 버젼 (학습 속도는 느려지지만 더 빠르게 수렴한다.)
wide_ann = nn.Sequential(
    nn.Linear(in_features=2, out_features=10, bias=True),
    nn.Sigmoid(),
    nn.Linear(in_features=10, out_features=1, bias=True)
)

# Single-layer Perceptron -> 학습 속도는 빠르지만 학습을 아무리 많이 해도 문제를 제대로 풀지 못함
single_ann = nn.Sequential(
    nn.Linear(2, 1, bias=True),
    nn.Sigmoid()
).to(device)

# Hidden layer의 층을 하나 더 쌓는 것은 선을 구부리는 효과를 가진다.
deep_ann = nn.Sequential(
    nn.Linear(2, 2, bias=True),
    nn.Sigmoid(),
    nn.Linear(2, 2, bias=True),
    nn.Sigmoid(),
    nn.Linear(2, 1, bias=True),
    nn.Sigmoid()
).to(device)

# 너무 깊게 쌓으니 오히려 문제를 잘 풀지 못함 -> 기울기 소실 문제가 발생
deeeper_ann = nn.Sequential(
    nn.Linear(2, 10, bias=True),
    nn.Sigmoid(),
    nn.Linear(10, 10, bias=True),
    nn.Sigmoid(),
    nn.Linear(10, 10, bias=True),
    nn.Sigmoid(),
    nn.Linear(10, 10, bias=True),
    nn.Sigmoid(),
    nn.Linear(10, 10, bias=True),
    nn.Sigmoid(),
    nn.Linear(10, 10, bias=True),
    nn.Sigmoid(),
    nn.Linear(10, 1, bias=True),
    nn.Sigmoid()
).to(device)

# 활성화 함수를 Sigmoid -> ReLU로 바꾸면서 기울기 소실 문제를 해결
deeeper_ann = nn.Sequential(
    nn.Linear(2, 10, bias=True),
    nn.ReLU(),
    nn.Linear(10, 10, bias=True),
    nn.ReLU(),
    nn.Linear(10, 10, bias=True),
    nn.ReLU(),
    nn.Linear(10, 10, bias=True),
    nn.ReLU(),
    nn.Linear(10, 10, bias=True),
    nn.ReLU(),
    nn.Linear(10, 10, bias=True),
    nn.ReLU(),
    nn.Linear(10, 1, bias=True),
    nn.ReLU()
).to(device)

# Regularization(정규화, 일반화) 기법 중 하나인 Drop out을 사용
dropout_ann = nn.Sequential(
    nn.Linear(2, 10, bias=True),
    nn.ReLU(),
    nn.Dropout(.5),
    nn.Linear(10, 10, bias=True),
    nn.ReLU(),
    nn.Dropout(.5),
    nn.Linear(10, 10, bias=True),
    nn.ReLU(),
    nn.Dropout(.5),
    nn.Linear(10, 10, bias=True),
    nn.ReLU(),
    nn.Dropout(.5),
    nn.Linear(10, 10, bias=True),
    nn.ReLU(),
    nn.Dropout(.5),
    nn.Linear(10, 10, bias=True),
    nn.ReLU(),
    nn.Dropout(.5),
    nn.Linear(10, 1, bias=True),
    nn.ReLU()
).to(device)

# 이진분류 크로스엔트로피 비용 함수 (Binary Cross Entropy Cost Function)
loss_func = torch.nn.BCELoss().to(device)
# 옵티마이저 함수 (역전파 알고리즘을 수행할 함수)
optimizer = torch.optim.SGD(model.parameters(), lr=1)

# 학습모드
model.train()

# 모델 학습
for epoch in range(1001):
    # 기울기 계산한 것을 초기화
    optimizer.zero_grad()
    
    # H(x) 계산: forward 연산
    hypothesis = model(input_features)
    
    # 비용 계산
    cost = loss_func(hypothesis, labels)

    # 역전파 수행
    cost.backward()
    optimizer.step()
    
    # 100 에폭마다 비용(Loss) 출력
    if epoch%100 == 0:
        print(epoch, cost.item())
        
# 평가 모드 세팅 (학습 시에 적용했던 드랍 아웃 여부 등을 비적용)
model.eval()

# 역전파 계산하지 않도록 context manager 설정
with torch.no_grad():
    hypothesis = model(input_features)
    # logits = argmax한 값
    logits = (hypothesis > .5).float()
    predicts = tensor2list(logits)
    goals = tensor2list(labels)
    print("PRED=", predicts)
    print("GOLD=", goals)
    print("Accuracy : {0:f}", format(accuracy_score(goals, predicts)))