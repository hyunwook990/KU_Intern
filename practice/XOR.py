import os
import numpy as np
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
from torch.utils.data import (DataLoader, RandomSampler, TensorDataset)

class XOR(nn.Module):
    
    def __init__(self, config):
        super(XOR, self).__init__()
        
        # attribute, filed, member variable
        # 입력 층 노드 수
        self.inode = config["input_node"]
        # 은닉층 데이터 크기
        self.hnode = config["hidden_node"]
        # 출력 층 노드 수: 분류해야 하는 레이블 수
        self.onode = config["output_node"]
        
        # 활성화 함수로 Sigmoid 사용
        self.activation = nn.Sigmoid()
        
        # 신경망 설계
        self.linear1 = nn.Linear(self.inode, self.hnode, bias=True)
        self.linear2 = nn.Linear(self.hnode, self.onode, bias=True)
        
    def forward(self, input_features):
        
        output1 = self.linear1(input_features)
        hypothesis1 = self.activation(output1)
        
        output2 = self.linear2(hypothesis1)
        hypothesis2 = self.activation(output2)
        
        return hypothesis2

# 데이터 읽기 함수
def load_dataset(file):
    data = np.loadtxt(file)
    print(type(data))
    print("DATA", data)
    
    input_features = data[:, 0:-1]
    print("INPUT_FEATURES=", input_features)
    
    labels = np.reshape(data[:,-1], (4,1))
    print("LABELS=", labels)
    
    input_features = torch.tensor(input_features, dtype=torch.float).cuda()
    labels = torch.tensor(labels, dtype=torch.float).cuda()
    
    return (input_features, labels)
    
# 모델 학습 함수
def train(config):
    
    # 모델 생성
    model = XOR(config).cuda()  # CPU 사용 시 .cuda() 제거
    
    # 데이터 읽기
    (input_features, labels) = load_dataset(config["input_data"])
    
    # TensorDataset/DataLoader를 통해 배치(batch) 단위로 데이터를 나누고 셔플(shuffle)
    train_features = TensorDataset(input_features, labels)
    train_dataloader = DataLoader(train_features, shuffle=True, batch_size=config["batch_size"])
    
    # 이진분류 크로스엔트로피 비용 함수
    loss_func = nn.BCELoss()
    # 옵티마이저 함수 (역전파 알고리즘을 수행할 함수)
    optimizer = torch.optim.SGD(model.parameters(), lr=config["learning_rate"])
    
    for epoch in range(config["epoch"] + 1):
        # 학습모드 세팅
        model.train()
        
        # epoch 마다 평균 비용을 저장하기 위한 리스트
        costs = []
        
        for (step, batch) in enumerate(train_dataloader):
            
            # batch = (input_features[step], labels[step])*batch_size
            # .cuda()를 통해 메모리에 업로드
            batch = tuple(t.cuda() for t in batch)
            
            # 각 feature 저장
            input_features, labels = batch
            
            # 역전파 변화도 초기화
            # .backward() 호출 시, 변화도 버퍼에 데이터가 계속 누적한 것을 초기화
            # 초기화 하지 않을 시 다음 작업에 영향이 갈 수 있음.
            optimizer.zero_grad()
            
            # H(x) 계산: forward 연산
            hypothesis = model(input_features)
            # 비용 계산
            cost = loss_func(hypothesis, labels)
            # 역전파 수행
            cost.backward()
            optimizer.step()
            
            # 현재 batch의 스탭 및 loss 저장
            costs.append(cost.data.item())
            
            if epoch%100 == 0:
                print("Average Loss = {0:f}".format(np.mean(costs)))
                torch.save(model.state_dict(), os.path.join(config["output_dir"], "epoch_{0:d}.pt".format(epoch)))
                do_test(model, train_dataloader)

# 모델 평가 함수            
def test(config):
    
    # 모델 생성
    model = XOR(config).cuda()
    
    # 저장된 모델 가중치 로드
    model.load_state_dict(torch.load(os.path.join(config["output_dir"], config["model_name"])))
    
    # 데이터 load
    (input_features, labels) = load_dataset(config["input_data"])
    
    test_features = TensorDataset(input_features, labels)
    test_dataloader = DataLoader(test_features, shuffle=False, batch_size=config["batch_size"])
    
    do_test(model, test_dataloader)
    
# 모델 평가 결과 계산을 위해 텐서를 리스트로 변환하는 함수
def tensor2list(input_tensor):
    return input_tensor.cpu().detach().numpy().tolist()

# 평가 수행 함수
def do_test(model, test_dataloader):
    
    # 평가모드 세팅
    model.eval()
    
    predicts, goals = [], []
    
    with torch.no_grad():
        
        for step, batch in enumerate(test_dataloader):
            
            # .cuda()를 통해 메모리에 업로드
            batch = tuple(t.cuda() for t in batch)
            
            input_features, labels = batch
            hypothesis = model(input_features)
            logits = (hypothesis > .5).float()
            x = tensor2list(logits)
            y = tensor2list(labels)
            
            # 예측값과 정답을 리스트에 추가
            predicts.extend(x)
            goals.extend(y)
    
    print("PRED=", predicts)
    print("GOAL=", goals)
    print("ACCURACY= {0:f}\n", format(accuracy_score(goals, predicts)))
    
    
if(__name__=="__main__"):

    root_dir = "/gdrive/My Drive/colab/ann/xor"
    output_dir = os.path.join(root_dir, "output")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    input_data = "{0:s}/{1:s}".format(root_dir,"train.txt")

    config = {"mode": "test",
              "model_name":"epoch_{0:d}.pt".format(1000),
              "output_dir":output_dir,
              "input_data":input_data,
              "input_node":2,
              "hidden_node":10,
              "output_node":1,
              "learn_rate":1,
              "batch_size":4,
              "epoch":1000,
              }

    if(config["mode"] == "train"):
        train(config)
    else:
        test(config)
        
# import numpy as np
# import torch
# import torch.nn as nn
# from sklearn.metrics import accuracy_score

# # 데이터 읽기 함수
# def load_dataset(file, device):
#   data = np.loadtxt(file)
#   print("DATA=",data)
  
#   input_features = data[:,0:-1]
#   print("INPUT_FEATURES=",input_features)
  
#   labels = np.reshape(data[:,-1],(4,1))
#   print("LABELS=",labels)
 
#   input_features = torch.tensor(input_features, dtype=torch.float).to(device)
#   labels = torch.tensor(labels, dtype=torch.float).to(device)

#   return (input_features, labels)

# # 모델 평가 결과 계산을 위해 텐서를 리스트로 변환하는 함수
# def tensor2list(input_tensor):
#     return input_tensor.cpu().detach().numpy().tolist()

# # GPU 사용 가능 여부 확인
# if torch.cuda.is_available():
#   device = 'cuda'
# else:
#   device = 'cpu'

# input_features, labels = load_dataset("/gdrive/My Drive/colab/ann/xor/train.txt",device)

# # NN 모델 만들기
# model = nn.Sequential(
#           nn.Linear(2, 10, bias=True), nn.ReLU(), nn.Dropout(0.1),
#           nn.Linear(10, 10, bias=True), nn.ReLU(), nn.Dropout(0.1),
#           nn.Linear(10, 10, bias=True), nn.ReLU(), nn.Dropout(0.1),
#           nn.Linear(10, 10, bias=True), nn.ReLU(), nn.Dropout(0.1),
#           nn.Linear(10, 10, bias=True), nn.ReLU(), nn.Dropout(0.1),
#           nn.Linear(10, 10, bias=True), nn.ReLU(), nn.Dropout(0.1),
#           nn.Linear(10, 10, bias=True), nn.ReLU(), nn.Dropout(0.1),
#           nn.Linear(10, 1, bias=True), nn.Sigmoid()).to(device)

# # 이진분류 크로스엔트로피 비용 함수 
# loss_func = torch.nn.BCELoss().to(device)
# # 옵티마이저 함수 (역전파 알고리즘을 수행할 함수)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.2)

# # 학습 모드 셋팅
# model.train()

# # 모델 학습
# for epoch in range(3001):

#     # 기울기 계산한 것들 초기화
#     optimizer.zero_grad()

#     # H(X) 계산: forward 연산
#     hypothesis = model(input_features)

#     # 비용 계산
#     cost = loss_func(hypothesis, labels)
#     # 역전파 수행
#     cost.backward()
#     optimizer.step()

#     # 1000 에폭마다 비용 출력
#     if epoch % 300 == 0:
#         print(epoch, cost.item())

# # 평가 모드 셋팅 (학습 시에 적용했던 드랍 아웃 여부 등을 비적용)
# model.eval()

# # 역전파를 적용하지 않도록 context manager 설정
# with torch.no_grad():
#     hypothesis = model(input_features)
#     logits = (hypothesis > 0.5).float()
#     predicts = tensor2list(logits)
#     golds = tensor2list(labels)
#     print("PRED=",predicts)
#     print("GOLD=",golds)
#     print("Accuracy : {0:f}".format(accuracy_score(golds, predicts)))