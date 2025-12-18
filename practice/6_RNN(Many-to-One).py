import os
import numpy as np
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
from torch.utils.data import (DataLoader, RandomSampler, TensorDataset)
import csv
from sklearn.preprocessing import MinMaxScaler

# 모델 Architecture 설계
class STOCK_RNN(nn.Module):
    
    def __init__(self, config):
        super(STOCK_RNN, self).__init__()
        
        self.input_size = config["input_size"]
        self.hidden_size = config["hidden_size"]
        self.output_size = config["output_size"]
        self.num_layers = config["num_layers"]
        self.batch_size = config["batch_size"]
        
        # LSTM 설계
        self.lstm = nn.LSTM(self.input_size,
                            self.hidden_size,
                            self.num_layers,
                            bidirectional=False,
                            batch_first=True)
        # 출력층 설계
        self.linear = nn.Linear(self.hidden_size, self.output_size)
        
    def forward(self, input_features):
        
        # LSTM 리턴 = output, (배치, 시퀀스, 은닉 상태), (hidden_state, cell_state)
        x, (h_n, c_n) = self.lstm(input_features)
        
        # ouput에서 마지막 시퀀스의 (배치, 은닉 상태) 정보를 가져옴
        h_t = x[:, -1, :]
        
        # 출력층: (배치, 출력)
        hypothesis = self.linear(h_t)
                
        return hypothesis

# 데이터셋 읽기 함수
def load_dataset(fname):
    f = open(fname, 'r', encoding='cp949')
    
    # CSV 파일 읽기
    data = csv.reader(f, delimiter='.')
    
    # 헤더 건너뛰기
    next(data)
    
    data_X = []
    data_Y = []
    
    for row in data:
        # 오픈, 고가, 저가, 거래량 -> 숫자 변환
        data_X.append([float(i) for i in row[2:]])
        # 종가 -> 숫자 변환
        data_Y.append(float(row[1]))
    
    # MinMax 정규화 (예측하려는 종가 제외)
    scaler = MinMaxScaler()
    scaler.fit(data_X)
    data_X = scaler.transform(data_X)
    data_num = len(data_X)
    sequence_len = config["sequence_len"]
    seq_data_X, seq_data_Y = [], []
    
    # 윈도우 크기만큼 슬라이딩 하면서 데이터 생성
    for i in range(data_num - sequence_len):
        window_size = i + sequence_len
        seq_data_X.append(data_X[i:window_size])
        seq_data_Y.append(data_Y[window_size - 1])
    
    (train_X, train_Y) = (np.array(seq_data_X[:]), np.array(seq_data_Y[:]))
    train_X = torch.tensor(train_X, dtype=torch.float)
    train_Y = torch.tensor(train_Y, dtype=torch.float)
    
    print(train_X.shape) # (73, 3, 4)
    print(train_Y.shape) # (73, 1)
    
    return (train_X, train_Y)
    
# 모델 학습 함수
def train(config):
    
    # 모델 생성
    model = STOCK_RNN(config).cuda()  # CPU 사용 시 .cuda() 제거
    
    # 데이터 읽기
    (input_features, labels) = load_dataset(config["input_data"])
    
    # TensorDataset/DataLoader를 통해 배치(batch) 단위로 데이터를 나누고 셔플(shuffle)
    train_features = TensorDataset(input_features, labels)
    train_dataloader = DataLoader(train_features, shuffle=True, batch_size=config["batch_size"])
    
    # 이진분류 크로스엔트로피 비용 함수
    loss_func = nn.MSELoss()
    # 옵티마이저 함수 (역전파 알고리즘을 수행할 함수)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    
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
    model = STOCK_RNN(config).cuda()
    
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
            # logits = (hypothesis > .5).float() -> 회귀 모델 -> argmax 필요 없음
            x = tensor2list(hypothesis[:, 0])
            y = tensor2list(labels)
            
            # 예측값과 정답을 리스트에 추가
            predicts.extend(x)
            goals.extend(y)
        # 소숫점 이하 1자리로 변환
        predicts = [round(i, 1) for i in predicts]
        goals = [round(i[0], 1) for i in goals]
    
    print("PRED=", predicts)
    print("GOAL=", goals)
    print("ACCURACY= {0:f}\n", format(accuracy_score(goals, predicts)))
    

if(__name__=="__main__"):

    root_dir = "/gdrive/My Drive/colab/rnn/stock"
    output_dir = os.path.join(root_dir, "output")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    config = {"mode": "train",
              "model_name":"epoch_{0:d}.pt".format(10),
              "output_dir":output_dir,
              "file_name": "{0:s}/samsung-2020.csv".format(root_dir),
              "sequence_len": 3,
              "input_size": 4,
              "hidden_size": 10,
              "output_size": 1,
              "num_layers": 1,
              "batch_size": 1,
              "learn_rate": 0.1,
              "epoch": 10,
              }

    if(config["mode"] == "train"):
        train(config)
    else:
        test(config)