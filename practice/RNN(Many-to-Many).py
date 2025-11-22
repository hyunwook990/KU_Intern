import os
import numpy as np
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
from torch.utils.data import (DataLoader, RandomSampler, TensorDataset)
import csv

# 모델 Architecture 설계
class SpacingRNN(nn.Module):
    
    def __init__(self, config):
        super(SpacingRNN, self).__init__()
        
        # 전체 음절 개수
        self.eumjeol_vocab_size = config["eumjeol_vocab_size"]
        # 음절 임베딩 사이즈
        self.embedding_size = config["embedding_size"]
        # RNN 히든 사이즈
        self.hidden_size = config["hidden_size"]
        # 분류할 라벨의 개수 -> 3 (B, I, PAD)
        self.number_of_labels = config["number_of_labels"]
        # 임베딩층: 랜덤 초기화 후 fine-tuning
        self.embedding = nn.Embedding(num_embeddings=self.eumjeol_vocab_size,
                                      embedding_dim=self.embedding_size,
                                      padding_idx=0)
        self.dropout = nn.Dropout(config['dropout'])
        
        # RNN layer
        # self.bi_gru = nn.GRU(input_size=self.embedding_size,
        #                      hidden_size=self.hidden_size,
        #                      num_layers=1,
        #                      batch_first=True,
        #                      bidirectional=True)
        self.bi_lstm = nn.LSTM(input_size=self.embedding_size,
                            hidden_size=self.hidden_size,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=True)
        
        # fully_connected layer를 통하여 출력 크기를 number_of_labels에 맞춰줌
        # (batch_size, max_lengh, hidden_size*2) -> (batch_size, max_length, number_of_labels)
        self.linear = nn.Linear(in_features=self.hidden_size*2, # bidirectional이므로 2배
                                out_features=self.number_of_labels)
        
    def forward(self, inputs):
        
        # (batch_size, max_length) -> (batch_size, max_length, embedding_size)
        emjeoul_inputs = self.embedding(inputs)
        
        # hidden_outputs, hidden states = self.bi_gru(emjeoul_inputs)
        hidden_outputs, hiden_states = self.bi_lstm(emjeoul_inputs)
        
        # (batch_size, max_length, hidden_size*2)
        hidden_outputs = self.dropout(hidden_outputs)
        
        # (batch_size, max_length, hidden_size*2) -> (batch_size, max_length, number_of_labels)
        hypothesis = self.linear(hidden_outputs)
                
        return hypothesis

# 데이터셋 읽기 함수
def read_datas(file_path):
    with open(file_path, 'r', encoding='utf-8') as inFile:
        lines = inFile.readlines()
    datas = []
    for line in lines:
        # 입력 문장을 \t로 분리
        pieces = line.strip().split('\t')
        # 입력 문자열을 음절 단위로 분리
        emjeol_sequence, label_sequence = pieces[0].split(), pieces[1].split()
        datas.append((emjeol_sequence, label_sequence))
    return datas

def read_vocab_data(eumjeol_vocab_data_path):
    label2idx, idx2label = {"<PAD>":0, "B":1, "I":2}, {0:"<PAD>", 1:"B", 2:"I"}
    eumjeol2idx, idx2eumjeol = {}, {}
    
    with open(eumjeol_vocab_data_path, 'r', encoding='utf-8') as inFile:
        lines = inFile.readlines()
        
    for line in lines:
        eumjeol = line.strip()
        eumjeol2idx[eumjeol] = len(eumjeol2idx)
        idx2eumjeol[eumjeol2idx(eumjeol)] = eumjeol
        
    return eumjeol2idx, idx2eumjeol, label2idx, idx2label

def load_dataset(config):
    datas = read_datas(config["input_data"])
    eumjeol2idx, idx2eumjeol, label2idx, idx2label = read_vocab_data(config["eumjeol_vocab"])
    
    for eumjeol_sequence, label_sequence in datas:
        eumjeol_features = [eumjeol2idx[eumjeol] for eumjeol in eumjeol_sequence]
        label_features = [label2idx[label] for label in label_sequence]
        
        # 음절 sequence의 실제 길이
        eumjeol_features_length = len(eumjeol_features)
        
        # 모든 입력 데이터를 고정된 길이로 맞춰주기 위한 padding 처리
        eumjeol_features += [0] * (config["max_seq_len"] - eumjeol_features_length)
        label_features += [0] * (config["max_seq_len"] - eumjeol_features_length)

    return eumjeol_features, eumjeol_features_length, label_features, eumjeol2idx, idx2eumjeol, label2idx, idx2label
    

# 모델 학습 함수
def train(config):
       
    # 데이터 읽기
    eumjeol_features, eumjeol_features_length, label_features, eumjeol2idx, idx2eumjeol, label2idx, idx2label = load_dataset(config)
    
    # 모델 생성
    model = SpacingRNN(config).cuda()  # CPU 사용 시 .cuda() 제거
    # 사전학습한 모델 파일로부터 가중치 로드
    model.load_state_dict(torch.load(os.path.join(config["output_dir_path"], config["model_name"])))
        
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
        
        for step, batch in enumerate(train_dataloader):
            
            # batch = (input_features[step], labels[step])*batch_size
            # .cuda()를 통해 메모리에 업로드
            batch = tuple(t.cuda() for t in batch)
            
            inputs, input_lengths, labels = batch[0], batch[1], batch[2]
            
            # 역전파 변화도 초기화
            # .backward() 호출 시, 변화도 버퍼에 데이터가 계속 누적한 것을 초기화
            # 초기화 하지 않을 시 다음 작업에 영향이 갈 수 있음.
            optimizer.zero_grad()
            
            # 모델 출력 결과 얻어오기
            hypothesis = model(inputs)
            
            # hypothesis = (batch_size, max_length, number_of_labels) -> (batch_size*max_length, number_of_labels)
            # labels = (batch_size, max_length) -> (batch_size*max_length)
            
            # 비용 계산
            cost = loss_func(hypothesis.reshape(-1, len(label2idx)), labels.flatten())
            # 역전파 수행
            cost.backward()
            optimizer.step()
            
            # 현재 batch의 스탭 및 loss 저장
            costs.append(cost.data.item())
            
            if epoch%100 == 0:
                print("Average Loss = {0:f}".format(np.mean(costs)))
                torch.save(model.state_dict(), os.path.join(config["output_dir_path"], "epoch_{0:d}.pt".format(epoch)))
                do_test(model, train_dataloader)

# 모델 평가 함수            
def test(config):
    # 데이터 읽기
    eumjeol_features, eumjeol_features_length, label_features, eumjeol2idx, idx2eumjeol, label2idx, idx2label = load_dataset(config)
    # 모델 생성
    model = SpacingRNN(config).cuda()
    
    # 저장된 모델 가중치 로드
    model.load_state_dict(torch.load(os.path.join(config["output_dir_path"], config["model_name"])))
    
    # 데이터 load
    (input_features, labels) = load_dataset(config["input_data"])
    
    test_features = TensorDataset(input_features, labels)
    test_dataloader = DataLoader(test_features, shuffle=False, batch_size=config["batch_size"])
    
    for step, batch in enumerate(test_dataloader):
        # 음절 데이터, 각 데이터의 실제 길이, 라벨 데이터
        inputs, input_lengths, labels = batch[0], batch[1], batch[2]
        
        # 모델 평가
        hypothesis = model(inputs)
        
        # (batch_size, max_length, number_of_labels) -> (batch_size, max_length)
        hypothesis = torch.argmax(hypothesis, dim=-1)
        
        # batch_size가 1이기 때문
        input_length = tensor2list(input_lengths)[0]
        input = tensor2list(inputs)[0][:input_length]
        label = tensor2list(labels)[0][:input_length]
        hypothesis = tensor2list(hypothesis[0])[:input_length]
        
        # 출력 결과와 정답을 리스트에 저장
        total_hypothesis+=hypothesis
        total_label+=label
    if (step<10):
        # 정답과 모델 출력 비교
        predict_sentence, correct_sentence = make_sentence(input, hypothesis, label, idx2eumjeol, idx2label)
        print("정답 : "+correct_sentence)
        print("출력 : "+predict_sentence)
        print()
    
    do_test(model, test_dataloader)
    
def make_sentence(inputs, predicts, labels, idx2eumjeol, idx2label):
    predict_sentence, correct_sentence = "", ""
    
    for index in range(len(inputs)):
        eumjeol = idx2eumjeol[inputs[index]]
        correct_label = idx2label[labels[index]]
        predict_label = idx2label[predicts[index]]
        
        # 시작 음절ㅇㄴ 경우 공백을 추가해줄 필요가 없음
        if (index == 0):
            predict_sentence += eumjeol
            correct_sentence += eumjeol
            continue
    
        # "B" 태그인 경우 어절의 시작 음절이므로 앞에 공백 추가
        if (predict_label == "B"):
            predict_sentence += " "
        predict_sentence += eumjeol
        
        # "B" 태그인 경우 어절의 시작 음절이므로 앞에 공백 추가
        if (correct_label == "B"):
            correct_sentence += " "
        correct_sentence += eumjeol
            
    return predict_sentence, correct_sentence
    
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