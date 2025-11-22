import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel, BertConfig
import torch
from torch.utils.data import (DataLoader, RandomSampler, TensorDataset)
from kobert_tokenizer import KoBERTTokenizer

class SentimentClassifier(BertPreTrainedModel):
    
    def __init__(self, config):
        # BERT 사전학습 모델 생성자 오버라이딩
        super(SentimentClassifier, self).__init__(config)
        
        # BERT 모델
        self.bert = BertModel(config)
        
        # 히든 사이즈
        self.hidden_size = config.hidden_size
        
        # 분류할 라벨의 개수
        self.num_labels = config.num_labels
        
        self.linear = nn.Linear(in_features=self.hidden_size, out_features=self.num_labels)
    
    def forward(self, input_ids):
        outputs = self.bert(input_ids=input_ids)
        
        # (batch_size, max_length, hidden_size)
        bert_output = outputs[0]
        
        # (batch_size, hidden_size)
        cls_vector = bert_output[:, 0, :]
        
        # class_output : (batch_size,num_labels)
        cls_output = self.linear(cls_vector)
        
        return cls_output
    
def read_data(file_path):
    with open(file_path, "r", encoding="utf-8") as inFile:
        lines = inFile.readlines()
        
    datas = []
    for line in lines:
        # 입력 데이터를 \t을 기준으로 분리
        pieces = line.strip().split('\t')
        
        # 리뷰, 정답
        input_sequence, label = pieces[0].split(" "), pieces[1]
        
        datas.append((input_sequence, label))
        
    return datas

def read_vocab_data(vocab_data_path):
    term2idx, idx2term = {}, {}
    
    with open(vocab_data_path, "r", encoding="utf-8")as inFile:
        lines = inFile.readlines()
    
    for line in lines:
        term = line.strip()
        term2idx[term] = len(term2idx)
        idx2term[term2idx[term]] = term
    
    return term2idx, idx2term

def convert_data2feature(datas, max_length, tokenizer, label2idx):
    input_ids_features, label_id_features = [], []
    
    for input_sequence, label in datas:
        # CLS, SEP 토큰 추가
        tokens = [tokenizer.cls_token]
        tokens += input_sequence
        tokens = tokens[:max_length-1]
        tokens += [tokenizer.sep_token]
        
        # word piece들을 대응하는 index로 치환
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        
        # padding 생성
        padding = [tokenizer._convert_token_to_id(tokenizer.pad_token)] + (max_length-len(input_ids))
        input_ids += padding
        
        label_id = label2idx[label]
        
        # 변환한 데이터를 각 리스트에 저장
        input_ids_features.append(input_ids)
        label_id_features.append(label_id)
        
def train(config):
    # BERT config 객체 생성
    bert_config = BertConfig.from_pretrained(pretraied_model_name_or_path=config["pretrained_model_name_or_path"],
                                             cache_dir=config["cache_dir_path"])
    setattr(bert_config, "num_labels", config["num_labels"])
    
    # BERT tokenizer 객체 생성
    bert_tokenizer = KoBERTTokenizer.from_pretrained(pretrained_model_nam_or_path=config["pretrained_model_name_or_path"],
                                                     cache_dir=config["cache_dir_path"])
    
    # 라벨 딕셔너리 생성
    label2idx, idx2label = read_vocab_data(vocab_data_path=config["label_vocab_data_path"])
    
    # 학습 및 평가 데이터 읽기
    train_datas = read_data(file_path=config["train_data_path"])
    
    # 입력 데이터 전처리
    train_input_ids_features, train_label_id_features = convert_data2feature(datas=train_datas,
                                                                             max_length=config["max_length"],
                                                                             tokenizer=bert_tokenizer,
                                                                             label2idx=label2idx)
    
    # 학습 데이터를 batch 단위로 추출하기 위한 DataLoader 객체 생성
    train_dataset = TensorDataset(train_input_ids_features, train_label_id_features)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=config["batch_size"],
                                  sampler=RandomSampler(train_dataset))
    
    # 사전 학습된 BERT 모델 파일로부터 가중치 불러옴
    model = SentimentClassifier.from_pretrained(pretrained_model_name_or_path=config["pretrained_model_name_or_path"],
                                                cache_dir=config["cache_dir_path"], config=bert_config).cuda()
    
    # loss를 계산하기 위한 함수
    loss_func = nn.CrossEntropyLoss()
    
    # 모델 학습을 위한 optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    
    for epoch in range(config["epoch"]):
        model.train()
        
        total_loss = []
        for batch in train_dataloader:
            batch = tuple(t.cuda() for t in batch)
            input_ids, label_id = batch
            
            # 역전파 단계를 실행하기 전에 변화도를 0으로 변경
            optimizer.zero_grad()
            
            # 모델 예측 결과
            hypothesis = model(input_ids)
            
            # loss 계산
            loss = loss_func(hypothesis, label_id)
            
            # loss 값으로부터 모델 내부 각 매개변수에 대하여 gradient 계산
            loss.backward()
            # 모델 내부 각 매개변수 가중치 갱신
            optimizer.step()
            
def test(config):
    # BERT config 객체 생성
    bert_config = BertConfig.from_pretrained(pretrained_model_nam_or_path=config["pretrained_model_name_or_path"],
                                                     cache_dir=config["cache_dir_path"])
    
    # BERT tokenizer 객체 생성
    bert_tokenizer = KoBERTTokenizer.from_pretrained(pretrained_model_nam_or_path=config["pretrained_model_name_or_path"],
                                                     cache_dir=config["cache_dir_path"])
    # 라벨 딕셔너리 생성
    label2idx, idx2label = read_vocab_data(vocab_data_path=config["label_vocab_data_path"])
    
    # 학습 및 평가 데이터 읽기
    test_datas = read_data(file_path=config["train_data_path"])
    test_datas = test_datas[:100]
    
    # 입력 데이터 전처리
    test_input_ids_features, test_label_id_features = convert_data2feature(datas=test_datas,
                                                                             max_length=config["max_length"],
                                                                             tokenizer=bert_tokenizer,
                                                                             label2idx=label2idx)
    # 학습한 모델 파일로부터 가중치 불러오기
    model = SentimentClassifier.from_pretrained(pretrained_model_name_or_path=config["pretrained_model_name_or_path"],
                                                cache_dir=config["cache_dir_path"], config=bert_config).cuda()
    # 학습 데이터를 batch 단위로 추출하기 위한 DataLoader 객체 생성
    test_dataset = TensorDataset(test_input_ids_features, test_label_id_features)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=config["batch_size"],
                                  sampler=RandomSampler(test_dataset))
    
    model.eval()
    
    for batch in test_dataloader:
        model.train()
        
        total_loss = []
        for batch in test_dataloader:
            batch = tuple(t.cuda() for t in batch)
            input_ids, label_id = batch
            
            with torch.no_grad():
                # 모델 예측 결과
                hypothesis = model(input_ids)
                # 모델의 출력값에 softmax와 argmax 함수를 적용
                hypothesis = torch.argmax(torch.softmax(hypothesis, dim=-1), dim=-1)
            
            # Tensor를 리스트로 변경
            hypothesis = hypothesis.cpu().detach().numpy().tolist()
            label_id = label_id.cpu().detach().numpy().tolist()
            
            for index in range(len(input_ids)):
                input_tokens = bert_tokenizer.convert_ids_to_tokens(input_ids[index])
                input_sequence = bert_tokenizer.convert_tokens_to_string(input_tokens[1:input_tokens.index(bert_tokenizer.sep_token)])
                predict = idx2label[hypothesis[index]]
                correct = idx2label[label_id[index]]
                
                print("입력: {}".format(input_sequence))
                print("출력: {}, 정답: {}\n".format(predict, correct))