import torch
import torch.nn as nn

# 임베딩 층 수동으로 만들어보기 ex) I like football이라는 문장이 들어오면 기존 vocab을 참고해서 현재 입력된 문장이 어떤 임베딩 층으로 매핑되는지 보여주는 코드
# 사전 만들기
train_data = 'I like deep learning I like NLP I enjoy flying'

# 중복으 제거한 단어들의 집합인 단어 집합 생성
word_set = set(train_data.split())

# 단어 집합의 각 단어에 고유한 정수 맵핑
vocab = {word: i+2 for i, word in enumerate(word_set)} # 아래의 unk, pad가 각각 index 0, 1을 차지하기 때문에 2부터 저장한다.
vocab['unk'] = 0
vocab['pad'] = 1
print(vocab)

# 임베딩 테이블: (단어 수, 임베딩 사이즈)
embedding_table = torch.FloatTensor([[0.0, 0.0, 0.0],
                                     [0.0, 0.0, 0.0],
                                     [0.1, 0.8, 0.3],
                                     [0.7, 0.8, 0.2],
                                     [0.1, 0.8, 0.7],
                                     [0.9, 0.2, 0.1],
                                     [0.1, 0.1, 0.9],
                                     [0.2, 0.1, 0.7],
                                     [0.3, 0.1, 0.1]])

# 임베딩 층 만들기
# 입력 문장
input_snt = 'I like football'.split()

# 각 단어를 정수로 변환
idxes = []

for word in input_snt:
    idx = vocab[word] if word in vocab else vocab['unk']
    idxes.append(idx)

idxes = torch.LongTensor(idxes)
print(idxes)

# 입력 문장의 임베딩 가져오기: ['I', 'like', 'football(unk)']
lookup_result = embedding_table[idxes, :]
print(lookup_result)

# 임베딩 층 만들기
embedding_layer = nn.Embedding(num_embeddings=len(vocab), embedding_dim=3, padding_idx=1)
print(embedding_layer.weight)