# 4.Word2Vec.py에서 생성한 전처리된 데이터를 사용

# pip install glove-python3
# mac에선 glove가 설치되지 않음 window에서 다시 해야할 듯

# Co-Occurrence Matrix 생성
from glove import Corpus

corpus = Corpus()
corpus.fit(result, window=1)

# GloVe 학습 시키기
from glove import Glove
glove = Glove(no_components=10, learning_rate=0.05) # 10차원 임베딩
glove.fit(corpus.matrix, epochs=20, no_threads=1, verbose=True)
glove.add_dictionary(corpus.dictionary)

# 값 읽어ㅗ기
print(glove.word_vectors[glove.dictionary['하늘']])
# 유사한 단어 가져오기
print(glove.most_similar('하늘'), '\n')

# Glove 모델 저장
glove.save('/gdrive/My_Drive/colab/text_rep/test_glove')

# Glove 모델 로드하기
loaded_model = glove.load('/gdrive/My_Drive/colab/text_rep/test_glove')

# 값 읽어오기
print(loaded_model.word_vectors[glove.dictionary['하늘']])
# 유사한 단어 가져오기
print(loaded_model.most_similar('하늘'))
