# 4.Word2Vec.py에서 생성한 전처리된 데이터를 사용
from konlpy.tag import Mecab

texts = ["죽는 날까지 하늘을 우러러 한 점 부끄럼이 없기를",
         "잎새에 이는 바람에도 나는 괴로워했다.",
         "별을 노래하는 마음으로 모든 죽어가는 것을 사랑해야지",
         "그리고 나한테 주어진 길을 걸어가야겠다.",
         "오늘 밤에도 별이 바람에 스치운다."]
m = Mecab()
result = []

for sent in texts:
    tag = m.pos(sent)
    words = []
    for(lex, pos) in tag:
        # print(lex, pos)
        if pos[0] == 'N':  # 명사류(체언)만 추출
            words.append(lex)
    result.append(words)
print(result, "\n")

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

# 값 읽어오기
print(glove.word_vectors[glove.dictionary['하늘']])
# 유사한 단어 가져오기
print(glove.most_similar('하늘'), '\n')

# # Glove 모델 저장
# glove.save('/gdrive/My_Drive/colab/text_rep/test_glove')

# # Glove 모델 로드하기
# loaded_model = glove.load('/gdrive/My_Drive/colab/text_rep/test_glove')

# # 값 읽어오기
# print(loaded_model.word_vectors[glove.dictionary['하늘']])
# # 유사한 단어 가져오기
# print(loaded_model.most_similar('하늘'))
