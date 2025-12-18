from konlpy.tag import Mecab
from gensim.models import Word2Vec, KeyedVectors

texts = ["죽는 날까지 하늘을 우러러 한 점 부끄럼이 없기를",
         "잎새에 이는 바람에도 나는 괴로워했다.",
         "별을 노래하는 마음으로 모든 죽어가는 것을 사랑해야지",
         "그리고 나한테 주어진 길을 걸어가야겠다.",
         "오늘 밤에도 별이 바람에 스치운다."]

# Word2Vec 학습에 사용할 데이터 만들기

m = Mecab()
result = []

print("###########################################")
for sent in texts:
    tag = m.pos(sent)
    words = []
    for(lex, pos) in tag:
        print(lex, pos)
        if pos[0] == 'N':  # 명사류(체언)만 추출
            words.append(lex)
    result.append(words)
print("###########################################")
print(result, "\n")

# Word2Vec 학습시키기
model = Word2Vec(sentences=result,vector_size=10, window=1, min_count=1, workers=1, sg=0) # sg: 0=CBOW, 1=Skip-gram

# 값 읽어오기
print(model.wv['하늘'])

# 유사한 단어 가져오기
print(model.wv.most_similar('하늘'), "\n")

# # Word2Vec 모델 저장하기
# model.wv.save_word2vec_format('/gdrive/My_Drive/colab/text_rep/test_w2v')

# # Word2Vec 모델 로드하기
# loaded_model = KeyedVectors.load_word2vec_format('/gdrive/My_Drive/colab/text_rep/test_w2v')

# # 값 읽어오기
# print(loaded_model['하늘'])

# # 유사한 단어 가져오기
# print(loaded_model.wv.most_similar('하늘'))