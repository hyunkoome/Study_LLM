import os
from dotenv import load_dotenv

load_dotenv('.huggingface_env2')
print(os.environ['HF_HOME'])

import torch
from utils.common import ignore_warnings
from sentence_transformers import SentenceTransformer, models, util
from PIL import Image


def mean_pooling(model_output, attention_mask, mode='mean'):
    """
    ## 예제 10.4 코드로 살펴보는 평균 모드 + ## 예제 10.5 코드로 살펴보는 최대 모드

    :param model_output:
    :param attention_mask:
    :return:
    """
    if mode == 'max':
        ## 예제 10.5 코드로 살펴보는 최대 모드
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        token_embeddings[input_mask_expanded == 0] = -1e9
        return torch.max(token_embeddings, 1)[0]

    ## 예제 10.4 코드로 살펴보는 평균 모드
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


if __name__ == '__main__':
    ignore_warnings()

    print("## 예제 10.3 Sentence-Transformers 라이브러리로 바이 인코더 생성하기")
    # 사용할 BERT 모델
    word_embedding_model = models.Transformer('klue/roberta-base')
    # 풀링 층 차원 입력하기
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    # 두 모듈 결합하기
    model1 = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    print("\n## 예제 10.6 한국어 문장 임베딩 모델로 입력 문장 사이의 유사도 계산")
    model2 = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')
    sentences = ['잠이 안 옵니다',
                 '졸음이 옵니다',
                 '기차가 옵니다']
    embs = model2.encode(sentences)
    cos_scores2 = util.cos_sim(embs, embs)
    print(cos_scores2)
    print(f"{sentences[0]} <-> {sentences[1]}: {cos_scores2[0][1]}")
    print(f"{sentences[1]} <-> {sentences[2]}: {cos_scores2[1][2]}")
    print(f"{sentences[0]} <-> {sentences[2]}: {cos_scores2[0][2]}")
    # tensor([[1.0000, 0.6410, 0.1887],
    #         [0.6410, 1.0000, 0.2730],
    #         [0.1887, 0.2730, 1.0000]])

    print("\n## 예제 10.7 CLIP 모델을 활용한 이미지와 텍스트 임베딩 유사도 계산")
    model3 = SentenceTransformer('clip-ViT-B-32')
    img_embs = model3.encode([Image.open('./data/dog.jpg'), Image.open('./data/cat.jpg')])
    print("img_embs: ", img_embs)
    text_embs = model3.encode(['A dog on grass', 'Brown cat on yellow background'])
    print("text_embs: ", text_embs)
    cos_scores3 = util.cos_sim(img_embs, img_embs)
    print("cos_scores3: ", cos_scores3)
    cos_scores4 = util.cos_sim(text_embs, text_embs)
    print("cos_scores4: ", cos_scores4)
    cos_scores5 = util.cos_sim(img_embs, text_embs)
    print("cos_scores5: ", cos_scores5)
    # # tensor([[0.2771, 0.1509],
    # #         [0.2071, 0.3180]])
    # tensor[0, 0] = 0.2771: dog.jpg 와 'A dog on grass'
    # tensor[0, 1] = 0.1509: dog.jpg 와 'Brown cat on yellow background'
    # tensor[1, 0] = 0.2071: cat.jpg 와 'A dog on grass'
    # tensor[1, 1] = 0.3180: cat.jpg 와 'Brown cat on yellow background'
