import os
from dotenv import load_dotenv

load_dotenv('.huggingface_env2')
print(os.environ['HF_HOME'])

from utils.common import ignore_warnings
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from sentence_transformers import InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers import datasets
from sentence_transformers import losses
from huggingface_hub import HfApi
from pathlib import Path
from huggingface_hub import login
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CECorrelationEvaluator
from datasets import load_dataset
import faiss
from sentence_transformers import SentenceTransformer
import time
import numpy as np
from tqdm.auto import tqdm


# def prepare_sts_examples(dataset):
#     """
#     # 유사도 점수를 0~1 사이로 정규화 하고 InputExample 객체에 담는다.
#
#     :param dataset:
#     :return:
#     """
#     examples = []
#     for data in dataset:
#         examples.append(
#             InputExample(
#                 texts=[data['sentence1'], data['sentence2']],
#                 label=data['labels']['label'] / 5.0)
#             )
#     return examples

# def add_ir_context(df):
#     """
#     ## 예제 11.14 질문과 관련이 없는 기사를 irrelevant_context 컬럼에 추가
#     :param df:
#     :return:
#     """
#     irrelevant_contexts = []
#     for idx, row in df.iterrows():
#         title = row['title']
#         irrelevant_contexts.append(df.query(f"title != '{title}'").sample(n=1)['context'].values[0])
#     df['irrelevant_context'] = irrelevant_contexts
#     return df

## 예제 11.30 임베딩을 저장하고 검색하는 함수 구현
def make_embedding_index(sentence_model, corpus):
    """

    :param sentence_model:
    :param corpus:
    :return:
    """
    embeddings = sentence_model.encode(corpus)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index


def find_embedding_top_k(query, sentence_model, index, k):
    """

    :param query:
    :param sentence_model:
    :param index:
    :param k:
    :return:
    """
    embedding = sentence_model.encode([query])
    distances, indices = index.search(embedding, k)
    return indices


## 예제 11.31 교차 인코더를 활용한 순위 재정렬 함수 정의
def make_question_context_pairs(question_idx, indices):
    """

    :param question_idx:
    :param indices:
    :return:
    """
    return [[klue_mrc_test['question'][question_idx], klue_mrc_test['context'][idx]] for idx in indices]


def rerank_top_k(cross_model, question_idx, indices, k):
    """

    :param cross_model:
    :param question_idx:
    :param indices:
    :param k:
    :return:
    """
    input_examples = make_question_context_pairs(question_idx, indices)
    relevance_scores = cross_model.predict(input_examples)
    reranked_indices = indices[np.argsort(relevance_scores)[::-1]]

    # 상위 k개의 인덱스만 선택
    top_k_indices = reranked_indices[:k]

    return top_k_indices


def evaluate_hit_rate(datasets, embedding_model, index, k=10):
    """
    ## 예제 11.32 성능 지표(히트율)와 평가에 걸린 시간을 반환하는 함수 정의

    :param datasets:
    :param embedding_model:
    :param index:
    :param k:
    :return:
    """
    start_time = time.time()
    predictions = []
    for question in datasets['question']:
        predictions.append(find_embedding_top_k(question, embedding_model, index, k)[0])
    total_prediction_count = len(predictions)
    hit_count = 0
    questions = datasets['question']
    contexts = datasets['context']
    for idx, prediction in enumerate(predictions):
        for pred in prediction:
            if contexts[pred] == contexts[idx]:
                hit_count += 1
                break
    end_time = time.time()
    return hit_count / total_prediction_count, end_time - start_time


def evaluate_hit_rate_with_rerank(datasets, embedding_model, cross_model, index, bi_k=30, cross_k=10):
    """
    ## 예제 11.35 순위 재정렬을 포함한 평가 함수
    :param datasets:
    :param embedding_model:
    :param cross_model:
    :param index:
    :param bi_k:
    :param cross_k:
    :return:
    """
    start_time = time.time()
    predictions = []
    for question_idx, question in enumerate(tqdm(datasets['question'])):
        indices = find_embedding_top_k(question, embedding_model, index, bi_k)[0]
        predictions.append(rerank_top_k(cross_model, question_idx, indices, k=cross_k))
    total_prediction_count = len(predictions)
    hit_count = 0
    questions = datasets['question']
    contexts = datasets['context']
    for idx, prediction in enumerate(predictions):
        for pred in prediction:
            if contexts[pred] == contexts[idx]:
                hit_count += 1
                break
    end_time = time.time()
    return hit_count / total_prediction_count, end_time - start_time, predictions


if __name__ == '__main__':
    ignore_warnings()

    number_of_test = 1000

    print(f"## 예제 11.29 평가를 위한 데이터셋을 불러와 {number_of_test}개만 선별")
    klue_mrc_test = load_dataset('klue', 'mrc', split='validation')
    klue_mrc_test = klue_mrc_test.train_test_split(test_size=number_of_test, seed=42)['test']

    print("\n## 예제 11.33 기본 임베딩 모델 평가")
    base_embedding_model = SentenceTransformer('hyunkookim/klue-roberta-base-klue-sts')
    base_index = make_embedding_index(base_embedding_model, klue_mrc_test['context'])
    hit_rate, cosumed_time = evaluate_hit_rate(klue_mrc_test, base_embedding_model, base_index, 10)
    print(f"hit_rate: {hit_rate}")
    print(f"cosumed_time: {cosumed_time}")
    # hit_rate: 0.864
    # cosumed_time: 5.510005474090576

    print("\n## 예제 11.34 미세 조정한 임베딩 모델 평가")
    finetuned_embedding_model = SentenceTransformer('hyunkookim/klue-roberta-base-klue-sts-mrc')
    finetuned_index = make_embedding_index(finetuned_embedding_model, klue_mrc_test['context'])
    hit_rate, cosumed_time = evaluate_hit_rate(klue_mrc_test, finetuned_embedding_model, finetuned_index, 10)
    print(f"hit_rate: {hit_rate}")
    print(f"cosumed_time: {cosumed_time}")
    # hit_rate: 0.95
    # cosumed_time: 5.402112007141113

    print("\n## 예제 11.36 임베딩 모델과 교차 인코드를 조합해 성능 평가")
    cross_model = CrossEncoder('hyunkookim/klue-roberta-small-cross-encoder')
    hit_rate, cosumed_time, predictions = evaluate_hit_rate_with_rerank(klue_mrc_test, finetuned_embedding_model,
                                                                        cross_model, finetuned_index, bi_k=30,
                                                                        cross_k=10)
    print(f"hit_rate: {hit_rate}")
    print(f"cosumed_time: {cosumed_time}")
    # hit_rate: 0.971
    # cosumed_time: 295.50445079803467

