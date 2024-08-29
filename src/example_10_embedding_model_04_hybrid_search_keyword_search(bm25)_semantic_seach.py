import os
from dotenv import load_dotenv

load_dotenv('.huggingface_env2')
print(os.environ['HF_HOME'])

from utils.common import ignore_warnings

from datasets import load_dataset
import math
import numpy as np
from typing import List
from transformers import PreTrainedTokenizer
from collections import defaultdict
from transformers import AutoTokenizer
from collections import defaultdict
from sentence_transformers import SentenceTransformer
import faiss
from llama_index.core import Document, VectorStoreIndex, ServiceContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# class BM25:
#     """
#     ## 예제 10.14 BM25 클래스 구현
#     """
#
#     def __init__(self, corpus: List[List[str]], tokenizer: PreTrainedTokenizer):
#         self.tokenizer = tokenizer
#         self.corpus = corpus
#         # self.tokenized_corpus = self.tokenizer(corpus, add_special_tokens=False)['input_ids']
#         self.tokenized_corpus = self.tokenizer(corpus, add_special_tokens=False, truncation=True, max_length=512)[
#             'input_ids']
#         self.n_docs = len(self.tokenized_corpus)
#         self.avg_doc_lens = sum(len(lst) for lst in self.tokenized_corpus) / len(self.tokenized_corpus)
#         self.idf = self._calculate_idf()
#         self.term_freqs = self._calculate_term_freqs()
#
#     def _calculate_idf(self):
#         idf = defaultdict(float)
#         for doc in self.tokenized_corpus:
#             for token_id in set(doc):
#                 idf[token_id] += 1
#         for token_id, doc_frequency in idf.items():
#             idf[token_id] = math.log(((self.n_docs - doc_frequency + 0.5) / (doc_frequency + 0.5)) + 1)
#         return idf
#
#     def _calculate_term_freqs(self):
#         term_freqs = [defaultdict(int) for _ in range(self.n_docs)]
#         for i, doc in enumerate(self.tokenized_corpus):
#             for token_id in doc:
#                 term_freqs[i][token_id] += 1
#         return term_freqs
#
#     def get_scores(self, query: str, k1: float = 1.2, b: float = 0.75):
#         query = self.tokenizer([query], add_special_tokens=False)['input_ids'][0]
#         scores = np.zeros(self.n_docs)
#         for q in query:
#             idf = self.idf[q]
#             for i, term_freq in enumerate(self.term_freqs):
#                 q_frequency = term_freq[q]
#                 doc_len = len(self.tokenized_corpus[i])
#                 score_q = idf * (q_frequency * (k1 + 1)) / (
#                         (q_frequency) + k1 * (1 - b + b * (doc_len / self.avg_doc_lens)))
#                 scores[i] += score_q
#         return scores
#
#     def get_top_k(self, query: str, k: int):
#         scores = self.get_scores(query)
#         top_k_indices = np.argsort(scores)[-k:][::-1]
#         top_k_scores = scores[top_k_indices]
#         return top_k_scores, top_k_indices

class BM25:
    """
    ## 예제 10.14 BM25 클래스 구현
    """

    def __init__(self, corpus: List[List[str]], tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer
        self.corpus = corpus
        self.tokenized_corpus = self.tokenizer(corpus, add_special_tokens=False, truncation=True, max_length=512)['input_ids']
        self.n_docs = len(self.tokenized_corpus)
        self.avg_doc_lens = sum(len(lst) for lst in self.tokenized_corpus) / len(self.tokenized_corpus)
        self.idf = self._calculate_idf()
        self.term_freqs = self._calculate_term_freqs()

    def _calculate_idf(self):
        idf = defaultdict(float)
        for doc in self.tokenized_corpus:
            for token_id in set(doc):
                idf[token_id] += 1
        for token_id, doc_frequency in idf.items():
            idf[token_id] = math.log(((self.n_docs - doc_frequency + 0.5) / (doc_frequency + 0.5)) + 1)
        return idf

    def _calculate_term_freqs(self):
        term_freqs = [defaultdict(int) for _ in range(self.n_docs)]
        for i, doc in enumerate(self.tokenized_corpus):
            for token_id in doc:
                term_freqs[i][token_id] += 1
        return term_freqs

    def get_scores(self, query: str, k1: float = 1.2, b: float = 0.75):
        query = self.tokenizer([query], add_special_tokens=False)['input_ids'][0]
        scores = np.zeros(self.n_docs)
        for q in query:
            idf = self.idf[q]
            for i, term_freq in enumerate(self.term_freqs):
                q_frequency = term_freq[q]
                doc_len = len(self.tokenized_corpus[i])
                score_q = idf * (q_frequency * (k1 + 1)) / (
                        (q_frequency) + k1 * (1 - b + b * (doc_len / self.avg_doc_lens)))
                scores[i] += score_q
        return scores

    def get_top_k(self, query: str, k: int):
        scores = self.get_scores(query)
        top_k_indices = np.argsort(scores)[-k:][::-1]
        top_k_scores = scores[top_k_indices]
        return top_k_scores, top_k_indices





def reciprocal_rank_fusion(rankings: List[List[int]], k=5):
    """
    ## 예제 10.18 상호 순위 조합 함수 구현

    :param rankings:
    :param k:
    :return:
    """
    rrf = defaultdict(float)
    for ranking in rankings:
        for i, doc_id in enumerate(ranking, 1):
            rrf[doc_id] += 1.0 / (k + i)
    return sorted(rrf.items(), key=lambda x: x[1], reverse=True)


def dense_vector_search(sentence_model, index, query: str, k: int):
    query_embedding = sentence_model.encode([query])
    distances, indices = index.search(query_embedding, k)
    return distances[0], indices[0]


def hybrid_search(sentence_model, index, query, k=20):
    """
    ## 예제 10.20 하이브리드 검색 구현하기
    :param query: 
    :param k: 
    :return: 
    """
    _, dense_search_ranking = dense_vector_search(sentence_model, index, query, 100)
    _, bm25_search_ranking = bm25.get_top_k(query, 100)

    results = reciprocal_rank_fusion([dense_search_ranking, bm25_search_ranking], k=k)
    return results


if __name__ == '__main__':
    ignore_warnings()

    print("## 예제 10.8 실습에 사용할 모델과 데이터셋 불러오기")
    klue_mrc_dataset = load_dataset('klue', 'mrc', split='train')

    sentence_model_name = "snunlp/KR-SBERT-V40K-klueNLI-augSTS"
    sentence_model = SentenceTransformer(sentence_model_name)
    #
    print("## 예제 10.9 실습 데이터에서 1,000개만 선택하고 문장 임베딩으로 변환")
    klue_mrc_dataset = klue_mrc_dataset.train_test_split(train_size=1000, shuffle=False)['train']
    embeddings = sentence_model.encode(klue_mrc_dataset['context'])
    print(embeddings.shape)
    # 출력 결과
    # (1000, 768)

    print("\n## 예제 10.10 KNN 검색 인덱스를 생성하고 문장 임베딩 저장")
    print("# 인덱스 만들기")
    index = faiss.IndexFlatL2(embeddings.shape[1])  # IndexFlatL2: 기본 KNN 알로리즘을 사용해, index를 생성함
    print("# 인덱스에 임베딩 저장하기")
    index.add(embeddings)

    print("## 예제 10.15 BM25 점수 계산 확인해 보기")
    tokenizer = AutoTokenizer.from_pretrained('klue/roberta-base')

    bm25 = BM25(['안녕하세요', '반갑습니다', '안녕 서울'], tokenizer)
    print(bm25.get_scores('안녕'))
    # array([0.44713859, 0.        , 0.52354835])

    print("\n## 예제 10.16 BM25 검색 결과의 한계")
    # BM25 검색 준비
    bm25 = BM25(klue_mrc_dataset['context'], tokenizer)

    query = "이번 연도에는 언제 비가 많이 올까?"
    print(query)
    _, bm25_search_ranking = bm25.get_top_k(query, 100)

    for idx in bm25_search_ranking[:3]:
        print(klue_mrc_dataset['context'][idx][:50])
    # 출력 결과
    # 갤럭시S5 언제 발매한다는 건지언제는 “27일 판매한다”고 했다가 “이르면 26일 판매한다 (오답)
    # 인구 비율당 노벨상을 세계에서 가장 많이 받은 나라, 과학 논문을 가장 많이 쓰고 의료 특 (오답)
    # 올여름 장마가 17일 제주도에서 시작됐다. 서울 등 중부지방은 예년보다 사나흘 정도 늦은  (정답)

    print("\n## 예제 10.17 BM25 검색 결과의 장점")
    query = klue_mrc_dataset[3]['question']  # 로버트 헨리 딕이 1946년에 매사추세츠 연구소에서 개발한 것은 무엇인가?
    print(query)
    _, bm25_search_ranking = bm25.get_top_k(query, 100)

    for idx in bm25_search_ranking[:3]:
        print(klue_mrc_dataset['context'][idx][:50])
    # 출력 결과
    # 미국 세인트루이스에서 태어났고, 프린스턴 대학교에서 학사 학위를 마치고 1939년에 로체스 (정답)
    # ;메카동(メカドン)                                                      (오답)
    # :성우 : 나라하시 미키(ならはしみき)
    # 길가에 버려져 있던 낡은 느티나
    # ;메카동(メカドン)                                                      (오답)
    # :성우 : 나라하시 미키(ならはしみき)
    # 길가에 버려져 있던 낡은 느티나

    print("\n## 예제 10.18 상호 순위 조합 함수 구현")
    print("\n## 예제 10.19 예시 데이터에 대한 상호 순위 조합 결과 확인하기")
    rankings = [[1, 4, 3, 5, 6], [2, 1, 3, 6, 4]]
    print(reciprocal_rank_fusion(rankings))
    # [(1, 0.30952380952380953),
    #  (3, 0.25),
    #  (4, 0.24285714285714285),
    #  (6, 0.2111111111111111),
    #  (2, 0.16666666666666666),
    #  (5, 0.1111111111111111)]

    print("\n## 예제 10.20 하이브리드 검색 구현하기")
    print("\n## 예제 10.21 예시 데이터에 대한 하이브리드 검색 결과 확인")

    query = "이번 연도에는 언제 비가 많이 올까?"
    print("검색 쿼리 문장: ", query)
    results = hybrid_search(sentence_model, index, query)
    for idx, score in results[:3]:
        print(klue_mrc_dataset['context'][idx][:50])

    print("=" * 80)
    query = klue_mrc_dataset[3]['question']  # 로버트 헨리 딕이 1946년에 매사추세츠 연구소에서 개발한 것은 무엇인가?
    print("검색 쿼리 문장: ", query)

    results = hybrid_search(sentence_model, index, query)
    for idx, score in results[:3]:
        print(klue_mrc_dataset['context'][idx][:50])
    # 출력 결과
    # 검색 쿼리 문장:  이번 연도에는 언제 비가 많이 올까?
    # 올여름 장마가 17일 제주도에서 시작됐다. 서울 등 중부지방은 예년보다 사나흘 정도 늦은  (정답)
    # 갤럭시S5 언제 발매한다는 건지언제는 “27일 판매한다”고 했다가 “이르면 26일 판매한다  (오답)
    # 연구 결과에 따르면, 오리너구리의 눈은 대부분의 포유류보다는 어류인 칠성장어나 먹장어, 그 (오답)
    # ================================================================================
    # 검색 쿼리 문장:  로버트 헨리 딕이 1946년에 매사추세츠 연구소에서 개발한 것은 무엇인가?
    # 미국 세인트루이스에서 태어났고, 프린스턴 대학교에서 학사 학위를 마치고 1939년에 로체스 (정답)
    # 1950년대 말 매사추세츠 공과대학교의 동아리 테크모델철도클럽에서 ‘해커’라는 용어가 처음 (오답)
    # 1950년대 말 매사추세츠 공과대학교의 동아리 테크모델철도클럽에서 ‘해커’라는 용어가 처음 (오답)
