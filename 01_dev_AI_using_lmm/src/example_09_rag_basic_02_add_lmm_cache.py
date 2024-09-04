import os
from dotenv import load_dotenv

load_dotenv('.huggingface_env2')
print(os.environ['HF_HOME'])

import time
import chromadb
from openai import OpenAI
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction


def response_text(openai_resp):
    return openai_resp.choices[0].message.content


class OpenAIOnlyEqualCache:
    """
    예제 9.8 파이썬 딕셔너리를 활용한 일치 캐시 구현
    """

    def __init__(self, openai_client):
        self.openai_client = openai_client
        self.cache = {}

    def generate(self, prompt):
        if prompt not in self.cache:
            response = self.openai_client.chat.completions.create(
                model='gpt-3.5-turbo',
                messages=[
                    {
                        'role': 'user',
                        'content': prompt
                    }
                ],
            )
            self.cache[prompt] = response_text(response)
        return self.cache[prompt]


class OpenAICache:
    """
    예제 9.8 파이썬 딕셔너리를 활용한 일치 캐시 구현 + 예제 9.9 유사 검색 캐시 추가 구현
    """

    def __init__(self, openai_client, semantic_cache):
        self.openai_client = openai_client
        self.cache = {}  # 일치 캐시
        self.semantic_cache = semantic_cache  # 유사 검색 캐시 (크로마 벡터 데이터베이스 클라이언트)

    def generate(self, prompt):
        if prompt not in self.cache:  # 입력 프롬프트가 캐쉬에 없다면
            # (크로마) 벡터 데이터베이스에 등록된 임베딩 모델을 사용해, 텍스트를 임베딩 벡터로 변환하고 검색 수행
            similar_doc = self.semantic_cache.query(query_texts=[prompt], n_results=1)

            # 충분히 가까운지(유사도가 0.2 미만인지?)
            if len(similar_doc['distances'][0]) > 0 and similar_doc['distances'][0][0] < 0.2:
                print("*** 유사 캐쉬 동작 ***")
                return similar_doc['metadatas'][0][0]['response']  # 검색된 문서 반환
            else:  # 충분히 가깝지 않다면(유사도가..)
                print("*** 새롭게 결과 생성 ***")
                response = self.openai_client.chat.completions.create(
                    model='gpt-3.5-turbo',
                    messages=[
                        {
                            'role': 'user',
                            'content': prompt
                        }
                    ],
                )
                # 일치 캐쉬에 저장
                self.cache[prompt] = response_text(response)
                # 유사 검색 캐쉬에 저장
                self.semantic_cache.add(documents=[prompt], metadatas=[{"response": response_text(response)}],
                                        ids=[prompt])
        print("*** 일치 캐쉬 반환 ***")
        return self.cache[prompt]  # 캐쉬 반환


if __name__ == '__main__':
    print("9.2절 LLM 캐시")
    print("예제 9.6 실습에 사용할 OpenAI와 크로마 클라이언트 생성")
    # os.environ["OPENAI_API_KEY"] = "자신의 OpenAI API 키 입력"
    openai_client = OpenAI()
    chroma_client = chromadb.Client()

    print("\n예제 9.7 LLM 캐시를 사용하지 않았을 때 동일한 요청 처리에 걸린 시간 확인")
    question = "북태평양 기단과 오호츠크해 기단이 만나 국내에 머무르는 기간은?"
    for _ in range(2):
        start_time = time.time()
        response = openai_client.chat.completions.create(
            model='gpt-3.5-turbo',
            messages=[
                {
                    'role': 'user',
                    'content': question
                }
            ],
        )
        response = response_text(response)
        print(f'질문: {question}')
        print("소요 시간: {:.2f}s".format(time.time() - start_time))
        print(f'답변: {response}\n')
    # 질문: 북태평양 기단과 오호츠크해 기단이 만나 국내에 머무르는 기간은?
    # 소요 시간: 2.71s
    # 답변: 북태평양 기단과 오호츠크해 기단이 만나 국내에 머무르는 기간은 겨울 시즌인 11월부터 다음 해 3월까지입니다.
    #       이 기간 동안 기온이 급격히 하락하며 한반도에 한기가 밀려오게 됩니다.

    # 질문: 북태평양 기단과 오호츠크해 기단이 만나 국내에 머무르는 기간은?
    # 소요 시간: 4.13s
    # 답변: 북태평양 기단과 오호츠크해 기단은 겨울에 만나 국내에 머무르는 것이 일반적입니다.
    #       이 기단들은 주로 11월부터 2월이나 3월까지 국내에 영향을 미치며,
    #       한국의 겨울철 추위와 함께 한반도 전역에 형성되는 강한 서북풍과 냉기를 가져옵니다.

    print("\n# 예제 9.8 파이썬 딕셔너리를 활용한 일치 캐시 구현")
    openai_cache = OpenAIOnlyEqualCache(openai_client)
    question = "북태평양 기단과 오호츠크해 기단이 만나 국내에 머무르는 기간은?"
    for _ in range(2):
        start_time = time.time()
        response = openai_cache.generate(question)
        print(f'질문: {question}')
        print("소요 시간: {:.2f}s".format(time.time() - start_time))
        print(f'답변: {response}\n')
    # 질문: 북태평양 기단과 오호츠크해 기단이 만나 국내에 머무르는 기간은?
    # 소요 시간: 2.74s
    # 답변: 북태평양 기단과 오호츠크해 기단이 만나 국내에 머무르는 기간은 겨울 시즌인 11월부터 다음해 4월까지입니다.
    #       이 기간 동안 기단의 영향으로 한반도에는 추운 날씨와 함께 강한 바람이 불게 되며, 대체로 한반도의 겨울철 기온은 매우 낮아집니다.

    # 질문: 북태평양 기단과 오호츠크해 기단이 만나 국내에 머무르는 기간은?
    # 소요 시간: 0.00s
    # 답변: 북태평양 기단과 오호츠크해 기단이 만나 국내에 머무르는 기간은 겨울 시즌인 11월부터 다음해 4월까지입니다.
    #       이 기간 동안 기단의 영향으로 한반도에는 추운 날씨와 함께 강한 바람이 불게 되며, 대체로 한반도의 겨울철 기온은 매우 낮아집니다.

    print("\n예제 9.10 유사 검색 캐시 결과 확인")

    openai_ef = OpenAIEmbeddingFunction(
        api_key=os.environ["OPENAI_API_KEY"],
        model_name="text-embedding-ada-002"
    )

    semantic_cache = chroma_client.create_collection(name="semantic_cache",
                                                     embedding_function=openai_ef, metadata={"hnsw:space": "cosine"})

    openai_cache = OpenAICache(openai_client, semantic_cache)

    questions = ["북태평양 기단과 오호츠크해 기단이 만나 국내에 머무르는 기간은?",
                 "북태평양 기단과 오호츠크해 기단이 만나 국내에 머무르는 기간은?",
                 "북태평양 기단과 오호츠크해 기단이 만나 한반도에 머무르는 기간은?",
                 "국내에 북태평양 기단과 오호츠크해 기단이 함께 머무리는 기간은?"]
    for question in questions:
        start_time = time.time()
        response = openai_cache.generate(question)
        print(f'질문: {question}')
        print("소요 시간: {:.2f}s".format(time.time() - start_time))
        print(f'답변: {response}\n')
    # 질문: 북태평양 기단과 오호츠크해 기단이 만나 국내에 머무르는 기간은?
    # 소요 시간: 3.49s
    # 답변: 북태평양 기단과 오호츠크해 기단이 만나 국내에 머무르는 기간은 겨울철인 11월부터 3월 또는 4월까지입니다. ...

    # 질문: 북태평양 기단과 오호츠크해 기단이 만나 국내에 머무르는 기간은?
    # 소요 시간: 0.00s
    # 답변: 북태평양 기단과 오호츠크해 기단이 만나 국내에 머무르는 기간은 겨울철인 11월부터 3월 또는 4월까지입니다. ...

    # 질문: 북태평양 기단과 오호츠크해 기단이 만나 한반도에 머무르는 기간은?
    # 소요 시간: 0.13s
    # 답변: 북태평양 기단과 오호츠크해 기단이 만나 국내에 머무르는 기간은 겨울철인 11월부터 3월 또는 4월까지입니다. ...

    # 질문: 국내에 북태평양 기단과 오호츠크해 기단이 함께 머무르는 기간은?
    # 소요 시간: 0.11s
    # 답변: 북태평양 기단과 오호츠크해 기단이 만나 국내에 머무르는 기간은 겨울철인 11월부터 3월 또는 4월까지입니다. ...
