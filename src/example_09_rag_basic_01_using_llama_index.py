import os
from dotenv import load_dotenv

load_dotenv('.huggingface_env2')
print(os.environ['HF_HOME'])

from datasets import load_dataset
from llama_index.core import Document, VectorStoreIndex, get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
# from llama_index.llms.openai import OpenAI

from llama_index.llms.openai import OpenAI


class CustomOpenAI(OpenAI):
    def _prepare_chat_with_tools(self, message):
        # Add here the appropriate implementation. Below is just a mockup.
        # You should replace it with the actual implementation.
        return {"text": message, "tools": "example_tool"}

if __name__ == '__main__':
    print("#9.1절 검색 증강 생성(RAG)")
    print("예제 9.1. 데이터셋 다운로드 및 API 키 설정")
    # os.environ["OPENAI_API_KEY"] = "자신의 OpenAI API 키 입력"
    llm = CustomOpenAI(model="gpt-3.5-turbo")

    dataset = load_dataset('klue', 'mrc', split='train')
    print(dataset[0])

    print("\n예제 9.2. 실습 데이터 중 첫 100개를 뽑아 임베딩 벡터로 변환하고 저장")
    text_list = dataset[:100]['context']
    documents = [Document(text=t) for t in text_list]

    # 인덱스 만들기
    index = VectorStoreIndex.from_documents(documents)

    print("\n예제 9.3 100개의 기사 본문 데이터에서 질문과 가까운 기사 찾기")
    example_question = dataset[0]['question']
    print(example_question)  # 북태평양 기단과 오호츠크해 기단이 만나 국내에 머무르는 기간은?

    retrieval_engine = index.as_retriever(similarity_top_k=5, verbose=True)
    response = retrieval_engine.retrieve(example_question)
    print(len(response))  # 출력 결과: 5
    print(response[0].node.text)

    print("\n예제 9.4 라마인덱스를 활용해 검색 증강 생성 수행하기")
    query_engine = index.as_query_engine(similarity_top_k=1, llm=llm)
    response = query_engine.query(example_question)
    # print(response)

    print("\n예제 9.5 라마인덱스 내부에서 검색 증강 생성을 수행하는 과정")
    # 코드 출처: https://docs.llamaindex.ai/en/stable/understanding/querying/querying.html"
    # 검색을 위한 Retriever 생성
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=1,
    )

    # 검색 결과를 질문과 결합하는 synthesizer

    response_synthesizer = get_response_synthesizer(llm=llm)

    # 위의 두 요소를 결합해 쿼리 엔진 생성
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
        node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.8)],
    )

    # RAG 수행
    response = query_engine.query("북태평양 기단과 오호츠크해 기단이 만나 국내에 머무르는 기간은?")
    print(response)
    # 장마전선에서 북태평양 기단과 오호츠크해 기단이 만나 국내에 머무르는 기간은 한 달 가량입니다.
