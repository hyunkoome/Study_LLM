import os
from dotenv import load_dotenv

load_dotenv('.huggingface_env2')
print(os.environ['HF_HOME'])

from utils.common import ignore_warnings

from datasets import load_dataset
from llama_index.core import Document, VectorStoreIndex, get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor

from pinecone import Pinecone, ServerlessSpec
# 라마인덱스에 파인콘 인덱스 연결
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import StorageContext


if __name__ == '__main__':
    ignore_warnings()

    print("#9.1절 검색 증강 생성(RAG)")
    print("예제 9.1. 데이터셋 다운로드 및 API 키 설정")
    # os.environ["OPENAI_API_KEY"] = "자신의 OpenAI API 키 입력"
    dataset = load_dataset('klue', 'mrc', split='train')
    print(dataset[0])

    print("\n예제 9.2. 실습 데이터 중 첫 100개를 뽑아 임베딩 벡터로 변환하고 저장")
    text_list = dataset[:100]['context']
    documents = [Document(text=t) for t in text_list]

    # 파인콘 기본 설정
    pc = Pinecone(api_key=os.environ['PINECONE_API_KEY'])
    pc.create_index(
        "quickstart", dimension=1536, metric="euclidean", spec=ServerlessSpec("aws", "us-east-1")
    )
    pinecone_index = pc.Index("quickstart")

    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
