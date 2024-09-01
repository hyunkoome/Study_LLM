import os
from dotenv import load_dotenv

load_dotenv('.huggingface_env2')
print(os.environ['HF_HOME'])

from utils.common import ignore_warnings

from pinecone import Pinecone, ServerlessSpec
from datasets import load_dataset
from sentence_transformers import SentenceTransformer


if __name__ == '__main__':
    ignore_warnings()

    print("## 예제 12.7 파인콘 계정 연결 및 인덱스 생성")
    pc = Pinecone(api_key=os.environ['PINECONE_API_KEY'])

    pc.create_index("llm-book", spec=ServerlessSpec("aws", "us-east-1"), dimension=768)
    """
    pc.create_index(
        name="quickstart",
        dimension=2, # Replace with your model dimensions
        metric="cosine", # Replace with your model metric
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        ) 
    )
    """
    index = pc.Index('llm-book')

    print("\n## 예제 12.8 임베딩 생성")
    # 임베딩 모델 불러오기
    sentence_model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')
    # 데이터셋 불러오기
    klue_dp_train = load_dataset('klue', 'dp', split='train[:100]')

    embeddings = sentence_model.encode(klue_dp_train['sentence'])

    print("\n## 예제 12.9 파인콘 입력을 위한 데이터 형태 변경")
    # 파이썬 기본 데이터 타입으로 변경
    embeddings = embeddings.tolist()
    # {"id": 문서 ID(str), "values": 벡터 임베딩(List[float]), "metadata": 메타 데이터(dict) ) 형태로 데이터 준비
    insert_data = []
    for idx, (embedding, text) in enumerate(zip(embeddings, klue_dp_train['sentence'])):
        insert_data.append({"id": str(idx), "values": embedding, "metadata": {'text': text}})

    print("\n## 예제 12.10 임베딩 데이터를 인덱스에 저장")
    upsert_response = index.upsert(vectors = insert_data, namespace='llm-book-sub')

    print("\n## 예제 12.11 인덱스 검색하기")
    query_response = index.query(
        namespace='llm-book-sub',  # 검색할 네임스페이스
        top_k=10,  # 몇 개의 결과를 반환할지
        include_values=True,  # 벡터 임베딩 반환 여부
        include_metadata=True,  # 메타 데이터 반환 여부
        vector=embeddings[0]  # 검색할 벡터 임베딩
    )
    print(query_response)

    # print("\n## 예제 12.12 파인콘에서 문서 수정 및 삭제")
    # new_text = '변경할 새로운 텍스트'
    # new_embedding = sentence_model.encode(new_text).tolist()
    # # 업데이트
    # update_response = index.update(
    #     id='기존_문서_id',
    #     values=new_embedding,
    #     set_metadata={'text': new_text},
    #     namespace='llm-book-sub'
    # )
    #
    # # 삭제
    # delete_response = index.delete(ids=['기존_문서_id'], namespace='llm-book-sub')
