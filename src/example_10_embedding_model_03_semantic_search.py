import os
from dotenv import load_dotenv

load_dotenv('.huggingface_env2')
print(os.environ['HF_HOME'])

from utils.common import ignore_warnings

from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import faiss

from llama_index.core import Document, VectorStoreIndex, ServiceContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


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

    print("\n## 예제 10.11 의미 검색의 장점: 키워드가 동일하지 않아도 의미가 유사하면 찾을 수 있다.")
    query = "이번 연도에는 언제 비가 많이 올까?"
    query_embedding = sentence_model.encode([query])
    print(query)
    distances, indices = index.search(query_embedding, 3) # 쿼리 문장의 임베딩과 가장 가까운 3개의 문서를 반환 받겠다는 의미

    for idx in indices[0]:
        print(klue_mrc_dataset['context'][idx][:50])
    # 출력 결과
    # 올여름 장마가 17일 제주도에서 시작됐다. 서울 등 중부지방은 예년보다 사나흘 정도 늦은   (정답)
    # 연구 결과에 따르면, 오리너구리의 눈은 대부분의 포유류보다는 어류인 칠성장어나 먹장어, 그 (오답)
    # 연구 결과에 따르면, 오리너구리의 눈은 대부분의 포유류보다는 어류인 칠성장어나 먹장어, 그 (오답)

    print("\n## 예제 10.12 의미 검색의 한계: 관련성이 떻어지는 검색 결과가 나오기도 한다.!")
    query = klue_mrc_dataset[3]['question']  # (4번째 문장) 로버트 헨리 딕이 1946년에 매사추세츠 연구소에서 개발한 것은 무엇인가?
    print(query)
    query_embedding = sentence_model.encode([query])
    distances, indices = index.search(query_embedding, 3)

    for idx in indices[0]:
        print(klue_mrc_dataset['context'][idx][:50])
    # 출력 결과
    # 태평양 전쟁 중 뉴기니 방면에서 진공 작전을 실시해 온 더글러스 맥아더 장군을 사령관으로 (오답)
    # 태평양 전쟁 중 뉴기니 방면에서 진공 작전을 실시해 온 더글러스 맥아더 장군을 사령관으로 (오답)
    # 미국 세인트루이스에서 태어났고, 프린스턴 대학교에서 학사 학위를 마치고 1939년에 로체스 (정답)

    print("\n## 예제 10.13 라마인덱스에서 Sentence-Transformers 임베딩 모델 활용")
    embed_model = HuggingFaceEmbedding(model_name=sentence_model_name)
    service_context = ServiceContext.from_defaults(embed_model=embed_model, llm=None)
    # 로컬 모델 활용하기
    # service_context = ServiceContext.from_defaults(embed_model="local")

    text_list = klue_mrc_dataset[:100]['context']
    documents = [Document(text=t) for t in text_list]

    query = klue_mrc_dataset[0]['question']

    index_llama = VectorStoreIndex.from_documents(
        documents,
        service_context=service_context,
    )
    # Query Engine 생성 및 쿼리 수행
    query_engine = index_llama.as_query_engine(similarity_top_k=1, verbose=True)
    response = query_engine.query(query)
    #
    print("query:", query)  # 북태평양 기단과 오호츠크해 기단이 만나 국내에 머무르는 기간은?
    print("response:", response)