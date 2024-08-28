import os
from dotenv import load_dotenv

load_dotenv('.huggingface_env2')
print(os.environ['HF_HOME'])

import warnings

# # 특정 경고 무시
# warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import wandb
import datetime
from openai import OpenAI
from wandb.sdk.data_types.trace_tree import Trace

from datasets import load_dataset
import llama_index
from llama_index.core import Document, VectorStoreIndex, ServiceContext
from llama_index.llms.openai import OpenAI as llamaOpenAI
from llama_index.core import set_global_handler
from utils.common import ignore_warnings
import time

import os
import wandb
import datetime
from llama_index.core import Document, VectorStoreIndex, ServiceContext, set_global_handler
from llama_index.llms.openai import OpenAI as llamaOpenAI
from wandb.sdk.data_types.trace_tree import Trace
from datasets import load_dataset
from utils.common import ignore_warnings


def open_api_wandb_logging(wandb_project_name, model_name, temperature, query):
    wandb.login()
    wandb.init(project=wandb_project_name)

    client = OpenAI()
    system_message = "You are a helpful assistant."

    response = client.chat.completions.create(model=model_name,
                                              messages=[{"role": "system", "content": system_message},
                                                        {"role": "user", "content": query}],
                                              temperature=temperature,
                                              )
    print("query:", query)
    print("response: ", response.choices[0].message.content)

    root_span = Trace(
        name="root_span",
        kind="llm",
        status_code="success",
        status_message=None,
        metadata={"temperature": temperature,
                  "token_usage": dict(response.usage),
                  "model_name": model_name},
        inputs={"system_prompt": system_message, "query": query},
        outputs={"response": response.choices[0].message.content},
    )

    root_span.log(name="openai_trace")


def llama_index_wandb_logging(wandb_project_name, model_name, temperature, dataset_documents, query):
    wandb.login()
    wandb.init(project=wandb_project_name)

    # 로깅을 위한 설정 추가
    llm = llamaOpenAI(model=model_name, temperature=temperature)
    set_global_handler("wandb", run_args={"project": wandb_project_name})

    # 서비스 컨텍스트 설정
    service_context = ServiceContext.from_defaults(llm=llm)
    index = VectorStoreIndex.from_documents(dataset_documents, service_context=service_context)

    # Query Engine 생성 및 쿼리 수행
    query_engine = index.as_query_engine(similarity_top_k=1, verbose=True)
    response = query_engine.query(query)

    print("query:", query)  # 북태평양 기단과 오호츠크해 기단이 만나 국내에 머무르는 기간은?
    print("response:", response)

    # Trace 생성 및 로깅
    root_span = Trace(
        name="root_span",
        kind="llm",
        status_code="success",
        status_message=None,
        metadata={
            "temperature": temperature,
            "model_name": model_name
        },
        inputs={
            "query": query
        },
        outputs={
            "response": str(response)  # response 객체를 문자열로 변환하여 로깅
        },
    )

    root_span.log(name="llama_index_trace")


if __name__ == '__main__':
    ignore_warnings()

    print("예제 9.16 OpenAI API 로깅하기")
    wandb_project_name = "trace-example"
    query = "대한민국의 수도는 어디야?"
    temperature = 0.2
    model_name = "gpt-3.5-turbo"
    open_api_wandb_logging(wandb_project_name=wandb_project_name, model_name=model_name, temperature=temperature,
                           query=query)

    time.sleep(20)  # 20초 대기

    print("\n예제 9.17 라마인덱스 W&B 로깅")
    wandb_project_name = "llamaindex-example"
    llama_model_name = "gpt-3.5-turbo"
    llama_temperature = 0

    # 데이터셋 로드 및 문서 생성
    dataset = load_dataset('klue', 'mrc', split='train')
    text_list = dataset[:100]['context']
    documents = [Document(text=t) for t in text_list]

    # LlamaIndex 및 W&B 로깅 실행
    llama_index_wandb_logging(wandb_project_name=wandb_project_name, model_name=llama_model_name,
                              temperature=llama_temperature,
                              dataset_documents=documents, query=dataset[0]['question'])
