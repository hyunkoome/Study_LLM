import os
from dotenv import load_dotenv

load_dotenv('.huggingface_env2')
print(os.environ['HF_HOME'])

from openai import OpenAI
from datasets import load_dataset


def make_prompt(ddl, question, query=''):
    prompt = f"""당신은 SQL을 생성하는 SQL 봇입니다. DDL의 테이블을 활용한 Question을 해결할 수 있는 SQL 쿼리를 생성하세요.

    ### DDL:
    {ddl}

    ### Question:
    {question}

    ### SQL:
    {query}"""
    return prompt


if __name__ == '__main__':
    """
    #** 예제 8.6. 온라인 서빙을 위한 vLLM API 서버 실행 **"
    
    ## 예제 8.7. vLLM API 서버 실행하기
    $ conda activate lmmbook
    $ python -m vllm.entrypoints.openai.api_server \
        --model shangrilar/yi-ko-6b-text2sql --host 127.0.0.1 --port 8887 --max-model-len 512
    
    -----------------------------------------------------------------------------------------------------------------
    ## 예제 8.7. 백그라운드에서 vLLM API 서버 실행하기"): 가능하면 백그라운드 실행은 안 하는 것이..
    $ conda activate lmmbook
    $ nohup python -m vllm.entrypoints.openai.api_server \
        --model shangrilar/yi-ko-6b-text2sql --host 127.0.0.1 --port 8887 --max-model-len 512 & > output.log 2>&1 &
    -----------------------------------------------------------------------------------------------------------------

    ## 예제 8.8. API 서버 실행 확인
    $ conda activate lmmbook
    $ curl http://localhost:8887/v1/models
    """

    print("## 예제 8.1. 실습에 사용할 데이터셋 준비")
    dataset = load_dataset("shangrilar/ko_text2sql", "origin")['test']
    dataset = dataset.to_pandas()

    for idx, row in dataset.iterrows():
        prompt = make_prompt(row['context'], row['question'])
        dataset.loc[idx, 'prompt'] = prompt

    print("\n## 예제 8.10. OpenAI 클라이언트를 사용한 API 요청")
    openai_api_key = os.environ["OPENAI_API_KEY"]
    openai_api_base = "http://localhost:8887/v1"
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    model_id = "shangrilar/yi-ko-6b-text2sql"
    completion = client.completions.create(model=model_id,
                                           prompt=dataset.loc[0, 'prompt'], max_tokens=128)
    print("생성 결과:", completion.choices[0].text)

