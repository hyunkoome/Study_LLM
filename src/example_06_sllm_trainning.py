import os
import typing
from dotenv import load_dotenv

load_dotenv('.huggingface_env2')
print(os.environ['HF_HOME'])

import torch
import json
import pandas as pd
from pathlib import Path
import asyncio  # for running API calls concurrently
from utils.api_request_parallel_processor import process_api_requests_from_file
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from transformers import BitsAndBytesConfig


def make_prompt(ddl, question, query=''):
    """
    ## 예제 6.2. SQL 프롬프트

    :param ddl:
    :param question:
    :param query:
    :return:
    """
    prompt = \
        f"""당신은 SQL을 생성하는 SQL 봇입니다. DDL의 테이블을 활용한 Question을 해결할 수 있는 SQL 쿼리를 생성하세요.
        ### DDL:
        {ddl}
        
        ### Question:
        {question}
        
        ### SQL:
        {query}"""
    return prompt


def make_requests_for_gpt_evaluation(df, filename, dir='requests'):
    """
    ## 예제 6.4. 평가를 위한 요청 jsonl 작성 함수
    :param df:
    :param filename:
    :param dir:
    :return:
    """

    if not Path(dir).exists():
        Path(dir).mkdir(parents=True)
    prompts = []
    for idx, row in df.iterrows():
        prompts.append(
            """Based on below DDL and Question, evaluate gen_sql can resolve Question. If gen_sql and gt_sql do equal job, return "yes" else return "no". Output JSON Format: {"resolve_yn": ""}"""
            + f"""
            
            DDL: {row['context']}
            Question: {row['question']}
            gt_sql: {row['answer']}
            gen_sql: {row['gen_sql']}"""
        )

    jobs = [{"model": "gpt-4-turbo-preview", "response_format": {"type": "json_object"},
             "messages": [{"role": "system", "content": prompt}]} for prompt in prompts]
    with open(Path(dir, filename), "w") as f:
        for job in jobs:
            json_string = json.dumps(job)
            f.write(json_string + "\n")


def change_jsonl_to_csv(input_file, output_file, prompt_column="prompt", response_column="response"):
    """
    예제 6.6. 결과 jsonl 파일을 csv로 변환하는 함수

    :param input_file:
    :param output_file:
    :param prompt_column:
    :param response_column:
    :return:
    """

    prompts = []
    responses = []
    with open(input_file, 'r') as json_file:
        for data in json_file:
            prompts.append(json.loads(data)[0]['messages'][0]['content'])
            responses.append(json.loads(data)[1]['choices'][0]['message']['content'])

    df = pd.DataFrame({prompt_column: prompts, response_column: responses})
    df.to_csv(output_file, index=False)
    return df


def make_inference_pipeline(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,  # 또는 load_in_8bit=True로 설정
        bnb_4bit_compute_dtype=torch.float16,
    )
    # model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", load_in_4bit=True,
    #                                              bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", quantization_config=quantization_config)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return pipe


if __name__ == '__main__':
    # 6.3.1 기초 모델 평가 하기
    ## 예제 6.7. 기초 모델로 생성하기
    model_id = 'beomi/Yi-Ko-6B'
    hf_pipe = make_inference_pipeline(model_id)

    example = """당신은 SQL을 생성하는 SQL 봇입니다. DDL의 테이블을 활용한 Question을 해결할 수 있는 SQL 쿼리를 생성하세요.
    
    ### DDL:
    CREATE TABLE players (
      player_id INT PRIMARY KEY AUTO_INCREMENT,
      username VARCHAR(255) UNIQUE NOT NULL,
      email VARCHAR(255) UNIQUE NOT NULL,
      password_hash VARCHAR(255) NOT NULL,
      date_joined DATETIME NOT NULL,
      last_login DATETIME
    );

    ### Question:
    사용자 이름에 'admin'이 포함되어 있는 계정의 수를 알려주세요.

    ### SQL:
    """

    hf_pipe(example, do_sample=False,
            return_full_text=False, max_length=512, truncation=True)

    ## 예제 6.8. 기초 모델 성능 측정
    # from datasets import load_dataset

    # 데이터셋 불러오기
    df = load_dataset("shangrilar/ko_text2sql", "origin")['test']
    df = df.to_pandas()
    for idx, row in df.iterrows():
        prompt = make_prompt(row['context'], row['question'])
        df.loc[idx, 'prompt'] = prompt
    # sql 생성
    gen_sqls = hf_pipe(df['prompt'].tolist(), do_sample=False,
                       return_full_text=False, max_length=512, truncation=True)
    gen_sqls = [x[0]['generated_text'] for x in gen_sqls]
    df['gen_sql'] = gen_sqls

    # 평가를 위한 requests.jsonl 생성
    eval_filepath = "text2sql_evaluation.jsonl"
    make_requests_for_gpt_evaluation(df, eval_filepath)

    request_url = "https://api.openai.com/v1/chat/completions"
    api_key = os.environ["OPENAI_API_KEY"]
    max_requests_per_minute = 300
    max_tokens_per_minute = 100000
    token_encoding_name = "cl100k_base"
    max_attempts = 5
    logging_level = 20

    asyncio.run(
        process_api_requests_from_file(
            requests_filepath=f"/home/hyunkoo/DATA/HDD8TB/GenAI/Study_LLM/data/requests/{eval_filepath}",
            save_filepath=f"/home/hyunkoo/DATA/HDD8TB/GenAI/Study_LLM/data/results/{eval_filepath}",
            request_url=request_url,
            api_key=api_key,
            max_requests_per_minute=float(max_requests_per_minute),
            max_tokens_per_minute=float(max_tokens_per_minute),
            token_encoding_name=token_encoding_name,
            max_attempts=int(max_attempts),
            logging_level=int(logging_level),
        )
    )

    # ## 예제 6.5. 비동기 요청 명령
    # """
    # import os
    # # os.environ["OPENAI_API_KEY"] = "자신의 OpenAI API 키 입력"
    #
    # python api_request_parallel_processor.py \
    #   --requests_filepath {요청 파일 경로} \
    #   --save_filepath {생성할 결과 파일 경로} \
    #   --request_url https://api.openai.com/v1/chat/completions \
    #   --max_requests_per_minute 300 \
    #   --max_tokens_per_minute 100000 \
    #   --token_encoding_name cl100k_base \
    #   --max_attempts 5 \
    #   --logging_level 20
    # """
    # requests_filepath = Path("../../data/requests_filepath").absolute()
    # print(requests_filepath)
    # save_filepath = Path("../../data/save_filepath").absolute()
    #
    # exit()
    # request_url = "https://api.openai.com/v1/chat/completions"
    # api_key = os.environ["OPENAI_API_KEY"]
    # max_requests_per_minute = 300
    # max_tokens_per_minute = 100000
    # token_encoding_name = "cl100k_base"
    # max_attempts = 5
    # logging_level = 20
    #
    # # asyncio.run(
    # #     process_api_requests_from_file(
    # #         requests_filepath=args.requests_filepath,
    # #         save_filepath=args.save_filepath,
    # #         request_url=args.request_url,
    # #         api_key=args.api_key,
    # #         max_requests_per_minute=float(args.max_requests_per_minute),
    # #         max_tokens_per_minute=float(args.max_tokens_per_minute),
    # #         token_encoding_name=args.token_encoding_name,
    # #         max_attempts=int(args.max_attempts),
    # #         logging_level=int(args.logging_level),
    # #     )
    # # )
    # asyncio.run(
    #     process_api_requests_from_file(
    #         requests_filepath=requests_filepath,
    #         save_filepath=save_filepath,
    #         request_url=request_url,
    #         api_key=api_key,
    #         max_requests_per_minute=float(max_requests_per_minute),
    #         max_tokens_per_minute=float(max_tokens_per_minute),
    #         token_encoding_name=token_encoding_name,
    #         max_attempts=int(max_attempts),
    #         logging_level=int(logging_level),
    #     )
    # )
