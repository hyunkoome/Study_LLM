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
import subprocess  # 명령어를 Python 스크립트에서 실행, Jupyter Notebook에서는 !
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, PeftModel


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

    # 6.3.2 미세 조정 수행
    ## 6.9 학습 데이터 불러오기
    # from datasets import load_dataset
    save_data_path = "/home/hyunkoo/DATA/HDD8TB/GenAI/Study_LLM/data/train_data"

    df_sql = load_dataset("shangrilar/ko_text2sql", "origin")["train"]
    df_sql = df_sql.to_pandas()
    df_sql = df_sql.dropna().sample(frac=1, random_state=42)
    df_sql = df_sql.query("db_id != 1")

    for idx, row in df_sql.iterrows():
        df_sql.loc[idx, 'text'] = make_prompt(row['context'], row['question'], row['answer'])

    df_sql.to_csv(os.path.join(save_data_path, "train.csv"), index=False)

    ## 미세조정
    base_model = 'beomi/Yi-Ko-6B'
    finetuned_model = 'yi-ko-6b-text2sql'

    # 실행할 명령어를 문자열로 작성
    command = f"""
    autotrain llm \
    --train \
    --model {base_model} \
    --project-name {finetuned_model} \
    --data-path {save_data_path}/ \
    --text-column text \
    --lr 2e-4 \
    --batch-size 8 \
    --epochs 1 \
    --block-size 1024 \
    --warmup-ratio 0.1 \
    --lora-r 16 \
    --lora-alpha 32 \
    --lora-dropout 0.05 \
    --weight-decay 0.01 \
    --gradient-accumulation 8 \
    --mixed-precision fp16 \
    --use-peft \
    --quantization int4 \
    --trainer sft
    """

    # 명령어를 실행
    subprocess.run(command, shell=True)

    ## 예제 6.11. LoRA 어댑터 결합 및 허깅페이스 허브 업로드
    model_name = base_model
    device_map = {"": 0}

    # LoRA와 기초 모델 파라미터 합치기
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map=device_map,
    )
    model = PeftModel.from_pretrained(base_model, finetuned_model)
    model = model.merge_and_unload()

    # 토크나이저 설정
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 허깅페이스 허브에 모델 및 토크나이저 저장
    model.push_to_hub(finetuned_model, use_temp_dir=False)
    tokenizer.push_to_hub(finetuned_model, use_temp_dir=False)

    ## 예제 6.12. 미세 조정한 모델로 예시 데이터에 대한 SQL 생성
    model_id = "shangrilar/yi-ko-6b-text2sql"
    hf_pipe = make_inference_pipeline(model_id)

    hf_pipe(example, do_sample=False,
            return_full_text=False, max_length=1024, truncation=True)
    # SELECT COUNT(*) FROM players WHERE username LIKE '%admin%';

    ## 예제 6.13. 미세 조정한 모델 성능 측정
    # sql 생성 수행
    gen_sqls = hf_pipe(df['prompt'].tolist(), do_sample=False,
                       return_full_text=False, max_length=1024, truncation=True)
    gen_sqls = [x[0]['generated_text'] for x in gen_sqls]
    df['gen_sql'] = gen_sqls

    # 평가를 위한 requests.jsonl 생성
    ft_eval_filepath = "text2sql_evaluation_finetuned.jsonl"
    make_requests_for_gpt_evaluation(df, ft_eval_filepath)

    # GPT-4 평가 수행
    """
    !python api_request_parallel_processor.py \
      --requests_filepath requests/{ft_eval_filepath} \
      --save_filepath results/{ft_eval_filepath} \
      --request_url https://api.openai.com/v1/chat/completions \
      --max_requests_per_minute 2500 \
      --max_tokens_per_minute 100000 \
      --token_encoding_name cl100k_base \
      --max_attempts 5 \
      --logging_level 20
    """
    request_url = "https://api.openai.com/v1/chat/completions"
    api_key = os.environ["OPENAI_API_KEY"]
    max_requests_per_minute = 2500
    max_tokens_per_minute = 100000
    token_encoding_name = "cl100k_base"
    max_attempts = 5
    logging_level = 20

    asyncio.run(
        process_api_requests_from_file(
            requests_filepath=f"/home/hyunkoo/DATA/HDD8TB/GenAI/Study_LLM/data/requests/{ft_eval_filepath}",
            save_filepath=f"/home/hyunkoo/DATA/HDD8TB/GenAI/Study_LLM/data/results/{ft_eval_filepath}",
            request_url=request_url,
            api_key=api_key,
            max_requests_per_minute=float(max_requests_per_minute),
            max_tokens_per_minute=float(max_tokens_per_minute),
            token_encoding_name=token_encoding_name,
            max_attempts=int(max_attempts),
            logging_level=int(logging_level),
        )
    )

    ft_eval = change_jsonl_to_csv(f"results/{ft_eval_filepath}", "results/yi_ko_6b_eval.csv", "prompt", "resolve_yn")
    ft_eval['resolve_yn'] = ft_eval['resolve_yn'].apply(lambda x: json.loads(x)['resolve_yn'])
    num_correct_answers = ft_eval.query("resolve_yn == 'yes'").shape[0]
    print(num_correct_answers)