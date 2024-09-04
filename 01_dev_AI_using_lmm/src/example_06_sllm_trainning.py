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
from tqdm import tqdm
import gc
from huggingface_hub import login


# from pathlib import Path


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


def make_requests_for_gpt_evaluation(df, filename, dir='./results_data/example6/requests'):
    """
    ## 예제 6.4. 평가를 위한 요청 jsonl 작성 함수
    :param df:
    :param filename:
    :param dir:
    :return:
    """

    if not Path(dir).exists():
        Path(dir).mkdir(parents=True, exist_ok=True)
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
    # jobs = [{"model": "gpt-4-turbo", "response_format": {"type": "json_object"},
    #          "messages": [{"role": "system", "content": prompt}]} for prompt in prompts]
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
            # 딕셔너리인지 확인하고, 'choices' 키가 있는지 확인
            if isinstance(json.loads(data)[1], dict) and 'choices' in json.loads(data)[1]:
                print("'choices' 키가 존재합니다.")
                prompts.append(json.loads(data)[0]['messages'][0]['content'])
                responses.append(json.loads(data)[1]['choices'][0]['message']['content'])
            else:
                print("'choices' 키가 존재하지 않습니다.")
                # prompts.append(json.loads(data)[0]['messages'][0]['content'])
                # responses.append(json.loads(data)[1]['choices'][0]['message']['content'])

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
    save_root_dir = "./results_data/example6"
    Path(f"{save_root_dir}").mkdir(parents=True, exist_ok=True)
    Path(f"{save_root_dir}/ko_text2sql").mkdir(parents=True, exist_ok=True)
    Path(f"{save_root_dir}/requests").mkdir(parents=True, exist_ok=True)
    Path(f"{save_root_dir}/openai").mkdir(parents=True, exist_ok=True)
    Path(f"{save_root_dir}/fine_tune_train_data").mkdir(parents=True, exist_ok=True)
    save_root_dir = str(Path(save_root_dir).resolve())
    print('save_root_dir: ', save_root_dir)
    # exit()

    torch.cuda.empty_cache()

    print('# 6.3.1 기초 모델 평가 하기')
    print('## 예제 6.7. 기초 모델로 생성하기')
    model_id = 'beomi/Yi-Ko-6B'
    hf_pipe = make_inference_pipeline(model_id=model_id)

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

    gen_sqls = hf_pipe(example, do_sample=False,
                       return_full_text=False, max_length=512, truncation=True)
    print(gen_sqls)

    print('## 예제 6.8. 기초 모델 성능 측정')

    print('# 데이터셋 불러오기')
    df = load_dataset("shangrilar/ko_text2sql", "origin")['test']
    df = df.to_pandas()
    for idx, row in tqdm(df.iterrows()):
        prompt = make_prompt(row['context'], row['question'])
        df.loc[idx, 'prompt'] = prompt

    print(df['prompt'].tolist()[0])

    print('# sql 생성')
    # gen_sqls = hf_pipe(df['prompt'].tolist(), do_sample=False,
    #                    return_full_text=False, max_length=512, truncation=True)
    # gen_sqls = [x[0]['generated_text'] for x in gen_sqls]
    # df['gen_sql'] = gen_sqls
    total_elements = len(df)
    for idx, row in df.iterrows():
        gen_sqls = hf_pipe(row['prompt'], do_sample=False,
                           return_full_text=False, max_length=512, truncation=True)
        gen_sqls = gen_sqls[0]['generated_text']
        df.loc[idx, 'gen_sql'] = gen_sqls
        print(f'... done: {idx}/{total_elements} ')

    df.to_csv(f"{save_root_dir}/ko_text2sql/df_test.csv", index=False)
    # df = pd.read_csv(f"{save_root_dir}/ko_text2sql/df_test.csv")

    print('# 평가를 위한 requests.jsonl 생성')
    eval_filepath = "text2sql_evaluation.jsonl"
    make_requests_for_gpt_evaluation(df, f"{save_root_dir}/requests/{eval_filepath}")

    print('# 기초 모델 성능 측정')
    request_url = "https://api.openai.com/v1/chat/completions"
    api_key = os.environ["OPENAI_API_KEY"]
    max_requests_per_minute = 200  # 300
    max_tokens_per_minute = 100000
    token_encoding_name = "cl100k_base"
    max_attempts = 5
    logging_level = 20

    asyncio.run(
        process_api_requests_from_file(
            requests_filepath=f"{save_root_dir}/requests/{eval_filepath}",
            save_filepath=f"{save_root_dir}/openai/{eval_filepath}",
            request_url=request_url,
            api_key=api_key,
            max_requests_per_minute=float(max_requests_per_minute),
            max_tokens_per_minute=float(max_tokens_per_minute),
            token_encoding_name=token_encoding_name,
            max_attempts=int(max_attempts),
            logging_level=int(logging_level),
        )
    )

    print('# 6.3.2 미세 조정 수행')
    print('## 6.9 학습 데이터 불러오기')
    df_sql = load_dataset("shangrilar/ko_text2sql", "origin")["train"]
    df_sql = df_sql.to_pandas()
    df_sql = df_sql.dropna().sample(frac=1, random_state=42)
    df_sql = df_sql.query("db_id != 1")

    for idx, row in tqdm(df_sql.iterrows()):
        df_sql.loc[idx, 'text'] = make_prompt(row['context'], row['question'], row['answer'])
    df_sql.to_csv(f"{save_root_dir}/ko_text2sql/df_sql_train.csv", index=False)
    df_sql.to_csv(f"{save_root_dir}/fine_tune_train_data/train.csv", index=False)

    print('## 미세조정')
    base_model_name = 'beomi/Yi-Ko-6B'
    finetuned_model_name = 'yi-ko-6b-text2sql'

    """
    1. AutoTrain LLM Fine-Tuning
        - 모델 구조: 
            AutoTrain LLM을 사용하여 fine-tune한 모델은 전체 모델을 직접 미세 조정(fine-tuning)합니다.
            이는 모델의 모든 가중치가 업데이트된다는 것을 의미합니다.
        - 파라미터 업데이트: 
            모델의 모든 파라미터가 학습 도중 업데이트됩니다. 
            따라서 모델이 원래 가지고 있던 파라미터들이 새로운 데이터셋에 맞춰 최적화됩니다.
        - 결과: 
            결과적으로, fine-tuning 후에는 하나의 완전한 모델이 생성되며, 모든 가중치가 fine-tuning을 반영합니다. 
            이는 모델의 성능을 극대화할 수 있지만, 특히 대규모 모델의 경우 계산 비용이 많이 들고 저장 공간이 많이 필요할 수 있습니다.
    """

    # 실행할 명령어를 문자열로 작성
    # --batch-size 8 -> 32
    print('# autotrain llm 명령어를 실행')
    autotrain_llm_command = f"""
            autotrain llm \
            --train \
            --model {base_model_name} \
            --project-name {finetuned_model_name}-full-fine-tuning \
            --data-path {save_root_dir}/fine_tune_train_data \
            --text-column text \
            --lr 2e-4 \
            --batch-size 32 \
            --epochs 1 \
            --block-size 1024 \
            --warmup-ratio 0.1 \
            --lora-r 16 \
            --lora-alpha 32 \
            --lora-dropout 0.05 \
            --weight-decay 0.01 \
            --gradient-accumulation 8 \
            --mixed-precision fp16 \
            --peft \
            --quantization int4 \
            --trainer sft \
            --username {os.getenv('HF_USERNAME')} \
            --token {os.getenv('HF_TOKEN')} \
            --push_to_hub
            """

    result = subprocess.run(autotrain_llm_command, shell=True)
    if result.returncode != 0:
        print("autotrain llm 명령어 실행 중 오류가 발생했습니다. 함수 종료.")
        exit()

    print('# mv 명령어를 실행')
    Path(f"{save_root_dir}/new_models").mkdir(parents=True, exist_ok=True)
    mv_command = f"mv ./{finetuned_model_name}-full-fine-tuning {save_root_dir}/new_models"

    result = subprocess.run(mv_command, shell=True)
    if result.returncode != 0:
        print("mv 명령어 실행 중 오류가 발생했습니다. 함수 종료.")
        exit()

    print("명령어가 성공적으로 실행되었습니다.")
    print("GPU VRAM 메모리 해제합니다.")
    torch.cuda.empty_cache()

    print('## 예제 6.11. LoRA 어댑터 결합 및 허깅페이스 허브 업로드')
    # login(token="본인의 허깅페이스 토큰 입력")
    print('HF 로그인')
    login(token=os.getenv('HF_TOKEN'))

    device_map = {"": 0}

    print('# LoRA와 기초 모델 파라미터 합치기')
    """
    2. LoRA + PEFT로 Fine-Tuning 및 Merge
    - 모델 구조:
        PEFT의 LoRA 방법은 전체 모델이 아닌 일부 가중치만을 업데이트하는 방법입니다.
        일반적으로, 대규모 모델에서 미세 조정해야 하는 파라미터의 수를 줄여 학습 비용을 절감하면서도 성능 저하를 최소화하는 방식입니다.
    - 파라미터 업데이트:
        LoRA는 모델의 일부 파라미터를 저차원(low-rank) 방식으로 업데이트합니다.
        원본 모델은 유지한 채로 추가적인 저차원 가중치를 학습하여 더 작은 모델로 학습하는 방식입니다.
    - Merge와 Unload:
        model.merge_and_unload() 메서드는 LoRA를 통해 학습된 저차원 가중치를 원본 모델에 병합(merge)한 뒤,
            불필요한 추가 메타데이터와 가중치를 메모리에서 해제(unload)하는 과정입니다.
        이를 통해 최종적으로 하나의 통합된 모델이 생성됩니다.
        하지만 이 경우에도 모델의 대부분의 가중치는 원본 모델의 것이고, 미세 조정된 저차원 가중치만 합쳐진 상태입니다.
    """
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map=device_map,
    )

    # merged_model = PeftModel.from_pretrained(base_model, model_id=f"{finetuned_model_name}-full-fine-tuning")
    merged_model = PeftModel.from_pretrained(base_model,
                                             model_id=f"{save_root_dir}/new_models/{finetuned_model_name}-full-fine-tuning")
    merged_model = merged_model.merge_and_unload()
    merged_model_id = f"hyunkookim/{finetuned_model_name}-finetune-lora-peft"

    output_dir = f"{save_root_dir}/new_models/{finetuned_model_name}-finetune-lora-peft"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print('# 토크나이저 설정')
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print('# 허깅페이스 허브에 모델 및 토크나이저 저장')
    merged_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    merged_model.push_to_hub(merged_model_id, working_dir=output_dir)
    tokenizer.push_to_hub(merged_model_id, working_dir=output_dir)

    """
    * 차이점 요약
        - Fine-Tuning 방식: AutoTrain LLM은 전체 모델을 미세 조정하는 반면, 
        - LoRA + PEFT: 파라미터의 일부만 저차원 방식으로 미세 조정합니다.
    * 결과 모델:
        - AutoTrain LLM: 모든 가중치가 새로 학습된 하나의 완전한 모델을 생성합니다.
        - LoRA + PEFT에서 merge_and_unload 후의 모델: 저차원 가중치를 원본 모델에 병합하여 생성된 모델이지만, 
                                                    원본 모델의 가중치 대부분이 그대로 유지됩니다.
    * 효율성: 
        - LoRA를 사용한 방법은 메모리와 계산 비용 측면에서 더 효율적입니다. 
        - 하지만 성능이 약간 손실될 수 있는 반면, 
        - AutoTrain LLM은 더 높은 성능을 낼 수 있지만, 리소스 요구 사항이 큽니다.
    * 결론적으로, 두 방식은 각각의 용도와 상황에 따라 선택됩니다. 
        - 더 많은 리소스가 필요하지만 최고의 성능을 목표로 할 때는 AutoTrain LLM 방식이 적합하고, 
        - 효율성을 중시할 때는 LoRA + PEFT 방식이 적합합니다.
    """
    print('## 예제 6.12. 미세 조정한 모델(AutoTrain LLM)로 예시 데이터에 대한 SQL 생성')
    model_id = f"hyunkookim/{finetuned_model_name}-full-fine-tuning"
    hf_pipe = make_inference_pipeline(model_id=model_id)

    gen_sqls = hf_pipe(example, do_sample=False,
                       return_full_text=False, max_length=1024, truncation=True)
    print(gen_sqls)
    # SELECT COUNT(*) FROM players WHERE username LIKE '%admin%';

    print('## 예제 6.13. 미세 조정한 모델(AutoTrain LLM) 성능 측정')
    # sql 생성 수행
    # gen_sqls = hf_pipe(df['prompt'].tolist(), do_sample=False,
    #                    return_full_text=False, max_length=1024, truncation=True)
    # gen_sqls = [x[0]['generated_text'] for x in gen_sqls]
    # df['gen_sql'] = gen_sqls
    total_elements = len(df)
    for idx, row in df.iterrows():
        gen_sqls = hf_pipe(row['prompt'], do_sample=False,
                           return_full_text=False, max_length=1024, truncation=True)
        gen_sqls = gen_sqls[0]['generated_text']
        df.loc[idx, 'gen_sql'] = gen_sqls
        print(f'... done: {idx}/{total_elements} ')
    df.to_csv(f"{save_root_dir}/ko_text2sql/df_text2sql_evaluation_{finetuned_model_name}-full-fine-tuning.csv", index=False)

    print('# 평가를 위한 requests.jsonl 생성')
    ft_eval_filepath = f"text2sql_evaluation_{finetuned_model_name}-full-fine-tuning.jsonl"
    make_requests_for_gpt_evaluation(df,
                                     f"{save_root_dir}/requests/{ft_eval_filepath}")

    print('# GPT-4 평가 수행')
    """
    !python api_request_parallel_processor.py \
      --requests_filepath requests/{ft_eval_filepath} \
      --save_filepath openai/{ft_eval_filepath} \
      --request_url https://api.openai.com/v1/chat/completions \
      --max_requests_per_minute 2500 \
      --max_tokens_per_minute 100000 \
      --token_encoding_name cl100k_base \
      --max_attempts 5 \
      --logging_level 20
    """
    request_url = "https://api.openai.com/v1/chat/completions"
    api_key = os.environ["OPENAI_API_KEY"]
    max_requests_per_minute = 200  # 2500
    max_tokens_per_minute = 100000
    token_encoding_name = "cl100k_base"
    max_attempts = 5
    logging_level = 20

    asyncio.run(
        process_api_requests_from_file(
            requests_filepath=f"{save_root_dir}/requests/{ft_eval_filepath}",
            save_filepath=f"{save_root_dir}/openai/{ft_eval_filepath}",
            request_url=request_url,
            api_key=api_key,
            max_requests_per_minute=float(max_requests_per_minute),
            max_tokens_per_minute=float(max_tokens_per_minute),
            token_encoding_name=token_encoding_name,
            max_attempts=int(max_attempts),
            logging_level=int(logging_level),
        )
    )

    ft_eval = change_jsonl_to_csv(f"{save_root_dir}/openai/{ft_eval_filepath}",
                                  f"{save_root_dir}/openai/yi_ko_6b_eval_{ft_eval_filepath}.csv",
                                  "prompt",
                                  "resolve_yn")
    ft_eval['resolve_yn'] = ft_eval['resolve_yn'].apply(lambda x: json.loads(x)['resolve_yn'])
    num_correct_answers = ft_eval.query("resolve_yn == 'yes'").shape[0]
    print(num_correct_answers)

    print('## 예제 6.14. 미세 조정한 모델(LoRA + PEFT)로 예시 데이터에 대한 SQL 생성')
    model_id = f"hyunkookim/{finetuned_model_name}-finetune-lora-peft"
    hf_pipe = make_inference_pipeline(model_id=model_id)

    gen_sqls = hf_pipe(example, do_sample=False,
                       return_full_text=False, max_length=1024, truncation=True)
    print(gen_sqls)
    # SELECT COUNT(*) FROM players WHERE username LIKE '%admin%';

    print('## 예제 6.13. 미세 조정한 모델(LoRA + PEFT) 성능 측정')
    # sql 생성 수행
    # gen_sqls = hf_pipe(df['prompt'].tolist(), do_sample=False,
    #                    return_full_text=False, max_length=1024, truncation=True)
    # gen_sqls = [x[0]['generated_text'] for x in gen_sqls]
    # df['gen_sql'] = gen_sqls
    total_elements = len(df)
    for idx, row in df.iterrows():
        gen_sqls = hf_pipe(row['prompt'], do_sample=False,
                           return_full_text=False, max_length=1024, truncation=True)
        gen_sqls = gen_sqls[0]['generated_text']
        df.loc[idx, 'gen_sql'] = gen_sqls
        print(f'... done: {idx}/{total_elements} ')
    df.to_csv(f"{save_root_dir}/ko_text2sql/df_text2sql_evaluation_{finetuned_model_name}-fullfine-lora-peft.csv",
              index=False)

    print('# 평가를 위한 requests.jsonl 생성')
    ft_eval_filepath = f"text2sql_evaluation_{finetuned_model_name}-fullfine-lora-peft.jsonl"
    make_requests_for_gpt_evaluation(df, f"{save_root_dir}/requests/{ft_eval_filepath}")

    print('# GPT-4 평가 수행')
    """
    !python api_request_parallel_processor.py \
      --requests_filepath requests/{ft_eval_filepath} \
      --save_filepath openai/{ft_eval_filepath} \
      --request_url https://api.openai.com/v1/chat/completions \
      --max_requests_per_minute 2500 \
      --max_tokens_per_minute 100000 \
      --token_encoding_name cl100k_base \
      --max_attempts 5 \
      --logging_level 20
    """
    request_url = "https://api.openai.com/v1/chat/completions"
    api_key = os.environ["OPENAI_API_KEY"]
    max_requests_per_minute = 200  # 2500
    max_tokens_per_minute = 100000
    token_encoding_name = "cl100k_base"
    max_attempts = 5
    logging_level = 20

    asyncio.run(
        process_api_requests_from_file(
            requests_filepath=f"{save_root_dir}/requests/{ft_eval_filepath}",
            save_filepath=f"{save_root_dir}/openai/{ft_eval_filepath}",
            request_url=request_url,
            api_key=api_key,
            max_requests_per_minute=float(max_requests_per_minute),
            max_tokens_per_minute=float(max_tokens_per_minute),
            token_encoding_name=token_encoding_name,
            max_attempts=int(max_attempts),
            logging_level=int(logging_level),
        )
    )

    ft_eval = change_jsonl_to_csv(f"{save_root_dir}/openai/{ft_eval_filepath}",
                                  f"{save_root_dir}/openai/yi_ko_6b_eval_{ft_eval_filepath}.csv",
                                  "prompt",
                                  "resolve_yn")
    ft_eval['resolve_yn'] = ft_eval['resolve_yn'].apply(lambda x: json.loads(x)['resolve_yn'])
    num_correct_answers = ft_eval.query("resolve_yn == 'yes'").shape[0]
    print(num_correct_answers)
