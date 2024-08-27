import os
from dotenv import load_dotenv
load_dotenv('.huggingface_env2')
print(os.environ['HF_HOME'])

import torch
import time
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from vllm import LLM, SamplingParams


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
    print("## 예제 8.1. 실습에 사용할 데이터셋 준비")
    dataset = load_dataset("shangrilar/ko_text2sql", "origin")['test']
    dataset = dataset.to_pandas()

    for idx, row in dataset.iterrows():
        prompt = make_prompt(row['context'], row['question'])
        dataset.loc[idx, 'prompt'] = prompt

    print("GPU VRAM 메모리 해제합니다.")
    torch.cuda.empty_cache()

    print("\n## 예제 8.2. 모델과 토크나이저를 불러와 추론 파이프라인 준비")
    model_id = "shangrilar/yi-ko-6b-text2sql"
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", load_in_4bit=True,
                                                 bnb_4bit_compute_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    hf_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)

    print("\n## 예제 8.3. 배치 크기에 따른 추론 시간 확인")
    for batch_size in [32]:
    # for batch_size in [1, 2, 4, 8, 16, 32]:
        start_time = time.time()
        hf_pipeline(dataset['prompt'].tolist(), max_new_tokens=128, batch_size=batch_size)
        print(f'{batch_size}: {time.time() - start_time}')

    """
    ### 안내
    모델을 여러 번 GPU에 올리기 때문에 CUDA out of memory 에러가 발생할 수 있습니다. 
    그런 경우 구글 코랩의 런타임 > 세션 다시 시작 후 예제 코드를 실행해주세요.
    예제 실행에 데이터셋이 필요한 경우 예제 8.1의 코드를 실행해주세요.
    """
    print("GPU VRAM 메모리 해제합니다.")
    torch.cuda.empty_cache()

    print("\n## 예제 8.4. vLLM 모델 불러오기")
    # model_id = "shangrilar/yi-ko-6b-text2sql"
    # llm = LLM(model=model_id, dtype=torch.float16, max_model_len=1024)
    llm = LLM(model=model_id, dtype=torch.float16, max_model_len=128)

    print("\n## 예제 8.5. vLLM을 활용한 오프라인 추론 시간 측정")
    for max_num_seqs in [1, 2, 4, 8, 16, 32]:
        start_time = time.time()
        llm.llm_engine.scheduler_config.max_num_seqs = max_num_seqs
        sampling_params = SamplingParams(temperature=1, top_p=1, max_tokens=128)
        outputs = llm.generate(dataset['prompt'].tolist(), sampling_params)
        print(f'{max_num_seqs}: {time.time() - start_time}')

    print("GPU VRAM 메모리 해제합니다.")
    torch.cuda.empty_cache()


