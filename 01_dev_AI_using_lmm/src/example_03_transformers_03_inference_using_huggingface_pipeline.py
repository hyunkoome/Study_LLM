# 실습을 새롭게 시작하는 경우 데이터셋 다시 불러오기 실행
import os
import torch
from dotenv import load_dotenv

load_dotenv('.huggingface_env2')
print(os.environ['HF_HOME'])

# import torch.nn.functional as F
from datasets import load_dataset
from huggingface_hub import login
from transformers import pipeline
from utils.utils_huggingface import inference_results

if __name__ == '__main__':
    dataset = load_dataset("klue", "ynat", split="validation")
    query_list = dataset["title"][:5]

    # login(token="본인의 허깅페이스 토큰 입력")
    login(token=os.getenv('HF_TOKEN'))

    num_epochs = 2
    # repo_id = f"본인의 아이디 입력/roberta-base-klue-ynat-classification"
    hub_model_id = f"hyunkookim/roberta-base-klue-ynat-classification-using-hg_api-epoch_{num_epochs}"
    print(hub_model_id)

    # device=-1: cpu 사용
    # device=0: 1번 gpu 사용
    # device=1: 2번 gpu 사용
    model_pipeline = pipeline(task="text-classification", model=hub_model_id, device=0)
    infer_results = model_pipeline(query_list)
    inference_results(query_list, infer_results)

    hub_model_id = f"hyunkookim/roberta-base-klue-ynat-classification-using-pytorch-epoch_{num_epochs}"
    print(hub_model_id)
    model_pipeline = pipeline(task="text-classification", model=hub_model_id, device=0)
    infer_results = model_pipeline(query_list)
    inference_results(query_list, infer_results)
