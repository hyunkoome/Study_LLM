# 실습을 새롭게 시작하는 경우 데이터셋 다시 불러오기 실행
import os
import typing

from dotenv import load_dotenv

load_dotenv('.huggingface_env2')
print(os.environ['HF_HOME'])

from datasets import load_dataset
from huggingface_hub import login

import torch
from torch.nn.functional import softmax
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from utils.utils_huggingface import inference_results


class CustomPipeline:
    def __init__(self, model_id):
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model.eval()

    def __call__(self, texts: typing.List):
        tokenized = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        # tokenized = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True,
        #                            clean_up_tokenization_spaces=True)

        with torch.no_grad():
            outputs = self.model(**tokenized)
            logits = outputs.logits

        probabilities = softmax(logits, dim=-1)
        scores, labels = torch.max(probabilities, dim=-1)
        labels_str = [self.model.config.id2label[label_idx] for label_idx in labels.tolist()]
        infer_results = [{"label": label, "score": score.item()} for label, score in zip(labels_str, scores)]

        return [{"label": label, "score": score.item()} for label, score in zip(labels_str, scores)]


if __name__ == '__main__':
    dataset = load_dataset("klue", "ynat", split="validation")
    query_list = dataset["title"][:5]

    # login(token="본인의 허깅페이스 토큰 입력")
    login(token=os.getenv('HUGGINGFACE_TOCKEN'))

    num_epochs = 2
    # repo_id = f"본인의 아이디 입력/roberta-base-klue-ynat-classification"
    hub_model_id = f"hyunkookim/roberta-base-klue-ynat-classification-using-hg_api-epoch_{num_epochs}"

    custom_pipeline = CustomPipeline(hub_model_id)
    infer_results = custom_pipeline(texts=query_list)
    inference_results(query_list=query_list, infer_results=infer_results)

    hub_model_id = f"hyunkookim/roberta-base-klue-ynat-classification-using-pytorch-epoch_{num_epochs}"
    print(hub_model_id)

    custom_pipeline = CustomPipeline(hub_model_id)
    infer_results = custom_pipeline(texts=query_list)
    inference_results(query_list=query_list, infer_results=infer_results)
