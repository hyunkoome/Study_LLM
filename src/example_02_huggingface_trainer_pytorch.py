import os
import torch
from dotenv import load_dotenv

load_dotenv('.huggingface_env2')
print(os.environ['HF_HOME'])

from huggingface_hub import login
from datasets import load_dataset
# import torch
import numpy as np
from transformers import (
    Trainer,
    TrainingArguments,
    AutoModelForSequenceClassification,
    AutoTokenizer
)

import torch
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from transformers import AdamW


def make_str_label(batch):
    batch['label_str'] = klue_tc_label.int2str(batch['label'])
    return batch


def tokenize_function(examples):
    return tokenizer(examples["title"], padding="max_length", truncation=True)  # , clean_up_tokenization_spaces=True)

# def compute_metrics(eval_pred):
#     logits, labels = eval_pred
#     predictions = np.argmax(logits, axis=-1)
#     return {"accuracy": (predictions == labels).mean()}

# def compute_metrics(eval_pred):
#     if isinstance(eval_pred, torch.Tensor):
#         print('call tensor')
#         return compute_metrics_with_only_tensor(eval_pred)
#
#     print('call not tensor')
#     compute_metrics_with_only_numpy(eval_pred)


def compute_metrics(eval_pred):
    logits, labels = eval_pred

    # logits와 labels가 텐서인 경우 NumPy로 변환
    if isinstance(logits, torch.Tensor):
        logits = logits.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    # 예측값 계산
    predictions = np.argmax(logits, axis=-1)

    # 정확도 계산 및 반환
    accuracy = (predictions == labels).mean()
    return {"accuracy": accuracy}

## 예제 3.25 Trainer를 사용하지 않는 학습: (2) 학습을 위한 데이터 준비
def make_dataloader(dataset, batch_size, shuffle=True):
    dataset = dataset.map(tokenize_function, batched=True).with_format("torch")  # 데이터셋에 토큰화 수행
    dataset = dataset.rename_column("label", "labels")  # 컬럼 이름 변경
    dataset = dataset.remove_columns(column_names=['title'])  # 불필요한 컬럼 제거
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

## 예제 3.26. Trainer를 사용하지 않는 학습: (3) 학습을 위한 함수 정의
def train_epoch(model, data_loader, optimizer):
    model.train()
    total_loss = 0
    for batch in tqdm(data_loader):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device) # 모델에 입력할 토큰 아이디
        attention_mask = batch['attention_mask'].to(device) # 모델에 입력할 어텐션 마스크
        labels = batch['labels'].to(device) # 모델에 입력할 레이블
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels) # 모델 계산
        loss = outputs.loss # 손실
        loss.backward() # 역전파
        optimizer.step() # 모델 업데이트
        total_loss += loss.item()
    avg_loss = total_loss / len(data_loader)
    return avg_loss

## 예제 3.27. Trainer를 사용하지 않는 학습: (4) 평가를 위한 함수 정의
def evaluate(model, data_loader):
    model.eval()
    total_loss = 0
    predictions = []
    true_labels = []
    with torch.no_grad():
        for batch in tqdm(data_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            logits = outputs.logits
            loss = outputs.loss
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=-1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    avg_loss = total_loss / len(data_loader)
    accuracy = np.mean(np.array(predictions) == np.array(true_labels))
    return avg_loss, accuracy



if __name__ == '__main__':
    # ## 예제 3.17. 모델 학습에 사용할 연합뉴스 데이터셋 다운로드
    klue_tc_train = load_dataset('klue', 'ynat', split='train')
    klue_tc_eval = load_dataset('klue', 'ynat', split='validation')
    # klue_tc_train

    # klue_tc_train[0]
    # klue_tc_train.features['label'].names
    # ['IT과학', '경제', '사회', '생활문화', '세계', '스포츠', '정치']

    ## 예제 3.18. 실습에 사용하지 않는 불필요한 컬럼 제거
    klue_tc_train = klue_tc_train.remove_columns(['guid', 'url', 'date'])
    klue_tc_eval = klue_tc_eval.remove_columns(['guid', 'url', 'date'])

    klue_tc_label = klue_tc_train.features['label']
    klue_tc_train = klue_tc_train.map(make_str_label, batched=True, batch_size=1000)

    ## 예제 3.20. 학습/검증/테스트 데이터셋 분할
    train_dataset = klue_tc_train.train_test_split(test_size=10000, shuffle=True, seed=42)['test']
    dataset = klue_tc_eval.train_test_split(test_size=1000, shuffle=True, seed=42)
    test_dataset = dataset['test']
    valid_dataset = dataset['train'].train_test_split(test_size=1000, shuffle=True, seed=42)['test']

    ## 예제 3.24. Trainer를 사용하지 않는 학습: (1) 학습을 위한 모델과 토크나이저 준비
    # 모델과 토크나이저 불러오기
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_id = "klue/roberta-base"
    model = AutoModelForSequenceClassification.from_pretrained(model_id,
                                                               num_labels=len(train_dataset.features['label'].names))
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model.to(device)

    # 모델의 예측 아이디와 문자열 레이블을 연결할 데이터를 모델 config에 저장
    id2label = {i: label for i, label in enumerate(train_dataset.features['label'].names)}
    label2id = {label: i for i, label in id2label.items()}
    model.config.id2label = id2label
    model.config.label2id = label2id

    ## 예제 3.25 Trainer를 사용하지 않는 학습: (2) 학습을 위한 데이터 준비
    # 데이터로더 만들기
    train_dataloader = make_dataloader(train_dataset, batch_size=8, shuffle=True)
    valid_dataloader = make_dataloader(valid_dataset, batch_size=8, shuffle=False)
    test_dataloader = make_dataloader(test_dataset, batch_size=8, shuffle=False)

    ## 예제 3.28 Trainer를 사용하지 않는 학습: (5) 학습 수행
    num_epochs = 2  # 1
    optimizer = AdamW(model.parameters(), lr=5e-5)

    # login(token="본인의 허깅페이스 토큰 입력")
    login(token=os.getenv('HUGGINGFACE_TOCKEN'))
    # repo_id = f"본인의 아이디 입력/roberta-base-klue-ynat-classification"
    hub_model_id = f"hyunkookim/roberta-base-klue-ynat-classification-using-pytorch-epoch_{num_epochs}"
    output_dir = f"../results/{hub_model_id}"
    os.makedirs(output_dir, exist_ok=True)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        train_loss = train_epoch(model, train_dataloader, optimizer)
        print(f"Training loss: {train_loss}")
        valid_loss, valid_accuracy = evaluate(model, valid_dataloader)
        print(f"Validation loss: {valid_loss}")
        print(f"Validation accuracy: {valid_accuracy}")

        # 모델 저장 및 푸시
        save_dir = f"{output_dir}/model_epoch_{epoch + 1}"
        os.makedirs(save_dir, exist_ok=True)
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)

        # Hugging Face Hub에 푸시
        model.push_to_hub(hub_model_id, commit_message=f"Epoch {epoch + 1} model")
        tokenizer.push_to_hub(hub_model_id, commit_message=f"Epoch {epoch + 1} tokenizer")

    # 최종 모델 테스트 및 푸시
    _, test_accuracy = evaluate(model, test_dataloader)
    print(f"Test accuracy: {test_accuracy}")

    # # 학습 루프
    # for epoch in range(num_epochs):
    #     print(f"Epoch {epoch + 1}/{num_epochs}")
    #     train_loss = train_epoch(model, train_dataloader, optimizer)
    #     print(f"Training loss: {train_loss}")
    #     valid_loss, valid_accuracy = evaluate(model, valid_dataloader)
    #     print(f"Validation loss: {valid_loss}")
    #     print(f"Validation accuracy: {valid_accuracy}")
    #
    # # Testing
    # _, test_accuracy = evaluate(model, test_dataloader)
    # print(f"Test accuracy: {test_accuracy}")  # 정확도 0.82

    ## 예제 3.29. 허깅페이스 허브에 모델 업로드





    #
    # # 직접 학습한 경우
    # model.push_to_hub(repo_id)
    # tokenizer.push_to_hub(repo_id)

