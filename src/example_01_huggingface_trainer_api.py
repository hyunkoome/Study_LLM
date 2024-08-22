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



if __name__ == '__main__':
    ## 예제 3.17. 모델 학습에 사용할 연합뉴스 데이터셋 다운로드
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

    ## 예제 3.21. 허깅스페이스 Trainer API를 사용한 학습: (1) 준비
    model_id = "klue/roberta-base"
    model = AutoModelForSequenceClassification.from_pretrained(model_id,
                                                               num_labels=len(train_dataset.features['label'].names))
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    valid_dataset = valid_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)

    ## 예제 3.22. 허깅스페이스 Trainer API를 사용한 학습: (2) 학습 인자와 평가 함수 정의
    train_epoch = 2

    # repo_id = f"본인의 아이디 입력/roberta-base-klue-ynat-classification"
    hub_model_id = f"hyunkookim/roberta-base-klue-ynat-classification-using-hg_api-epoch_{train_epoch}"
    output_dir = f"../results/{hub_model_id}"

    # 모델의 예측 아이디와 문자열 레이블을 연결할 데이터를 모델 config에 저장
    id2label = {i: label for i, label in enumerate(train_dataset.features['label'].names)}
    label2id = {label: i for i, label in id2label.items()}
    model.config.id2label = id2label
    model.config.label2id = label2id

    # hub_model_id: Hub에 동기화할 리포지토리의 이름을 지정하는 옵션입니다.
    # 지정하지 않으면 기본적으로 output_dir의 이름이 리포지토리 이름으로 사용됩니다.
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,  # "Overwrite the content of the output directory. "
        hub_model_id=hub_model_id,
        # num_train_epochs=1,
        num_train_epochs=train_epoch,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        eval_strategy="epoch",
        learning_rate=5e-5,
        push_to_hub=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # res =
    trainer.train()
    # print(res)

    # eval = trainer.evaluate(test_dataset)  # 정확도 0.84
    trainer.evaluate(test_dataset)  # 정확도 0.84
    # print(eval)

    # login(token="본인의 허깅페이스 토큰 입력")
    login(token=os.getenv('HUGGINGFACE_TOCKEN'))

    # Trainer를 사용한 경우
    # trainer.push_to_hub(repo_id)
    trainer.push_to_hub()
