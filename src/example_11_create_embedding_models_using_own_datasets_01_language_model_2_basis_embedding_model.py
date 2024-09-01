import os
from dotenv import load_dotenv

load_dotenv('.huggingface_env2')
print(os.environ['HF_HOME'])

from utils.common import ignore_warnings
from sentence_transformers import SentenceTransformer, models
from datasets import load_dataset
from sentence_transformers import InputExample
from torch.utils.data import DataLoader
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers import losses
from pathlib import Path
from huggingface_hub import login
from huggingface_hub import HfApi

def prepare_sts_examples(dataset):
    """
    # 유사도 점수를 0~1 사이로 정규화 하고 InputExample 객체에 담는다.

    :param dataset:
    :return:
    """
    examples = []
    for data in dataset:
        examples.append(
            InputExample(
                texts=[data['sentence1'], data['sentence2']],
                label=data['labels']['label'] / 5.0)
            )
    return examples



if __name__ == '__main__':
    ignore_warnings()

    print("## 예제 11.1 사전 학습된 언어 모델을 불러와 문장 임베딩 모델 만들기")
    transformer_model = models.Transformer('klue/roberta-base')

    pooling_layer = models.Pooling(
        transformer_model.get_word_embedding_dimension(),
        pooling_mode_mean_tokens=True
    )
    embedding_model = SentenceTransformer(modules=[transformer_model, pooling_layer])

    print("\n## 예제 11.2 실습 데이터셋 다운로드 및 확인")
    klue_sts_train = load_dataset('klue', 'sts', split='train')
    klue_sts_test = load_dataset('klue', 'sts', split='validation')
    print(klue_sts_train[0])
    # {'guid': 'klue-sts-v1_train_00000',
    #  'source': 'airbnb-rtt',
    #  'sentence1': '숙소 위치는 찾기 쉽고 일반적인 한국의 반지하 숙소입니다.',
    #  'sentence2': '숙박시설의 위치는 쉽게 찾을 수 있고 한국의 대표적인 반지하 숙박시설입니다.',
    #  'labels': {'label': 3.7, 'real-label': 3.714285714285714, 'binary-label': 1}}

    print("\n## 예제 11.3 학습 데이터에서 검증 데이터셋 분리하기")
    # 학습 데이터셋의 10%를 검증 데이터셋으로 구성한다.
    klue_sts_train = klue_sts_train.train_test_split(test_size=0.1, seed=42)
    klue_sts_train, klue_sts_eval = klue_sts_train['train'], klue_sts_train['test']

    print("\n## 예제 11.4 label 정규화하기")
    train_examples = prepare_sts_examples(klue_sts_train)
    eval_examples = prepare_sts_examples(klue_sts_eval)
    test_examples = prepare_sts_examples(klue_sts_test)

    print("\n## 예제 11.5 학습에 사용할 배치 데이터셋 만들기")
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

    print("\n## 예제 11.6 검증을 위한 평가 객체 준비")
    eval_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(eval_examples)
    test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_examples)

    print("\n## 예제 11.7 언어 모델을 그대로 활용할 경우 문장 임베딩 모델의 성능")
    print(test_evaluator(embedding_model))
    # 0.36460670798564826

    print("\n## 예제 11.8 임베딩 모델 학습")
    save_root_dir = "/home/hyunkoo/DATA/HDD8TB/GenAI/Study_LLM/src/results_data/example11"
    Path(save_root_dir).parent.mkdir(parents=True, exist_ok=True)

    num_epochs = 4
    model_name = 'klue/roberta-base'
    model_save_path = f'{save_root_dir}/training_sts_' + model_name.replace("/", "-")
    train_loss = losses.CosineSimilarityLoss(model=embedding_model)

    # 임베딩 모델 학습
    embedding_model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=eval_evaluator,
        epochs=num_epochs,
        evaluation_steps=1000,
        warmup_steps=100,
        output_path=model_save_path
    )

    print('\n## 예제 11.9 학습한 임베딩 모델의 성능 평가')
    trained_embedding_model = SentenceTransformer(model_save_path)
    print(test_evaluator(trained_embedding_model))
    # 0.8965595666246748

    print("\n## 예제 11.10 허깅페이스 허브에 모델 저장")
    # login(token='허깅페이스 허브 토큰 입력')
    login(token=os.environ['HF_TOKEN'])
    api = HfApi()
    repo_id = "klue-roberta-base-klue-sts"
    api.create_repo(repo_id=repo_id)

    api.upload_folder(
        folder_path=model_save_path,
        # repo_id=f"본인의 허깅페이스 아이디 입력/{repo_id}",
        repo_id=f"hyunkookim/{repo_id}",
        repo_type="model",
    )
