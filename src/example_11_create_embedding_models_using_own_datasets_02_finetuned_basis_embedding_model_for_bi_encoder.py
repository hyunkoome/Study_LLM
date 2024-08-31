import os
from dotenv import load_dotenv

load_dotenv('.huggingface_env2')
print(os.environ['HF_HOME'])

from utils.common import ignore_warnings
from pathlib import Path
from huggingface_hub import login
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from sentence_transformers import InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers import datasets
from sentence_transformers import losses
from huggingface_hub import HfApi


def add_ir_context(df):
    """
    ## 예제 11.14 질문과 관련이 없는 기사를 irrelevant_context 컬럼에 추가
    :param df:
    :return:
    """
    irrelevant_contexts = []
    for idx, row in df.iterrows():
        title = row['title']
        irrelevant_contexts.append(df.query(f"title != '{title}'").sample(n=1)['context'].values[0])
    df['irrelevant_context'] = irrelevant_contexts
    return df

if __name__ == '__main__':
    ignore_warnings()

    print("## 예제 11.11 실습 데이터를 내려받고 예시 데이터 확인")
    klue_mrc_train = load_dataset('klue', 'mrc', split='train')
    print(klue_mrc_train[0])
    # {'title': '제주도 장마 시작 … 중부는 이달 말부터',
    #  'context': '올여름 장마가 17일 제주도에서 시작됐다. 서울 등 중부지방은 예년보다 사나흘 정도 늦은 이달 말께 장마가 시작될 전망이다.17일 기상청에 따르면 제주도 남쪽 먼바다에 있는 장마전선의 영향으로 이날 제주도 산간 및 내륙지역에 호우주의보가 내려지면서 곳곳에 100㎜에 육박하는 많은 비가 내렸다. 제주의 장마는 평년보다 2~3일, 지난해보다는 하루 일찍 시작됐다. 장마는 고온다습한 북태평양 기단과 한랭 습윤한 오호츠크해 기단이 만나 형성되는 장마전선에서 내리는 비를 뜻한다.장마전선은 18일 제주도 먼 남쪽 해상으로 내려갔다가 20일께 다시 북상해 전남 남해안까지 영향을 줄 것으로 보인다. 이에 따라 20~21일 남부지방에도 예년보다 사흘 정도 장마가 일찍 찾아올 전망이다. 그러나 장마전선을 밀어올리는 북태평양 고기압 세력이 약해 서울 등 중부지방은 평년보다 사나흘가량 늦은 이달 말부터 장마가 시작될 것이라는 게 기상청의 설명이다. 장마전선은 이후 한 달가량 한반도 중남부를 오르내리며 곳곳에 비를 뿌릴 전망이다. 최근 30년간 평균치에 따르면 중부지방의 장마 시작일은 6월24~25일이었으며 장마기간은 32일, 강수일수는 17.2일이었다.기상청은 올해 장마기간의 평균 강수량이 350~400㎜로 평년과 비슷하거나 적을 것으로 내다봤다. 브라질 월드컵 한국과 러시아의 경기가 열리는 18일 오전 서울은 대체로 구름이 많이 끼지만 비는 오지 않을 것으로 예상돼 거리 응원에는 지장이 없을 전망이다.',
    #  'news_category': '종합',
    #  'source': 'hankyung',
    #  'guid': 'klue-mrc-v1_train_12759',
    #  'is_impossible': False,
    #  'question_type': 1,
    #  'question': '북태평양 기단과 오호츠크해 기단이 만나 국내에 머무르는 기간은?',
    #  'answers': {'answer_start': [478, 478], 'text': ['한 달가량', '한 달']}}

    print("\n## 예제 11.12 기본 임베딩 모델 불러오기")
    sentence_model = SentenceTransformer('hyunkookim/klue-roberta-base-klue-sts')

    print("\n## 예제 11.13 데이터 전처리")
    klue_mrc_train = load_dataset('klue', 'mrc', split='train')
    klue_mrc_test = load_dataset('klue', 'mrc', split='validation')

    df_train = klue_mrc_train.to_pandas()
    df_test = klue_mrc_test.to_pandas()

    df_train = df_train[['title', 'question', 'context']]
    df_test = df_test[['title', 'question', 'context']]

    print("\n## 예제 11.14 질문과 관련이 없는 기사를 irrelevant_context 컬럼에 추가")
    df_train_ir = add_ir_context(df_train)
    df_test_ir = add_ir_context(df_test)

    print("\n## 예제 11.15 성능 평가에 사용할 데이터 생성")
    examples = []
    for idx, row in df_test_ir[:100].iterrows():
        examples.append(InputExample(texts=[row['question'], row['context']], label=1))
        examples.append(InputExample(texts=[row['question'], row['irrelevant_context']], label=0))

    print("\n## 예제 11.16 기본 임베딩 모델의 성능 평가 결과")
    evaluator = EmbeddingSimilarityEvaluator.from_input_examples(examples)
    print(evaluator(sentence_model))
    # 0.8151553052035344

    print("\n## 예제 11.17 긍정 데이터만으로 학습 데이터 구성")
    train_samples = []
    for idx, row in df_train_ir.iterrows():
        train_samples.append(InputExample(texts=[row['question'], row['context']]))

    print("\n## 예제 11.18 중복 학습 데이터 제거")
    batch_size = 16
    loader = datasets.NoDuplicatesDataLoader(train_samples, batch_size=batch_size)

    print("\n## 예제 11.19 MNR 손실 함수 불러오기")
    loss = losses.MultipleNegativesRankingLoss(sentence_model)

    print("\n## 예제 11.20 MRC 데이터셋으로 미세 조정")
    save_root_dir = "/home/hyunkoo/DATA/HDD8TB/GenAI/Study_LLM/src/results_data/example11"
    Path(save_root_dir).parent.mkdir(parents=True, exist_ok=True)

    epochs = 1
    save_path = f'{save_root_dir}/training_sts_klue-roberta-base-klue-sts-mrc'
    sentence_model.fit(
        train_objectives=[(loader, loss)],
        epochs=epochs,
        warmup_steps=100,
        output_path=save_path,
        show_progress_bar=True
    )

    print("\n## 예제 11.21 미세 조정한 모델 성능 평가")
    print(evaluator(sentence_model))
    # 0.8600968992433692

    print("\n## 예제 11.22 허깅페이스 허브에 미세 조정한 모델 업로드")
    # login(token='허깅페이스 허브 토큰 입력')
    login(token=os.environ['HF_TOKEN'])

    api = HfApi()
    repo_id = "klue-roberta-base-klue-sts-mrc"
    api.create_repo(repo_id=repo_id)

    api.upload_folder(
        folder_path=save_path,
        repo_id=f"hyunkookim/{repo_id}",
        repo_type="model",
    )
