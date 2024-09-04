import os
from dotenv import load_dotenv

load_dotenv('.huggingface_env2')
print(os.environ['HF_HOME'])

from utils.common import ignore_warnings

from datasets import load_dataset
import requests
import base64
from io import BytesIO
import os
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
import torch
from tqdm.auto import trange
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, CLIPTextModelWithProjection
from transformers import AutoProcessor, CLIPVisionModelWithProjection
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path


# from datasets import load_dataset
# from llama_index.core import Document, VectorStoreIndex, get_response_synthesizer
# from llama_index.core.retrievers import VectorIndexRetriever
# from llama_index.core.query_engine import RetrieverQueryEngine
# from llama_index.core.postprocessor import SimilarityPostprocessor
#
# from pinecone import Pinecone, ServerlessSpec
# # 라마인덱스에 파인콘 인덱스 연결
# from llama_index.core import VectorStoreIndex
# from llama_index.vector_stores.pinecone import PineconeVectorStore
# from llama_index.core import StorageContext


## 예제 12.15 GPT-4o 요청에 사용할 함수

def make_base64(image, image_png_save_full_path):
    buffered = BytesIO()
    # 이미지를 PNG 형식으로 인코딩하여 buffered 객체에 저장합니다. 이 과정에서 이미지가 바이너리 데이터로 변환
    # 파일로 저장되지 않고, 메모리 상의 buffered 객체에 png 형식의 이미지를 저장합니다.
    image.save(buffered, format="PNG")
    # image.save(buffered, format="PNG") 부분이 주석 처리되면, buffered 객체에는 아무것도 저장되지 않습니다.
    # 이 줄을 주석 처리하면
    # -> 이미지 데이터가 buffered에 저장되지 않기 때문에
    # -> buffered_data = buffered.getvalue()가 빈 데이터를 반환하므로,
    # -> Base64로 인코딩한 img_str도 빈 문자열, 즉 img_str이 공백으로 나오는 것

    buffered_data = buffered.getvalue() # buffered 객체에 저장된 바이너리 데이터를 가져와 buffered_data 변수에 저장
    # 메모리에서 파일로 저장
    with open(image_png_save_full_path, "wb") as f:
        f.write(buffered_data)

    # 가져온 바이너리 데이터를 Base64로 인코딩하여 문자열로 변환합니다. 이 문자열이 img_str에 저장됨
    img_str = base64.b64encode(buffered_data).decode('utf-8')
    return img_str


def generate_description_from_image_gpt4(prompt, image64, openai_client):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_client.api_key}"
    }
    payload = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image64}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 300
    }
    response_oai = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    result = response_oai.json()['choices'][0]['message']['content']
    return result


## 예제 12.23 프롬프트로 이미지를 생성하고 저장하는 함수 정의
def generate_image_dalle3(prompt, openai_client):
    response_oai = openai_client.images.generate(
        model="dall-e-3",
        prompt=str(prompt),
        size="1024x1024",
        quality="standard",
        n=1,
    )
    result = response_oai.data[0].url
    return result


def get_generated_image(image_url, image_png_save_full_path):
    generated_image = requests.get(image_url).content
    with open(image_png_save_full_path, "wb") as image_file:
        image_file.write(generated_image)
    return Image.open(image_png_save_full_path)


def get_generated_images(original_image, original_prompt_image, searched_prompt_image, gpt4o_prompt_image,
                         image_png_save_full_path):
    # 이미지 리스트와 타이틀 리스트 정의
    images = [original_image, original_prompt_image, searched_prompt_image, gpt4o_prompt_image]
    titles = ['(a) original_image', '(b) original_prompt_image', '(c) searched_prompt_image', '(d) gpt4o_prompt_image']

    # 서브플롯 생성
    fig, axes = plt.subplots(1, len(images), figsize=(15, 5))

    # 각 이미지에 라벨과 타이틀 설정
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(title, fontsize=10)

    # 레이아웃 조정 및 플롯 표시
    plt.tight_layout()
    plt.show()

    # 플롯을 파일로 저장
    fig.savefig(image_png_save_full_path, bbox_inches='tight')


def create_pinecone_index(pinecone_client, index_name, embedding_dimension, similarity="cosine"):
    print(pinecone_client.list_indexes())

    try:
        pinecone_client.create_index(
            name=index_name,
            dimension=embedding_dimension,  # 512
            metric=similarity,  # "cosine"
            spec=ServerlessSpec(
                "aws", "us-east-1"
            )
        )
        print(pinecone_client.list_indexes())
    except:
        print("Index already exists")
    index = pinecone_client.Index(index_name)
    return index


def get_embedding_dimension(openai_clip_model_name):
    # 모델의 임베딩 벡터 확인
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 모델 및 토크나이저 로드
    text_model = CLIPTextModelWithProjection.from_pretrained(openai_clip_model_name).to(device)  # 모델을 GPU로 이동
    tokenizer = AutoTokenizer.from_pretrained(openai_clip_model_name)

    # 예제 텍스트 입력
    inputs = tokenizer("A sample text", return_tensors="pt").to(device)
    outputs = text_model(**inputs)

    # 임베딩 벡터 크기 확인
    print(outputs.text_embeds.shape)  # 예: torch.Size([1, 512])
    embedding_dim = outputs.text_embeds.shape[1]
    print("embedding_dim: ", embedding_dim)
    print("text_model.config.projection_dim: ", text_model.config.projection_dim)
    assert embedding_dim == text_model.config.projection_dim

    return embedding_dim


if __name__ == '__main__':
    ignore_warnings()

    print("## 예제 12.14 실습 데이터셋 다운로드")
    # 1000개 짜리 데이터셋
    dataset = load_dataset("poloclub/diffusiondb", "2m_first_1k", split='train')

    example_index = 867
    original_image = dataset[example_index]['image']
    original_prompt = dataset[example_index]['prompt']
    print(original_prompt)
    # "cute fluffy baby cat rabbit lion hybrid mixed creature": 귀여운, 털이 복슬복슬한 아기 고양이, 토끼, 사자 혼합 생물 캐릭터.
    # "with long flowing mane blowing in the wind": 바람에 흩날리는 긴 갈기를 가진 생물.
    # "long peacock feather tail": 긴 공작 깃털로 이루어진 꼬리를 가지고 있음.
    # "wearing headdress of tribal peacock feathers and flowers": 공작 깃털과 꽃으로 이루어진 부족 스타일의 머리 장식을 착용.
    # "detailed painting, renaissance, 4K": 르네상스 시대 스타일의 정교한 그림처럼 묘사되며, 4K 해상도로 매우 세밀하게 표현된 모습.

    print("\n## 예제 12.17 클라이언트 준비")
    pinecone_client = Pinecone(api_key=os.environ['PINECONE_API_KEY'])
    openai_client = OpenAI()

    print("\n## 예제 12.16 이미지 설명 생성")
    image_save_dir = "/home/hyunkoo/DATA/HDD8TB/GenAI/Study_LLM/src/data/example12"
    Path(image_save_dir).mkdir(parents=True, exist_ok=True)

    image_base64 = make_base64(image=original_image, image_png_save_full_path=f"{image_save_dir}/original_image.png")
    described_result = generate_description_from_image_gpt4(prompt="Describe provided image",
                                                            image64=image_base64,
                                                            openai_client=openai_client)
    print(described_result)

    print("\n## 예제 12.18 인덱스 생성")
    openai_clip_model_name = "openai/clip-vit-base-patch32"
    embedding_vector_dim = get_embedding_dimension(openai_clip_model_name=openai_clip_model_name)  # 512 차원

    pinecone_index_name = "llm-multimodal"
    pinecone_index = create_pinecone_index(pinecone_client=pinecone_client,
                                           index_name=pinecone_index_name,
                                           embedding_dimension=int(embedding_vector_dim),  # 512 차원
                                           similarity="cosine")

    print("\n## 예제 12.19 프롬프트 텍스트를 텍스트 임베딩 모델을 활용해 임베딩 벡터로 변환: 임베딩 벡터가 512 차원임")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 모델을 GPU로 이동
    text_model = CLIPTextModelWithProjection.from_pretrained(openai_clip_model_name).to(device)  # 임베딩 벡터가 512 차원
    tokenizer = AutoTokenizer.from_pretrained(openai_clip_model_name)

    tokens = tokenizer(dataset['prompt'], padding=True, return_tensors="pt", truncation=True).to(device)  # 텐서를 GPU로 이동
    batch_size = 16
    text_embs = []
    for start_idx in trange(0, len(dataset), batch_size):
        with torch.no_grad():
            outputs = text_model(input_ids=tokens['input_ids'][start_idx:start_idx + batch_size],
                                 attention_mask=tokens['attention_mask'][start_idx:start_idx + batch_size])
            text_emb_tmp = outputs.text_embeds
        text_embs.append(text_emb_tmp)
    text_embs = torch.cat(text_embs, dim=0).to(device)  # 최종 결과도 GPU로 이동
    print(text_embs.shape)  # (1000, 512): 데이터셋 개수가 1000개, 임베딩 벡터가 512 차원

    print("\n## 예제 12.20 텍스트 임베딩 벡터를 파인콘 인덱스에 저장")
    # GPU에 있는 텐서를 CPU로 이동시키고 리스트로 변환
    text_embs_cpu = text_embs.cpu().tolist()

    input_data = []
    for id_int, emb, prompt in zip(range(0, len(dataset)), text_embs_cpu, dataset['prompt']):
        input_data.append({"id": str(id_int), "values": emb, "metadata": {"prompt": prompt}})
    pinecone_index.upsert(vectors=input_data)

    print("\n## 예제 12.21 이미지 임베딩을 사용한 유사 프롬프트 검색")
    vision_model = CLIPVisionModelWithProjection.from_pretrained(openai_clip_model_name)
    processor = AutoProcessor.from_pretrained(openai_clip_model_name)

    inputs = processor(images=original_image, return_tensors="pt")
    outputs = vision_model(**inputs)
    image_embeds = outputs.image_embeds
    search_results = pinecone_index.query(vector=image_embeds[0].tolist(),
                                          top_k=3,
                                          include_values=False,
                                          include_metadata=True)
    searched_idx = int(search_results['matches'][0]['id'])

    print("\n## 예제 12.22 이미지 임베딩을 사용해 검색한 유사 프롬프트 확인")
    print("search_results: ", search_results)
    print("searched_idx: ", searched_idx)  # 918
    # {'matches': [{'id': '918',
    #               'metadata': {'prompt': 'cute fluffy bunny cat lion hybrid mixed '
    #                                      'creature character concept, with long '
    #                                      'flowing mane blowing in the wind, long '
    #                                      'peacock feather tail, wearing headdress '
    #                                      'of tribal peacock feathers and flowers, '
    #                                      'detailed painting, renaissance, 4 k '},
    #               'score': 0.372838408,
    #               'values': []},
    #              {'id': '867',
    #               'metadata': {'prompt': 'cute fluffy baby cat rabbit lion hybrid '
    #                                      'mixed creature character concept, with '
    #                                      'long flowing mane blowing in the wind, '
    #                                      'long peacock feather tail, wearing '
    #                                      'headdress of tribal peacock feathers and '
    #                                      'flowers, detailed painting, renaissance, '
    #                                      '4 k '},
    #               'score': 0.371655703,
    #               'values': []},
    # ...

    print("\n## 예제 12.24 준비한 3개의 프롬프트로 이미지 생성")

    print("\n# 원본 프롬프트로 이미지 생성")
    print("원본 이미지의 원본 프롬프트: ", original_prompt)
    # cute fluffy baby cat rabbit lion hybrid mixed creature character concept,
    # with long flowing mane blowing in the wind, long peacock feather tail,
    # wearing headdress of tribal peacock feathers and flowers, detailed painting,
    # renaissance, 4 k
    original_prompt_image_url = generate_image_dalle3(original_prompt, openai_client=openai_client)
    original_prompt_image = get_generated_image(image_url=original_prompt_image_url,
                                                image_png_save_full_path=f"{image_save_dir}/original_prompt_image.png")
    # original_prompt_image

    print("\n# 원본이미지의 이미지 임베딩으로 검색한 유사 이미지의 프롬프트로 이미지 생성")
    print("원본이미지의 이미지 임베딩으로 검색한 유사 이미지의 프롬프트: ", dataset[searched_idx]['prompt'])
    # cute fluffy bunny cat lion hybrid mixed creature character concept,
    # with long flowing mane blowing in the wind, long peacock feather tail,
    # wearing headdress of tribal peacock feathers and flowers, detailed painting,
    # renaissance, 4 k
    searched_prompt_image_url = generate_image_dalle3(dataset[searched_idx]['prompt'], openai_client=openai_client)
    searched_prompt_image = get_generated_image(image_url=searched_prompt_image_url,
                                                image_png_save_full_path=f"{image_save_dir}/searched_prompt_image.png")
    # searched_prompt_image

    print("\n# GPT-4o가 만든 프롬프트로 이미지 생성")
    print("원본 이미지를 보고, GPT-4o가 만든 프롬프트: ", described_result)
    gpt_described_image_url = generate_image_dalle3(described_result, openai_client=openai_client)
    gpt4o_prompt_image = get_generated_image(image_url=gpt_described_image_url,
                                             image_png_save_full_path=f"{image_save_dir}/gpt4o_prompt_image.png")
    # gpt4o_prompt_image

    print("\n## 예제 12.25 이미지 출력")
    get_generated_images(original_image=original_image,
                         original_prompt_image=original_prompt_image,
                         searched_prompt_image=searched_prompt_image,
                         gpt4o_prompt_image=gpt4o_prompt_image,
                         image_png_save_full_path=f"{image_save_dir}/all_images.png")
