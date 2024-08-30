import os
from dotenv import load_dotenv

load_dotenv('.huggingface_env2')
print(os.environ['HF_HOME'])

import torch
from utils.common import ignore_warnings
from transformers import CLIPProcessor, CLIPModel
import requests
from PIL import Image

if __name__ == '__main__':
    ignore_warnings()

    print("## 예제 14.1 허깅페이스로 CLIP 모델 활용")
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    print("\n## 예제 14.2 CLIP 모델 추론)")
    #코드 출처: https://huggingface.co/openai/clip-vit-large-patch14"
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    text_list = ["a photo of a cat", "a photo of a dog"]
    inputs = processor(text=text_list, images=image, return_tensors="pt", padding=True)

    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    print(probs) # tensor([[9.9925e-01, 7.5487e-04]], grad_fn=<SoftmaxBackward0>)
    # 최대 확률의 인덱스 찾기
    max_prob_index = torch.argmax(probs).item()
    # 해당 인덱스의 텍스트 출력
    print(text_list[max_prob_index])