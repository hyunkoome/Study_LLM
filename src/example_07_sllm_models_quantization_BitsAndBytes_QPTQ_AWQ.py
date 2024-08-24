import os
from dotenv import load_dotenv

load_dotenv('.huggingface_env2')
print(os.environ['HF_HOME'])

import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from transformers import AutoTokenizer, GPTQConfig
from awq import AutoAWQForCausalLM

if __name__ == '__main__':
    torch.cuda.empty_cache()
    print('## 예제 7.1. 비츠앤바이츠 양자화 모델 불러오기')
    print('# 8비트 양자화 모델 불러오기')
    bnb_config_8bit = BitsAndBytesConfig(load_in_8bit=True)
    model_8bit = AutoModelForCausalLM.from_pretrained("facebook/opt-350m", quantization_config=bnb_config_8bit)

    torch.cuda.empty_cache()
    print('# 4비트 양자화 모델 불러오기')
    bnb_config_4bit = BitsAndBytesConfig(load_in_4bit=True,
                                         bnb_4bit_quant_type="nf4")

    model_4bit = AutoModelForCausalLM.from_pretrained("facebook/opt-350m",
                                                      low_cpu_mem_usage=True,
                                                      quantization_config=bnb_config_4bit)

    print('## 예제 7.1. 끝.....')

    torch.cuda.empty_cache()
    print('## 예제 7.2. GPTQ 양자화 수행 코드')
    # 코드 출처: https://huggingface.co/blog/gptq-integration
    model_id = "facebook/opt-125m"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    quantization_config = GPTQConfig(bits=4, dataset="c4", tokenizer=tokenizer)

    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", quantization_config=quantization_config)

    print('## 예제 7.2. 끝.....')

    torch.cuda.empty_cache()
    print('## 예제 7.3. GPTQ 양자화된 모델 불러오기')
    model = AutoModelForCausalLM.from_pretrained("TheBloke/zephyr-7B-beta-GPTQ",
                                                 device_map="auto",
                                                 trust_remote_code=False,
                                                 revision="main")

    print('## 예제 7.3. 끝.....')

    torch.cuda.empty_cache()
    print('## 예제 7.4. AWQ 양자화 모델 불러오기')
    model_name_or_path = "TheBloke/zephyr-7B-beta-AWQ"
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=False)
    model = AutoAWQForCausalLM.from_quantized(model_name_or_path, fuse_layers=True, trust_remote_code=False,
                                              safetensors=True)
    print('## 예제 7.4. 끝.....')
    torch.cuda.empty_cache()