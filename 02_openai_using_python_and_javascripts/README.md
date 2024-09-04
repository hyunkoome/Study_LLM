### 파이썬과 자바스크립트로 배우는 OpenAI 프로그래밍

<img src="./figures/02_openai_using_python_and_javascripts.png"></img><br/>

### This code is a modified version from [here](https://github.com/moseskim/openaiapi).
- This code has been confirmed and tested to work correctly on both GPUs of A100 (80GB) and RTX 4090 (24GB).

### Setting Python Environments
- Dev Environments
  - OS: Ubuntu 20.04
  - CUDA: 12.1
  - CUDNN: 8.9.0
  - Python: 3.9 
```shell
$ git clone https://github.com/hyunkoome/Study_LLM.git
$ cd Study_LLM
$ conda create -n llmbook python==3.9
$ conda activate llmbook
$ pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
$ pip install transformers[torch]==4.44.1
$ cd AutoGPTQ
$ pip install -vvv --no-build-isolation -e .
$ cd ..
$ pip install -r src/requirements.txt --use-deprecated=legacy-resolver
```

### 주요 책 목차 
- 