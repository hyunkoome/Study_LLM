# Study_LLM


### These codes are modified version from:
1. LLM을 활용한 실전 AI 애플리케이션 개발: [Read Me](./01_dev_AI_using_lmm/README.md)  
   - 참고: 공식 코드 [GitHub](https://github.com/onlybooks/llm)
2. 파이썬과 자바스크립트로 배우는 OpenAI 프로그래밍: [Read Me](./02_openai_using_python_and_javascripts/README.md) 
   - 참고: 공식 코드 [GitHub](https://github.com/moseskim/openaiapi)

### Setting Python Environments
- Confirmed and tested to work correctly on both GPUs of A100 (80GB) and RTX 4090 (24GB).
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

