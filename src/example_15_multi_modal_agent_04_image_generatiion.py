import os
from dotenv import load_dotenv

load_dotenv('.huggingface_env2')
print(os.environ['HF_HOME'])

from utils.common import ignore_warnings
import os
import json
import re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import matplotlib.pyplot as plt
import PIL
import requests
from openai import OpenAI
from PIL import Image
from pathlib import Path

import autogen
from autogen import Agent, ConversableAgent, UserProxyAgent
from autogen.agentchat.contrib.img_utils import _to_pil, get_image_data
from autogen.agentchat.contrib.multimodal_conversable_agent import MultimodalConversableAgent

def termination_msg(x):
    return isinstance(x, dict) and "TERMINATE" == str(x.get("content", ""))[-9:].upper()

## 예제 15.13 DALLEAgent 정의
def dalle_call(client, prompt, model="dall-e-3", size="1024x1024", quality="standard", n=1) -> str:
    response = client.images.generate(
        model=model,
        prompt=prompt,
        size=size,
        quality=quality,
        n=n,
    )
    image_url = response.data[0].url
    img_data = get_image_data(image_url)
    return img_data


class DALLEAgent(ConversableAgent):
    def __init__(self, name, llm_config: dict, **kwargs):
        super().__init__(name, llm_config=llm_config, **kwargs)

        try:
            config_list = llm_config["config_list"]
            api_key = config_list[0]["api_key"]
        except Exception as e:
            print("Unable to fetch API Key, because", e)
            api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key)
        self.register_reply([Agent, None], DALLEAgent.generate_dalle_reply)
        self.save_path = "./generated_image.png"

    def generate_dalle_reply(self, messages, sender, config):
        client = self.client if config is None else config
        if client is None:
            return False, None
        if messages is None:
            messages = self._oai_messages[sender]

        prompt = messages[-1]["content"]
        print(prompt)
        img_data = dalle_call(client=self.client, prompt=prompt)
        generated_image = _to_pil(img_data)

        # 이미지 저장
        generated_image.save(self.save_path)  # 저장 경로와 파일 이름 설정
        print(f"Image saved at {self.save_path}")

        return True, self.save_path

    def set_save_path(self, save_path):
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        self.save_path = save_path

if __name__ == '__main__':
    ignore_warnings()

    print("## 예제 15.2 OpenAI API 키 설정")
    # openai_api_key = "자신의 API 키 입력"
    openai_api_key = os.environ['OPENAI_API_KEY']
    with open('OAI_CONFIG_LIST.json', 'w') as f:
        config_list = [
            {
                "model": "gpt-4-turbo-preview",
                "api_key": openai_api_key
            },
            {
                "model": "gpt-4o",
                "api_key": openai_api_key,
            },
            {
                "model": "dall-e-3",
                "api_key": openai_api_key,
            }
        ]
        json.dump(config_list, f)

    llm_config = {
        "config_list": config_list,
        "temperature": 0,
    }

    print("\n## 예제 15.12 실습 준비하기")
    config_list_4o = autogen.config_list_from_json(
        "OAI_CONFIG_LIST.json",
        filter_dict={
            "model": ["gpt-4o"],
        },
    )

    config_list_dalle = autogen.config_list_from_json(
        "OAI_CONFIG_LIST.json",
        filter_dict={
            "model": ["dall-e-3"],
        },
    )

    print("\n## 예제 15.14 이미지 생성 에이전트 실행")
    painter = DALLEAgent(name="Painter", llm_config={"config_list": config_list_dalle})

    user_proxy = UserProxyAgent(
        name="User_proxy", system_message="A human admin.", human_input_mode="NEVER", max_consecutive_auto_reply=0
    )

    print("\n# 이미지 생성 작업 실행하기(text -> 이미지 생성)")
    # 저장 경로 설정
    painter.set_save_path(save_path= './data/example15/generated_image_using_txt01.png')
    user_proxy.initiate_chat(
        painter,
        message="갈색의 털을 가진 귀여운 강아지를 그려줘",
    )

    print("\n## 예제 15.15 이미지를 입력으로 받을 수 있는 GPT-4o 에이전트 생성")
    image_agent = MultimodalConversableAgent(
        name="image-explainer",
        system_message="Explane input image for painter to create similar image.",
        max_consecutive_auto_reply=10,
        llm_config={"config_list": config_list_4o, "temperature": 0.5, "max_tokens": 1500},
    )

    user_proxy = autogen.UserProxyAgent(
        name="User_proxy",
        system_message="A human admin.",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=0,
        code_execution_config=False
    )

    groupchat = autogen.GroupChat(agents=[user_proxy, image_agent, painter], messages=[], max_round=12)
    manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

    print("\n## 예제 15.16 유사한 이미지를 생성하도록 에이전트 실행")
    painter.set_save_path(save_path='./data/example15/generated_image_using_txt_image.png')
    user_proxy.initiate_chat(
        manager,
        message=f"""아래 이미지랑 비슷한 이미지를 만들어줘.
    <img https://th.bing.com/th/id/R.422068ce8af4e15b0634fe2540adea7a?rik=y4OcXBE%2fqutDOw&pid=ImgRaw&r=0>.""",
    )

    print("\n## 예제 15.18 멀티 모달 에이전트에 텍스트로 명령")
    painter.set_save_path(save_path='./data/example15/generated_image_using_txt02.png')
    user_proxy.initiate_chat(
        manager,
        message="갈색의 털을 가진 귀여운 강아지를 그려줘",
    )