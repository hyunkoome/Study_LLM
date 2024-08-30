import os
from dotenv import load_dotenv

load_dotenv('.huggingface_env2')
print(os.environ['HF_HOME'])

from utils.common import ignore_warnings
import json
import autogen
from autogen import AssistantAgent, UserProxyAgent

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

    print("\n## 예제 15.3 에이전트에 사용할 설정 불러오기")
    config_list = autogen.config_list_from_json(
        "OAI_CONFIG_LIST.json",
        file_location=".",
        filter_dict={
            "model": ["gpt-4-turbo-preview"],
        },
    )

    llm_config = {
        "config_list": config_list,
        "temperature": 0,
    }

    print("\n## 예제 15.4 AutoGen의 핵심 구성요소인 UserProxyAgent와 AssistantAgent")
    assistant = AssistantAgent(name="assistant", llm_config=llm_config)
    user_proxy = UserProxyAgent(name="user_proxy",
                                is_termination_msg=lambda x: x.get("content", "") and x.get("content",
                                                                                            "").rstrip().endswith(
                                    "TERMINATE"),
                                human_input_mode="NEVER",
                                code_execution_config={"work_dir": "data/coding", "use_docker": False})

    print("\n## 예제 15.5 삼성전자의 3개월 주식 가격 그래프를 그리는 작업 실행")
    user_proxy.initiate_chat(recipient=assistant,
                             message="""
                             삼성전자의 지난 3개월 주식 가격 그래프를 그려서 samsung_stock_price.png 파일로 저장해줘.                             
                             plotly 라이브러리를 사용하고 그래프 아래를 투명한 녹색으로 채워줘.
                             값을 잘 확인할 수 있도록 y축은 구간 최소값에서 시작하도록 해줘.
                             이미지 비율은 보기 좋게 적절히 설정해줘.
                             """)
