import os
from dotenv import load_dotenv

load_dotenv('.huggingface_env2')
print(os.environ['HF_HOME'])

from utils.common import ignore_warnings
import json
import autogen
from autogen.agentchat.contrib.retrieve_assistant_agent import RetrieveAssistantAgent
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent
from chromadb.utils import embedding_functions

# from autogen import AssistantAgent, UserProxyAgent
# def termination_msg(x):
#     return isinstance(x, dict) and "TERMINATE" == str(x.get("content", ""))[-9:].upper()

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

    print("\n## 예제 15.6 RAG 에이전트 클래스를 사용한 작업 실행")
    assistant = RetrieveAssistantAgent(
        name="assistant",
        system_message="You are a helpful assistant.",
        llm_config=llm_config,
    )

    ragproxyagent = RetrieveUserProxyAgent(
        name="ragproxyagent",
        retrieve_config={
            "task": "qa",
            "docs_path": "https://raw.githubusercontent.com/microsoft/autogen/main/README.md",
            "collection_name": "default-sentence-transformers"
        },
        # is_termination_msg=termination_msg,
        # max_consecutive_auto_reply=3,
        # human_input_mode="NEVER",
        # code_execution_config=False,
        # retrieve_config={
        #     "task": "code",
        #     "docs_path": "https://raw.githubusercontent.com/microsoft/autogen/main/README.md",
        #     "chunk_token_size": 1000,
        #     "collection_name": "groupchat-rag",
        # },
    )

    assistant.reset()
    ragproxyagent.initiate_chat(assistant, problem="AutoGen이 뭐야?")
    # assistant (to ragproxyagent):
    # AutoGen은 여러 에이전트가 상호 대화하여 작업을 해결할 수 있는 LLM(Large Language Model) 애플리케이션 개발을 가능하게 하는 프레임워크입니다.
    # AutoGen 에이전트는 사용자 정의 가능하며, 대화 가능하고, 인간 참여를 원활하게 허용합니다.
    # LLM, 인간 입력, 도구의 조합을 사용하는 다양한 모드에서 작동할 수 있습니다.

    print("\n## 예제 15.7 외부 정보를 활용하지 못하는 기본 에이전트의 답변")
    assistant.reset()
    userproxyagent = autogen.UserProxyAgent(
        name="userproxyagent",
    )
    userproxyagent.initiate_chat(assistant, message="Autogen이 뭐야?")
    # assistant (to userproxyagent):
    # "Autogen"은 자동 생성을 의미하는 용어로, 주로 컴퓨터 프로그래밍에서 사용됩니다.
    # 이는 코드, 문서, 또는 다른 데이터를 자동으로 생성하는 프로세스를 가리킵니다.
    # 이는 반복적인 작업을 줄이고, 효율성을 높이며, 오류를 줄일 수 있습니다. 특정 컨텍스트에 따라 "Autogen"의 정확한 의미는 다를 수 있습니다.


