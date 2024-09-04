import os
from dotenv import load_dotenv

load_dotenv('.huggingface_env2')
print(os.environ['HF_HOME'])

from utils.common import ignore_warnings
import json
import autogen
from autogen import AssistantAgent, UserProxyAgent
from autogen.agentchat.contrib.retrieve_assistant_agent import RetrieveAssistantAgent
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent
from chromadb.utils import embedding_functions
def termination_msg(x):
    return isinstance(x, dict) and "TERMINATE" == str(x.get("content", ""))[-9:].upper()


## 예제 15.10 RAG 사용 여부에 따른 2개의 그룹챗 정의 및 실행
def _reset_agents(user, user_rag, coder, pm):
    user.reset()
    user_rag.reset()
    coder.reset()
    pm.reset()

def rag_chat(user, user_rag, coder, pm):
    _reset_agents(user, user_rag, coder, pm)
    groupchat = autogen.GroupChat(
        agents=[user_rag, coder, pm],
        messages=[], max_round=12, speaker_selection_method="round_robin"
    )
    manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

    user_rag.initiate_chat(
        manager,
        problem=PROBLEM,
    )

def norag_chat(user, user_rag, coder, pm):
    _reset_agents(user, user_rag, coder, pm)
    groupchat = autogen.GroupChat(
        agents=[user, coder, pm],
        messages=[],
        max_round=12,
        speaker_selection_method="auto",
        allow_repeat_speaker=False,
    )
    manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

    user.initiate_chat(
        manager,
        message=PROBLEM,
    )

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

    print("\n## 예제 15.8 OpenAI 임베딩 모델을 사용하도록 설정하기")
    assistant = RetrieveAssistantAgent(
        name="assistant",
        system_message="You are a helpful assistant.",
        llm_config=llm_config,
    )

    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=openai_api_key,
        model_name="text-embedding-3-small"
    )

    ragproxyagent = RetrieveUserProxyAgent(
        name="ragproxyagent",
        is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
        human_input_mode="NEVER",
        retrieve_config={
            "task": "qa",
            "docs_path": "https://raw.githubusercontent.com/microsoft/autogen/main/README.md",
            "embedding_function": openai_ef,
            "collection_name": "openai-embedding-3",
        },
    )

    assistant.reset()
    ragproxyagent.initiate_chat(assistant, problem="Autogen이 뭐야?")
    # assistant (to ragproxyagent):
    # AutoGen은 여러 에이전트가 상호 대화하여 작업을 해결할 수 있는 LLM(Large Language Model) 애플리케이션 개발을 가능하게 하는 프레임워크입니다.
    # AutoGen 에이전트는 사용자 정의 가능하며, 대화 가능하고, 인간 참여를 원활하게 허용합니다.
    # LLM, 인간 입력, 도구의 조합을 사용하는 다양한 모드에서 작동할 수 있습니다.

    print("\n## 예제 15.9 대화에 참여할 에이전트")
    # RAG를 사용하지 않는 사용자 역할 에이전트
    user = autogen.UserProxyAgent(
        name="Admin",
        is_termination_msg=termination_msg,
        human_input_mode="NEVER",
        system_message="The boss who ask questions and give tasks.",
        code_execution_config=False,
        default_auto_reply="Reply `TERMINATE` if the task is done.",
    )
    # RAG를 사용하는 사용자 역할 에이전트
    user_rag = RetrieveUserProxyAgent(
        name="Admin_RAG",
        is_termination_msg=termination_msg,
        system_message="Assistant who has extra content retrieval power for solving difficult problems.",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=3,
        code_execution_config=False,
        retrieve_config={
            "task": "code",
            "docs_path": "https://raw.githubusercontent.com/microsoft/autogen/main/samples/apps/autogen-studio/README.md",
            "chunk_token_size": 1000,
            "collection_name": "groupchat-rag",
        }
    )
    # 프로그래머 역할의 에이전트
    coder = AssistantAgent(
        name="Senior_Python_Engineer",
        is_termination_msg=termination_msg,
        system_message="You are a senior python engineer. Reply `TERMINATE` in the end when everything is done.",
        llm_config=llm_config,
    )
    # 프로덕트 매니저 역할의 에이전트
    pm = autogen.AssistantAgent(
        name="Product_Manager",
        is_termination_msg=termination_msg,
        system_message="You are a product manager. Reply `TERMINATE` in the end when everything is done.",
        llm_config=llm_config,
    )

    PROBLEM = "AutoGen Studio는 무엇이고 AutoGen Studio로 어떤 제품을 만들 수 있을까?"

    print("\n## 예제 15.11 2개의 그룹챗을 실행한 결과 비교")
    print("\n********************************************* norag_chat() *********************************************")
    norag_chat(user, user_rag, coder, pm)
    # AutoGen Studio는 자동화된 코드 생성 도구입니다. 이 도구를 사용하면 개발자들이 더 빠르게, 더 효율적으로 코드를 작성할 수 있습니다.
    # AutoGen Studio를 사용하면 다양한 유형의 소프트웨어 제품을 만들 수 있습니다.
    # 예를 들어, 웹 애플리케이션, 모바일 애플리케이션, 데스크톱 애플리케이션, API, 데이터베이스 등을 만들 수 있습니다.
    # ...

    print("\n********************************************** rag_chat() **********************************************")
    rag_chat(user, user_rag, coder, pm)
    # AutoGen Studio는 AutoGen 프레임워크를 기반으로 한 AI 앱입니다.
    # 이 앱은 AI 에이전트를 빠르게 프로토타입화하고, 스킬을 향상시키고, 워크플로우로 구성하고, 작업을 완료하기 위해 그들과 상호 작용하는 데 도움을 줍니다.
    # 이 앱은 GitHub의 [microsoft/autogen](https://github.com/microsoft/autogen/tree/main/samples/apps/autogen-studio)에서 코드를 찾을 수 있습니다.
    # AutoGen Studio를 사용하면 다음과 같은 기능을 수행할 수 있습니다:
    # - 에이전트를 구축/구성하고, 그들의 구성(예: 스킬, 온도, 모델, 에이전트 시스템 메시지, 모델 등)을 수정하고, 워크플로우로 구성합니다.
    # ...
