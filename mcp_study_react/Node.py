# 필요 라이브러리 임포트
import json
import requests
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from State import *
from Mcp_Tool import *
from chain import identity_chain, intent_classify_chain, intent_extract_chain, talk_chain, food_recommand_chain
from logger.observability import log_node_outputs

# 사용자 입력 노드 정의
@log_node_outputs("user_input_node",
                  include_keys=["user_input","messages","run_id"])
def user_input_node(state: InputState) -> OverallState:
    print("디버깅: state 입력값 =", state)
    user_input = state["user_input"]
    return {
        **state,
        "user_input": user_input,
        "messages": [("user", user_input)],
    }

# ──────────────────────── 1. LangGraph 노드 정의 ────────────────────────

# 의도 분류 노드
@log_node_outputs("intent_classify_node",
                  include_keys=["intent","run_id"])
def intent_classify_node(state: OverallState) -> OverallState:
    user_input = state["user_input"]
    result = intent_classify_chain.invoke({"user_input": user_input})
    return {
        **state,
        "intent": result["intent"]
    }

# 음식 추천 노드
@log_node_outputs("food_recommand_node",
                  include_keys=["exit_message","run_id"], max_str=400)
def food_recommand_node(state: OverallState) -> EndState:
    user_input = state["user_input"]
    result = food_recommand_chain.invoke({"user_input": user_input})
    print("디버깅: food_recommandnode 결과 =", result)
    return {
        **state,
        "exit_message": result["content"]
    }

# 정체성 정의 노드
@log_node_outputs("identity_node",
                  include_keys=["exit_message","run_id"], max_str=400)
def identity_node(state: OverallState) -> EndState:
    user_input = state["user_input"]
    result = identity_chain.invoke({"user_input": user_input})
    print("디버깅: identity_node 결과 =", result)
    return {
        **state,
        "exit_message": result["content"]
    }

# 의도 파악 불가 노드
@log_node_outputs("exit_node",
                  include_keys=["exit_message","run_id"])
def exit_node(state: OverallState) -> EndState:
    exit_message = "죄송합니다. 말씀하신 내용을 이해하지 못하였습니다. 식사비서 재규니는 현재 베타서비스 운영 중에 있습니다. 추후 서비스 확장하여 이런 요청도 처리할 수 있도록 하겠습니다."
    return {
        **state,
        "exit_message": exit_message
    }

# 일상 대화 노드
@log_node_outputs("talk_node",
                  include_keys=["exit_message","run_id"])
def talk_node(state: OverallState) -> EndState:
    user_input = state["user_input"]

    result = talk_chain.invoke({"user_input": user_input})
    print("디버깅: talk_node 결과 =", result)
    return {
        **state,
        "exit_message": result["content"]
    }

# 의도 추출 노드
@log_node_outputs("intent_extract_node",
                  include_keys=["location","conditions","condition_food_map","run_id"],
                  max_items=3)  # condition_food_map 미리보기 제한
def intent_extract_node(state: OverallState) -> OverallState:
    user_input = state["user_input"]
    result = intent_extract_chain.invoke({"user_input": user_input})
    return {
        **state,
        "location": result["location"],
        "conditions": result["conditions"],
        "condition_food_map": result["condition_food_map"]
    }

# 테스트 노드
@log_node_outputs("test_node",
                  include_keys=["intent","run_id"])
def test_node(state: OverallState) -> OverallState:
    user_input = state["user_input"]
    return {
        **state,
        "intent": "test"
    }

# 키워드 스코어링 노드
@log_node_outputs("keywords_rank_node",
                  include_keys=["food_scores","run_id"],
                  max_items=10)
def keywords_rank_node(state: OverallState) -> OverallState:
    user_input = state["user_input"]
    food_map = state["condition_food_map"]

    all_keywords = list({food for foods in food_map.values() for food in foods})

    prompt = f"""
당신은 숙련된 요리 추천 가이드 입니다.
사용자의 요청과 음식리스트를 받아 사용자가 가장 좋아할 법한 음식들을 추천해주어야 합니다.

아래는 사용자의 요청입니다:
[사용자 요청]
{user_input}

아래 음식 키워드들을 이 요청에 어울리는 순서대로 스코어링 해주세요.
가장 점수가 높은 음식 5개를 추출해주세요.

[음식 목록]
{chr(10).join(f"- {food}" for food in all_keywords)}

[출력 형식]
정렬된 음식 이름 목록과 스코어 점수를 출력해주세요.
출력은 반드시 JSON 형식만 출력하세요. 
코드블럭 (```json 등)은 사용하지 마세요.
그 외의 말은 절대 하지 마세요.

다음과 같이 출력하세요. 오직 JSON 형식만 출력하고 코드블럭은 사용하지 마세요.
{{
"음식1":10,
"음식2":8,
...
"음식N":3
}}
    """

    llm = ChatOpenAI(model="gpt-4o-mini")
    result = llm.invoke(prompt)
    food_scores = json.loads(result.content.strip())

    return {
        **state,
        "food_scores": food_scores
    }

# 검색 쿼리 생성 노드
@log_node_outputs("query_make_node",
                  include_keys=["query_list","run_id"],
                  max_items=5)
def query_make_node(state: OverallState) -> OverallState:
    food_scores = state["food_scores"]
    location = state["location"]
    query_list = [f"{location} {food}" for food in food_scores.keys()]
    return {
        **state,
        "query_list": query_list
    }

# 외부 MCP(n8n) 호출 노드
@log_node_outputs("run_mcp_node",
                  include_keys=["search_results","run_id"],
                  max_str=300, max_items=3)  # 응답이 크면 요약
def run_mcp_node(state: OverallState) -> OverallState:
    query_list = state["query_list"]
    url = "http://localhost:5678/webhook/76b4d5d4-57a9-46af-ae0c-66fa0fcc3e46"

    try:
        response = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            json={"queries": query_list}
        )
        result = response.json()
        print("n8n 응답:", result)

        return {
            **state,
            "search_results": result
        }
    except Exception as e:
        return state

# 결과 생성 노드
@log_node_outputs("result_make_node",
                  include_keys=["final_recommendations","run_id"],
                  max_str=500)
def result_make_node(state: OverallState) -> OverallState:
    search_results = state["search_results"]
    user_input = state["user_input"]

    prompt = f"""
당신은 숙련된 식당 추천 어시스턴트입니다.  
사용자의 요청과 아래 음식점 리스트를 참고하여 **Top 3 장소**를 추천해주세요.  
반드시 사용자의 상황과 요청 의도를 반영해서 선택해주세요.

답변은 마크다운 형식으로 예쁘게 정리해주세요.

---

[사용자 요청]
{user_input}

---

[음식점 리스트]
아래는 음식점 정보입니다. JSON 형태로 구성되어 있습니다.
```json
{search_results}
"""

    llm = ChatOpenAI(model="gpt-4o-mini")
    result = llm.invoke(prompt)

    return {
        **state,
        "final_recommendations": result.content
    }
