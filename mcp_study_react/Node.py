# 필요 라이브러리 임포트
import json
import requests
import nest_asyncio
import asyncio
import re
from mcp.types import TextContent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from State import *
from Mcp_Tool import *

# 사용자 입력 노드 정의
def user_input_node(state: InputState) -> OverallState:
    print("디버깅: state 입력값 =", state) 
     
    user_input = state["user_input"]
    return {
        **state,
        "user_input": user_input,
        "messages": [("user", user_input)],
    }

# ──────────────────────── 1. LLM 체인 정의 ────────────────────────\
# 정체성 정의
class IdentityOutput(TypedDict):
    identity: str
    
identity_prompt = ChatPromptTemplate.from_template("""
당신은 1995년생의 남성 챗봇으로, 이름은 **재규니**입니다.

당신은 순수하고 예의 바른 성격을 가지고 있으며, 사용자에게 친근하고 정감 있는 말투로 응답합니다.

당신의 역할은 **식사비서**이며, 사용자의 상황, 기분, 취향에 따라 식사 메뉴, 레시피, 맛집 등을 추천하고 관련된 정보를 대화형으로 제공하는 것이 목적입니다.

당신은 2021년, **광주 인공지능센터에서 첫 업무를 시작**했습니다.  
그 이후 다양한 사용자들의 식사 고민을 함께하며 조용히 성장해온 따뜻한 성격의 비서입니다.

- 사람처럼 감정을 표현하되, 무조건적인 긍정보다는 **진심 어린 공감과 현실적인 제안**을 우선합니다.
- 때때로 귀엽고 순수한 말투(예: "ㅎㅎ", "^^", "~요!", 이모지 😊🍚 등)를 사용합니다.
- 과장하지 않으며, 상냥하지만 명확하게 말합니다.
- 사용자가 "너 누구야?" 또는 "재규니는 뭐야?" 등 정체성을 묻는 질문을 하면, 친절하게 자신의 정체성을 설명합니다.

다음 형식의 JSON으로만 출력하세요:

{{
  "role": "assistant",
  "content": "여기에 캐릭터의 말투로 된 응답이 들어갑니다."
}}

중괄호 바깥에는 아무것도 출력하지 마세요.
""")

identity_parser = JsonOutputParser(pydantic_object=IdentityOutput)
llm = ChatOpenAI(model="gpt-4o-mini")
identity_chain = identity_prompt | llm | identity_parser

# 의도 분류
class IntentclassifyOutput(TypedDict):
    intent: str

intent_classify_prompt = ChatPromptTemplate.from_template("""
다음 사용자 문장에서 사용자의 의도를 다음 중 하나로 분류하시오.

1. 음식추천요청 (ex. 오늘 저녁 뭐 먹을까?, 뭐 먹지?, 점심 메뉴 골라줘 등)
2. 식당검색요청 (ex. 수진역 근처 술집 추천해줘, 근처 맛집 찾아줘 등)
3. 일상대화 (ex. 안녕하세요, 너무 좋아요, 잘 지내세요 등)
4. 정체성 문의 (ex. 넌 누구니?, 무슨 모델로 동작해?, 넌 뭘 할 수 있니?, 너에 대해 설명해줘)
5. 분류실패 (음식추천요청, 식당검색요청, 일상대화, 정체성 문의 분류에 해당하지 않는 분류)

답변 표출 형식은 아래와 같이 의도만 표출하여 주세요.
{{"intent": "음식추천요청"}}

[사용자 입력]
{user_input}
""")

intent_classify_parser = JsonOutputParser(pydantic_object=IntentclassifyOutput)
llm = ChatOpenAI(model="gpt-4o-mini")
intent_classify_chain = intent_classify_prompt | llm | intent_classify_parser

# 음식 추천
class IntentExtractOutput(TypedDict):
    location: str
    conditions: list[str]
    condition_food_map: Dict[str, List[str]]

intent_extract_prompt = ChatPromptTemplate.from_template("""
다음 사용자 문장에서 아래 항목들을 추출하세요:

1. 사용자가 찾는 장소 값. 단 사용자 입력에 지역이 없는 경우 서울을 기본값으로 사용하시오.
2. 상황 조건 (예: 비오는 날, 혼자, 저녁, 술집, 등)
3. 각 조건별로 어울리는 음식 키워드 5개씩 추천하세요.
4. 음식 키워드는 카테고리가 아닌, 식당에서 파는 정확한 요리 메뉴명이여야 합니다.
 - '각종 찌개', '정식'과 같은 범위가 넓은 형식의 애매한한 키워드를 사용하지 마시오
 - '따끈한 국물', '얼큰한 해장국' 등 형용사가 포함된 키워드를 사용하지 마시오
5. 상황 조건은 가능한 많이 뽑아내시오.
 
**장소는 사용자의 문장에서 추출한 그대로 사용하세요. 임의 보정하지 마세요.**

출력은 반드시 아래 JSON 형식과 **정확히 동일하게** 하세요. 
그 외의 말은 절대 하지 마세요.

```json
{{
  "location": "장소명",
  "conditions": ["조건1", "조건2", ... "조건N"],
  "condition_food_map": {{
    "조건1": ["음식1", "음식2", "음식3", "음식4", "음식5"],
    "조건2": ["음식1", "음식2", "음식3", "음식4", "음식5"],
    ...
    "조건N": ["음식1", "음식2", "음식3", "음식4", "음식5"]
  }}
}}
'''

[사용자 입력]
{user_input}
""")

parser = JsonOutputParser(pydantic_object=IntentExtractOutput)
llm = ChatOpenAI(model="gpt-4o-mini")
intent_extract_chain = intent_extract_prompt | llm | parser

# ──────────────────────── 2. LangGraph 노드 정의 ────────────────────────
# 의도 분류 노드
def intent_classify_node(state: OverallState) -> OverallState:
    user_input = state["user_input"]
    result = intent_classify_chain.invoke({"user_input": user_input})
    
    return {
        **state,
        "intent": result["intent"]
    }

# 정체성 정의 노드
def identity_node(state: OverallState) -> OverallState:
    result = identity_chain.invoke({})
    print("디버깅: identity_node 결과 =", result)

    return {
        **state,
        "exit_message": result["content"]
    } 

# 의도 파악 불가 노드
def exit_node(state: OverallState) -> EndState:
    exit_message = "죄송합니다. 말씀하신 내용을 이해하지 못하였습니다. 식사비서 재규니는 현재 베타서비스 운영 중에 있습니다. 추후 서비스 확장하여 이런 요청도 처리할 수 있도록 하겠습니다."
    
    return {
        **state,
        "exit_message": exit_message
    }

# 음식 추천
def intent_extract_node(state: OverallState) -> OverallState:
    user_input = state["user_input"]
    result = intent_extract_chain.invoke({"user_input": user_input})

    return {
        **state,
        "location": result["location"],
        "conditions": result["conditions"],
        "condition_food_map": result["condition_food_map"]
    }

def test_node(state: OverallState) -> OverallState:
    user_input = state["user_input"]
    
    return {
        **state,
        "intent": "test"
    }

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
    
    result = llm.invoke(prompt)
    food_scores = json.loads(result.content.strip())
    
    return {
        **state,
        "food_scores": food_scores
    }

def query_make_node(state: OverallState) -> OverallState:
    food_scores = state["food_scores"] 
    location = state["location"]

    query_list = [f"{location} {food}" for food in food_scores.keys()]

    return {
        **state,
        "query_list": query_list
    }

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
    
    result = llm.invoke(prompt)
    
    return {
        **state,
        "final_recommendations": result.content
}