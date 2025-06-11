# 필요 라이브러리 임포트
import json
import asyncio
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
    print("🔥 디버깅: state 입력값 =", state) 
     
    user_input = state["start_input"]
    return {
        **state,
        "user_input": user_input,
        "messages": [("user", user_input)],
    }

# ──────────────────────── 1. LLM 체인 정의 ────────────────────────
class IntentExtractOutput(TypedDict):
    location: str
    conditions: list[str]
    condition_food_map: Dict[str, List[str]]

prompt = ChatPromptTemplate.from_template("""
다음 사용자 문장에서 아래 항목들을 추출하세요:

1. 사용자가 찾는 장소 값. 단 사용자 입력에 지역이 없는 경우 서울을 기본값으로 사용하시오.
2. 상황 조건 (예: 비오는 날, 혼자, 저녁 등)
3. 각 조건별로 어울리는 음식 키워드 5개씩 추천하세요.
                                          
장소는 LLM이 임의로 보정하지 말고 **사용자 문장에서 추출한 그대로** 사용하세요.

출력은 반드시 JSON 형식으로 아래와 같이 반환하세요:

{{
  "location": "장소명",
  "conditions": ["조건1", "조건2", ...],
  "condition_food_map": {{
    "조건1": ["음식1", "음식2", ...],
    "조건2": ["음식1", "음식2", ...],
    ...
  }}
}}

[사용자 입력]
{{user_input}}
""")

parser = JsonOutputParser(pydantic_object=IntentExtractOutput)
llm = ChatOpenAI(model="gpt-4o-mini")
intent_extract_chain = prompt | llm | parser

# ──────────────────────── 2. LangGraph 노드 정의 ────────────────────────
def intent_extract_node(state: OverallState) -> OverallState:
    user_input = state["user_input"]
    result = intent_extract_chain.invoke({"user_input": user_input})

    return {
        **state,
        "location": result["location"],
        "conditions": result["conditions"],
        "condition_food_map": result["condition_food_map"]
    }