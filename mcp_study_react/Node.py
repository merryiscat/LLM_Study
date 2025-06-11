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
    user_input = state.get("start_input", "")
    return {
        **state,
        "user_input": user_input,
        "messages": [("user", user_input)],
    }

# ──────────────────────── 1. LLM 체인 정의 ────────────────────────
class IntentExtractOutput(TypedDict):
    conditions: list[str]
    top_foods: list[str]

prompt = ChatPromptTemplate.from_template("...")
parser = JsonOutputParser(pydantic_object=IntentExtractOutput)
llm = ChatOpenAI(model="gpt-4o-mini")
intent_extract_chain = prompt | llm | parser

# ──────────────────────── 2. LangGraph 노드 정의 ────────────────────────
def intent_extract_node(state: InputState) -> OverallState:
    user_input = state["start_input"]
    result = intent_extract_chain.invoke({"user_input": user_input})
    return {
        "user_input": user_input,
        "conditions": result["conditions"],
        "top_foods": result["top_foods"]
    }