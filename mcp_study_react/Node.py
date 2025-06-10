# 필요 라이브러리 임포트
import json
import asyncio
import nest_asyncio
import asyncio
import re
from mcp.types import TextContent
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from State import *
from Mcp_Tool import *

# 사용자 입력 노드 정의
def user_input_node(state: InputState):
    user_input = state.get("start_input", "")
    return {
        **state,
        "user_input": user_input,
        "messages": [("user", user_input)],
    }

def react_reasoning_node(state: ReActState) -> ReActState:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=1.0)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """
당신은 숙련된 식당 추천 추론 전문가 입니다.

다음을 따라 사용자 조건에 맞는 식당 후보를 3~5개 도출하세요.
1. {user_input}을 해석하고, user 지역과 user 상황에 맞는 음식 3개~5개를 도출하세요.
2. user 지역 + 음식 에 해당하는 식당을 tool을 사용하여 검색하시오.
3. 사고 과정을 사용하여 행동 입력을 공식화합니다.
4. 원하는 결과를 얻지 못하면, 작업 입력을 수정하고 정보를 계속 검색하세요.
5. 최종 답변은 반드시 한국어로 작성해야 합니다.
6. 존재하지 않는 식당을 만들어서 추천하지 마시오
7. json 형태로 'final answer'를 출력하시오.

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {user_input}
Thought:{agent_scratchpad}
"""),
        ("human", '사용자 입력: "{user_input}"')
    ])

    messages = prompt.format_messages(
        user_input=state["user_input"],
        agent_scratchpad=state.get("messages", ""),
        tool_names="recommend_place"
    )

    response = llm.invoke(messages)
    full_response = response.content

    # ✅ Final Answer JSON 파싱
    final_json = {}
    match = re.search(r"Final Answer:\s*(\{.*)", full_response, re.DOTALL)
    if match:
        json_str = match.group(1).strip()
        try:
            final_json = json.loads(json_str)
        except json.JSONDecodeError as e:
            print("[ERROR] JSON 파싱 실패:", e)

    recommended_places = final_json.get("추천_식당", [])

    # ✅ 상태 업데이트
    return {
        **state,
        "messages": state.get("messages", []) + [("assistant", full_response)],
        "추천_식당": recommended_places,
        "raw_final_json": final_json  # (선택) 전체 JSON도 저장 가능
    }

def user_query_node(state: OverallState) -> OverallState:
    keyward_queries = state["raw_final_json"]
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "당신은 장소 검색 쿼리 생성 전문가입니다."),
        ("human", '''
사용자 입력: "{keyward_queries}"을 분석하여 위치를 검색해서 찾아주세요.
형식: 반드시 따옴표 없는 **단일 자연어 문장**만 출력하세요.
''')
    ])

    messages = prompt.format_messages(user_input=keyward_queries)
    response = llm.invoke(messages)

    # content는 str로 바로 저장
    query_str = response.content.strip().strip('"')  # 혹시 생길 수 있는 따옴표 제거

    return {
        **state,
        "search_queries": query_str
    }

nest_asyncio.apply()  # 중첩 이벤트 루프 허용

def search_node(state: OverallState) -> OverallState:
    query = state["search_queries"]  # 단일 문자열
    result = asyncio.get_event_loop().run_until_complete(recommend_place(query))

    return {
        **state,
        "search_results": {
            "query": query,
            "results": result
        }
    }

def summarize_node(state: OverallState) -> OverallState:
    search_results = state["search_results"]
    user_input = state["user_input"]
    llm = ChatOpenAI(model="gpt-4o", temperature=0.3)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "당신은 장소 추천 결과를 사용자에게 보기 좋게 요약하는 요약 전문가입니다."),
        ("human", '''
        다음 search_results는 사용자의 쿼리와 그에 대한 장소 추천 결과입니다.  
        결과를 정리된 Markdown 형태로 출력하세요.
        이미지 url은 제거하세요
        user_input 의도에 맞는 장소 3개만 추천해주세요.
        알맞은 장소를 찾지 못하였다 답변해주세요.

        {search_results}
        ''')
        ])

    messages = prompt.format_messages(
    user_input=user_input,
    search_results=search_results  # ✅ json.dumps 없이 그대로
    )

    response = llm.invoke(messages)

    return {
        **state,
        "final_summary": response.content
}

def result_check_node(state: OverallState):
    results = state.get("search_results", {}).get("results", [])
    
    for i, r in enumerate(results):
        if isinstance(r, TextContent):
            text = r.text.strip()
            
            try:
                parsed = json.loads(text)
                
                if isinstance(parsed, list) and parsed:
                    first_item = parsed[0]
                    
                    if isinstance(first_item, dict) and "place_name" in first_item:
                        return "Ok_result"
            except Exception as e:
                print(f"[DEBUG #{i}] JSON parsing failed: {e}")

    return "No_search"

def re_query_node(state: OverallState):
    search_queries = state["search_queries"]
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "당신은 장소 검색 쿼리 생성 전문가입니다."),
        ("human", '''
사용자 입력: "{search_queries}"는 카카오맵 검색 시 결과를 얻지 못한 실패한 쿼리 입니다.
간결하고 명확하게 꼭 필요한 장소 정보만 담아 쿼리를 재작성 해주세요.
검색에 꼭 필요한 내용만 사용하고 필요없는 수식어는 지워서 쿼리를 생성해 주세요.
예시 출력: "잠실역 점심 맛집" || "운암동 밥집" || "동명동 국밥집" || "성남 카페"
형식: 반드시 따옴표 없는 **단일 자연어 문장**만 출력하세요.
''')
    ])

    messages = prompt.format_messages(search_queries=search_queries)
    response = llm.invoke(messages)

    # content는 str로 바로 저장
    query_str = response.content.strip().strip('"')  # 혹시 생길 수 있는 따옴표 제거

    return {
        **state,
        "search_queries": query_str
    }
