# 필요 라이브러리 임포트
import json
import asyncio
import nest_asyncio
import asyncio
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

    return {
        **state,
        "user_input": user_input,
        "messages": [("user", user_input)],
    }

def user_query_node(state: OverallState) -> OverallState:
    user_input = state["user_input"]
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "당신은 장소 검색 쿼리 생성 전문가입니다."),
        ("human", '''
사용자 입력: "{user_input}"을 분석하여 찾고 싶은 위치를 분석하여 검색 쿼리를 만들어주세요.
형식: 반드시 따옴표 없는 **단일 자연어 문장**만 출력하세요.
''')
    ])

    messages = prompt.format_messages(user_input=user_input)
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
    print(f"[DEBUG] Number of results: {len(results)}")

    for i, r in enumerate(results):
        if isinstance(r, TextContent):
            text = r.text.strip()
            print(f"[DEBUG #{i}] text[:100]: {repr(text[:100])}")

            try:
                parsed = json.loads(text)
                print(f"[DEBUG #{i}] Parsed JSON type: {type(parsed)}")

                if isinstance(parsed, list) and parsed:
                    first_item = parsed[0]
                    print(f"[DEBUG #{i}] Keys in first item: {list(first_item.keys())}")

                    if isinstance(first_item, dict) and "place_name" in first_item:
                        print(f"[DEBUG #{i}] Found place_name, returning Ok_result")
                        return "Ok_result"
            except Exception as e:
                print(f"[DEBUG #{i}] JSON parsing failed: {e}")

    print("[DEBUG] No valid result found, returning No_search")
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
