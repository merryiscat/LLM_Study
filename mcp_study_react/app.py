# streamlit_recipe_app.py

import streamlit as st
from streamlit_chat import message
from dotenv import load_dotenv
import os

from State import *
from Mcp_Tool import *
from Node import *
from Graph import *

# =============================
# 🌱 초기 설정
# =============================

# 환경 변수 로드 (OpenAI API 키 등)
load_dotenv()

# LangGraph 인스턴스 생성
graph_app = Project_Graph()

# 페이지 설정
st.set_page_config(page_title="식사비서 재규니", layout="wide", page_icon="")

# 챗봇 메시지 형식 출력
with st.chat_message("assistant"):
    st.markdown("안녕하세요! 저는 당신의 **식사비서 재규니**입니다. 무엇을 도와드릴까요?")

# 사이드바
with st.sidebar:
    st.header("식사비서 재규니")
    st.button("New Chat")

# =============================
# 💾 세션 상태 초기화
# =============================
if "messages" not in st.session_state:
    st.session_state.messages = []

# =============================
# ✨ 유틸 함수 정의
# =============================

def render_user_input(user_input: str):
    """유저 입력 메시지 출력"""
    st.chat_message("user").markdown(user_input)

def render_response(result: dict):
    """LangGraph 응답 처리 및 출력"""
    if "final_recommendations" in result:
        response = result["final_recommendations"]

        # 마크다운 줄바꿈 적용
        if isinstance(response, str):
            st.markdown(response.replace('\n', '  \n'))
        else:
            st.write(response)

    elif "exit_message" in result:
        # 의도 분류 실패 또는 종료 응답
        st.info(result["exit_message"] + "\n\n다른 방식으로 다시 말씀해 주세요 🙂")

    else:
        # 예외 처리
        st.error("❌ 추천 결과를 불러오는 데 실패했습니다.")
        with st.expander("📦 디버그 정보 보기"):
            st.json(result)

# =============================
# 💬 기존 대화 출력
# =============================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# =============================
# 🤖 챗봇 처리
# =============================

user_input = st.chat_input("오늘 뭐 먹을까?")

if user_input:
    # 유저 메시지 출력 및 저장
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        with st.spinner("채팅작성중.."):
            try:
                result = graph_app.invoke({
                    "user_input": user_input,
                    "thread_id": "run-ui-001"
                })

                if isinstance(result, dict):
                    # 응답 메시지 추출
                    if "final_recommendations" in result:
                        response = result["final_recommendations"]
                    elif "exit_message" in result:
                        response = result["exit_message"] + "\n\n다른 방식으로 다시 말씀해 주세요 🙂"
                    else:
                        response = "❌ 추천 결과를 불러오는 데 실패했습니다."

                    # 응답 출력 및 저장
                    if isinstance(response, str):
                        st.markdown(response.replace("\n", "  \n"))
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    else:
                        st.write(response)
                        st.session_state.messages.append({"role": "assistant", "content": str(response)})

                else:
                    st.error("❌ 시스템 오류: 올바른 결과를 반환하지 못했습니다.")

            except Exception as e:
                st.error("❌ LangGraph 실행 중 오류가 발생했습니다.")
                st.exception(e)
