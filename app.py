import streamlit as st
from streamlit_chat import message
import openai
import os
from State import *
from Mcp_Tool import *
from Node import *
from Graph import *

# API KEY를 환경변수로 관리하기 위한 설정 파일
from dotenv import load_dotenv

# API KEY 정보로드
load_dotenv()

# Streamlit UI 설정
st.set_page_config(page_title="장소 추천 챗봇", layout="centered")
st.title("장소 추천 챗봇")

# 대화 기록 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

# LangGraph 인스턴스 초기화
graph = Project_Graph()
config = {"configurable": {"thread_id": "1"}}

# 사용자 입력 받기
user_input = st.chat_input("찾고 싶은 장소를 입력하세요!")

if user_input:
    # 사용자 입력 메시지 저장
    st.session_state.messages.append({"role": "user", "content": user_input})

    # LangGraph 실행
    with st.spinner("추천 장소를 찾는 중입니다..."):
        result = graph.invoke({"start_input": user_input}, config=config)
        final_summary = result.get("final_summary", "결과를 가져오지 못했습니다.")

    # 결과 메시지 저장
    st.session_state.messages.append({"role": "assistant", "content": final_summary})

# 메시지 렌더링
for i, msg in enumerate(st.session_state.messages):
    is_user = msg["role"] == "user"
    message(msg["content"], is_user=is_user, key=f"{i}-{msg['role']}")