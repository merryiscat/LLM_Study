# streamlit_recipe_app.py
import streamlit as st
from streamlit_chat import message
import openai
import os
from State import *
from Mcp_Tool import *
from Node import *
from Graph import *
from dotenv import load_dotenv

# 환경 변수 불러오기 (API 키 등)
load_dotenv()

# LangGraph 인스턴스 생성
graph_app = Project_Graph()

# 페이지 제목
st.set_page_config(page_title="식사비서 재규니", layout="wide")
st.title("안녕하세요! 저는 재규니입니다.")

# 사이드바
with st.sidebar:
    st.header("식사비서 재규니")
    st.button("New Chat")

# 유저 입력 받기
user_input = st.chat_input("오늘 뭐 먹을까?")

if user_input:
    # 사용자 메시지 표시
    st.chat_message("user").markdown(user_input)

    # LangGraph 실행
    with st.chat_message("assistant"):
        with st.spinner("맛집 찾는 중..."):
            result = graph_app.invoke({
                "user_input": user_input,
                "thread_id": "run-ui-001"
            })

            # 결과 확인 및 출력
            if isinstance(result, dict) and "final_recommendations" in result:
                response = result["final_recommendations"]

                # 문자열이 맞으면 마크다운으로 예쁘게 출력
                if isinstance(response, str):
                    # 마크다운 줄바꿈 맞춰서 출력
                    st.markdown(response.replace('\n', '  \n'))
                else:
                    # fallback
                    st.write(response)
            else:
                st.error("❌ 추천 결과를 불러오는 데 실패했습니다.")
                st.write("📦 디버그 정보:", result)