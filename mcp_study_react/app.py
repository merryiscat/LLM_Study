from State import *
from Mcp_Tool import *
from Node import *
from Graph import *

# API KEY를 환경변수로 관리하기 위한 설정 파일
from dotenv import load_dotenv

# API KEY 정보로드
load_dotenv()

graph_app = Project_Graph()

result = graph_app.invoke({
    "user_input": "신논현 역 주변에서 저녁을 먹을껀데, 진짜 인생 맛집을 찾고 싶어어",
    "thread_id": "run-0001" 
})

print("결과:", result)
