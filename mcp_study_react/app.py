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
    "user_input": "아 피곤하다",
    "thread_id": "run-0001" 
})

print("결과:", result)
