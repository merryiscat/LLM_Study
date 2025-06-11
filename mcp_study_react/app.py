from State import *
from Mcp_Tool import *
from Node import *
from Graph import *

# API KEY를 환경변수로 관리하기 위한 설정 파일
from dotenv import load_dotenv

# API KEY 정보로드
load_dotenv()

test_input = {
    "start_input": "비오는 날 수진역 혼자 저녁 먹을 식당 추천해줘"
}

graph_app = Project_Graph()

result = graph_app.invoke(
    test_input,
    config={"thread_id": "run-001"}  # ✅ thread_id 명시
)

print("✅ 결과:", result)
print("입력 디버깅:", test_input)