# 필요 라이브러리 임포트
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from State import *
from Node import *

def Project_Graph():
    memory = MemorySaver()
    builder = StateGraph(OverallState)

    builder.add_node("user_input_node", user_input_node)
    builder.add_node("intent_extract", intent_extract_node)

    builder.set_entry_point("user_input_node")
    builder.add_edge("user_input_node", "intent_extract")
    builder.set_finish_point("intent_extract")

    app = builder.compile(checkpointer=memory)  # ✔️ 여기서만 컴파일
    return app

# 앱 인스턴스 만들기
graph_app = Project_Graph()