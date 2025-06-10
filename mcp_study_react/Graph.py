# 필요 라이브러리 임포트
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from State import *
from Node import *

def Project_Graph():
    memory = MemorySaver()
    graph = graph_builder.compile(checkpointer=memory)
    return graph

# 1. 그래프 인스턴스 생성
memory = MemorySaver()
graph_builder = StateGraph(OverallState, ReActState, input=InputState, output=EndState)

# 노드 등록
graph_builder.add_node("input_node", user_input_node)
graph_builder.add_node("react_reasoning", react_reasoning_node)
graph_builder.add_node("query_generation", user_query_node)
graph_builder.add_node("search", search_node)
graph_builder.add_node("summarize", summarize_node)
graph_builder.add_node("re_query", re_query_node)

# 흐름 연결
graph_builder.add_edge(START, "input_node")
graph_builder.add_edge("input_node", "react_reasoning")
graph_builder.add_edge("react_reasoning", "query_generation")
graph_builder.add_edge("query_generation", "search")
graph_builder.add_conditional_edges(
    "search", 
    result_check_node, 
    {"Ok_result": "summarize", "No_search": "re_query"}
)
graph_builder.add_edge("re_query", "search")
graph_builder.add_edge("summarize", END)

# 그래프 컴파일
graph = graph_builder.compile()

##### edges.py에서 Graph Export #####
def Project_Graph():
    graph = graph_builder.compile(checkpointer=memory)
    return graph