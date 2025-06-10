# 필요 라이브러리 임포트
from typing import Annotated, List, Dict
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

######## states 정의 ########
class InputState(TypedDict):
    start_input: str

class ReActState(TypedDict):
    user_input: str
    messages: Annotated[list, add_messages]
    raw_final_json: str  # Action에서 추출된 퀴리
    search_results: dict
    final_summary: str

class OverallState(TypedDict):
    user_input: str
    messages: Annotated[list, add_messages]
    raw_final_json: str
    search_results: dict
    final_summary: str
    re_queries: str
    
class EndState(TypedDict):
    final_summary: str