# 필요 라이브러리 임포트
from typing import Annotated, List, Dict
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

######## states 정의 ########
class InputState(TypedDict):
    user_input: str

class OverallState(TypedDict):
    user_input: str
    intent: str
    location: str
    conditions: list[str]
    condition_weights: Dict[str, int]
    condition_food_map: Dict[str, List[str]]
    food_scores: Dict[str, int]
    top_foods: List[str]
    query_list: List[str]	
    search_results: List[Dict]
    summarized_places: List[Dict]
    final_recommendations: List[Dict]
    messages: Annotated[list, add_messages]
     
class EndState(TypedDict):
   final_recommendations: List[Dict]