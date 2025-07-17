### 상태정의
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

class InputState(TypedDict):
    # 메시지 정의(list type 이며 add_messages 함수를 사용하여 메시지를 추가)
    user_input: Annotated[list, add_messages]

class stockvalueState(TypedDict):
    # 주식 가치를 계산하기 위해 필요한 정보 정의(tool 활용 값 검색)
    DCF: float # Discounted Cash Flow, 할인된 현금 흐름)
    DDM: float # Dividend Discount Model, 배당 할인 모형
    PER: float # Price-to-Earnings Ratio, 주가수익비율
    PBR: float # Price-to-Book Ratio, 주가순자산비율

