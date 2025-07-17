# 시작노드 - 확인 필요한 주식명을 user_input으로 받는 단계
def user_input_node(state: InputState):
    print("================================= Stock value Calculator =================================")
    print("주식의 적정가치를 계산합니다. 궁금하신 주식명을 말씀해주세요.")
    # time.sleep(1)
    user_input = input("User: ")
    
    return {"messages": [("user", user_input)], "tools_call_switch": True}


# 주가계산 노드 - 주식의 적정가치를 계산하기 위해 필요한 상태 값이 전부 들어왔는지 확인하고, 적절한 tool을 활용하여 필요한 상태 값을 채워넣는 단계
def stock_value_calculation_node(state: stockvalueState):