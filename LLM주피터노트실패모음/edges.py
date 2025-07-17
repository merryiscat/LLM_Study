######## edges.py ########
from states import *
from nodes import *
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.tools.tavily_search import TavilySearchResults

# 라우팅을 위한 함수
def select_next_node(state: SearchQueryState):
    if state["is_revise"]:
        return "is_revise"
    
    return '__end__'

def simple_route(state: PersonaState):
    """
    Simplery Route Tools or Next or retrieve
    """
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0 and ai_message.tool_calls[0]["name"] == "tavily_search_results_json":
        # print("Tavily Search Tool Call")
        return "tools"
    elif hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0 and ai_message.tool_calls[0]["name"] == "retrieve_trends":
        # print("Retrieve Call")
        return "retrieve"

    return "next"

def retrieve_route(state: PersonaState):
    """
    RAG Need Check?
    """
    if state['retrieve_check']:
        return "rewrite"

    return "return"

tool_node, retrieve = tool_nodes_exporter()

# 추가적인 필요사항 정리하고 그래프 빌딩
memory = MemorySaver()
graph_builder = StateGraph(OverallState, input=InputState, output=EndState)

graph_builder.add_node("User Input", user_input_node)
graph_builder.add_node("Character Make", character_make_node)
graph_builder.add_node("Character Retrieve Check", retrieve_check_node)
graph_builder.add_node("Rewrite Tool", rewrite_node)
graph_builder.add_node("Rewrite-Search", rewrite_search_node)
graph_builder.add_node("Persona Setup", persona_setup_node)
graph_builder.add_node("Search Sentence", search_setence_node)
graph_builder.add_node("Query Check", query_check_node)
graph_builder.add_node("Query Revise Tool", query_revise_node)
graph_builder.add_node("Tavily Search Tool", tool_node)
graph_builder.add_node("RAG Tool", retrieve)

graph_builder.add_edge(START, "User Input")
graph_builder.add_edge("User Input", "Character Make")
graph_builder.add_edge("Tavily Search Tool", "Character Make")
graph_builder.add_edge("RAG Tool", "Character Retrieve Check")
graph_builder.add_edge("Rewrite Tool", "Rewrite-Search")
graph_builder.add_edge("Rewrite-Search", "Character Make")
graph_builder.add_edge("Persona Setup", "Search Sentence")
graph_builder.add_edge("Search Sentence", "Query Check")
graph_builder.add_edge("Query Revise Tool", "Query Check")
graph_builder.add_conditional_edges(
    "Query Check", 
    select_next_node, 
    {"is_revise": "Query Revise Tool", END: END}
)
graph_builder.add_conditional_edges(
    "Character Make",
    simple_route,
    {"tools": "Tavily Search Tool", "next": "Persona Setup", "retrieve": "RAG Tool"}
)
graph_builder.add_conditional_edges(
    "Character Retrieve Check", 
    retrieve_route, 
    {"rewrite": "Rewrite Tool", "return": "Character Make"}
)

##### edges.py에서 Graph Export #####
def Project_Graph():
    graph = graph_builder.compile(checkpointer=memory)
    return graph