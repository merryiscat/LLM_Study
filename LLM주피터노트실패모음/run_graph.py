####### run_graph.py #######
from .edges import Project_Graph

graph = Project_Graph()
config = {"configurable": {"thread_id": "1"}}

with open("graph_output.png", "wb") as f:
    f.write(graph.get_graph().draw_mermaid_png())
    
graph.invoke({"start_input": ""}, config=config)