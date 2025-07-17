######## nodes.py ########
import os
import json
from states import *
from pydantic import BaseModel, Field
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.prebuilt import ToolNode
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.tools.retriever import create_retriever_tool

# API KEY를 환경변수로 관리하기 위한 설정 파일
from dotenv import load_dotenv

# API KEY 정보로드
load_dotenv()

# ChromaDB 로드
vectorstore = Chroma(
    collection_name="rag-chroma",
    embedding_function=OpenAIEmbeddings(),
    persist_directory="./chroma_db",
)
retriever = vectorstore.as_retriever()

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# 시작노드 - 페르소나에 대한 정보를 요구하는 노드임
def user_input_node(state: InputState):
    print("================================= Make Persona =================================")
    print("페르소나를 결정합니다. 성별, 나이, 거주지, 취미 등 정보를 알려주세요.")
    # time.sleep(1)
    user_input = input("User: ")
    
    return {"messages": [("user", user_input)], "tools_call_switch": True}

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# 노드 1 - 입력된 문장으로부터 새로운 페르소나를 만들어내는 노드.
# 검색용 Tavily 툴 로드하고 노드만듦.
tool = TavilySearchResults(max_results=3)
web_search_tool = TavilySearchResults(max_results=5)

# 노드 1-1. 검색용 노드
tool_node = ToolNode(tools=[tool])

# 검색용 RAG 툴 로드하고 노드만듦
retriever_tool = create_retriever_tool(
    retriever,
    "retrieve_trends",
    "Search for the latest trends in fashion and hobbies and return relevant information.",
)
# 노드 1-2. RAG용 노드.
retrieve = ToolNode([retriever_tool])

def tool_nodes_exporter():
    return tool_node, retrieve

# 두 개 툴 엮어서 리스트 만듦.
tools = [tool, retriever_tool]

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# 노드 1-3. RAG 검증노드
# 노드 1-2의 Tools Output을 받아서, User Input에 잘 맞는지 검증해서 Yes Or No로 대답함.
# 만약 Yes라면 그대로 다시 Character Make Node로 보내서 최종 답변을 생성하도록 하고
# 아니라면 검색을 진행하고 새로운 값을 받아서 보낼거임.

class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score:str = Field(..., description="Documents are relevant to the question, 'yes' or 'no'", enum=['yes', 'no'])

rag_check_model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
rag_check_model = rag_check_model.with_structured_output(GradeDocuments)

def retrieve_check_node(state: PersonaState):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """
            You are a consultation expert who provides appropriate information in response to user input.
            Return 'yes' or 'no' if you can provide an accurate answer to the user's question from the given documentation.
            If you can't provide a clear answer, be sure to return NO.
            """),
            ("human", "Retrieved document: \n\n {document} \n\n User's input: {question}"),
        ]
    )
    
    retrieval_msg = state['messages'][-1].content
    human_msg = state['user_input']
    retrieval_grader = prompt | rag_check_model
    response = retrieval_grader.invoke({"document": retrieval_msg, "question": human_msg})
    retrieve_handle = response.binary_score
    retrieve_check = False
    
    if retrieve_handle == "no":
        print("=============================== Need to Check ===============================")
        retrieve_check = True
    if retrieve_handle == "yes":
        print("============================== No Need to Check =============================")
        
    return {"retrieve_check": retrieve_check, "retrieval_msg": retrieval_msg}

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# 노드 1-4. 쿼리 재-작성 노드
# 노드 1-2에서 산출된 retrieve가 입력값과 적절하게 매치되지 않는 경우, 입력값을 수정하게 됨.
# state User_input 이용
# 이는 노드 1-3에서 yes를 반환하는 경우에 실행됨.

class Rewrite_Output(TypedDict):
    """
    Sturctured_output을 생성하기위한 클래스
    """
    query: Annotated[str, ..., "Rewritten query to find appropriate material on the web"]

rewrite_model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
rewrite_model = rewrite_model.with_structured_output(Rewrite_Output)

def rewrite_node(state: PersonaState):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """
            You're an expert in improving search relevance.\n
            Look at previously entered search queries and rewrite them to better find that information on the internet.
            """),
            ("human", "Previously entered search queries: \n{user_input}"),
        ]
    )
    
    user_input = state['user_input']
    rewrite_chain = prompt | rewrite_model
    response = rewrite_chain.invoke({"user_input": user_input})
    rewrited_query = response['query']
    print(f"================================ Rewrited Query ================================\nRewritted Query: {rewrited_query}")

    return {"rewrite_query": rewrited_query}

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# 노드 1-5. 재작성된 쿼리를 이용해서 인터넷 검색하는 노드

def rewrite_search_node(state: PersonaState):
    print("================================ Search Web ================================")
    docs = web_search_tool.invoke({"query": state['rewrite_query']})
    web_results = "\n\n".join([d["content"] for d in docs])
    web_results = web_results + "\n\n" + state['retrieval_msg']
    # print(web_results)

    new_messages = [ToolMessage(content=web_results, tool_call_id="tavily_search_results_json")]
            
    return {"messages": new_messages}

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# 노드 1번 작성된 것.
# 인간 입력이랑 Retrieve를 받을 수 있는 놈임.

character_model = ChatOpenAI(model="gpt-4o", temperature=0.2)
character_model_with_tools = character_model.bind_tools(tools)

def character_make_node(state: PersonaState):
    prompt = ChatPromptTemplate.from_messages([
        ("system","""
        You are an expert in creating characters for fiction.\n
        Whatever input the user is presented with, you must return a description of the completed character.\n
        If no information is available, randomly generate and return the character's attributes.\n
        Based on the values entered by the user, envision and present the character, including the character's age, gender, job, location, interests, hobbies and etc.\n
        The returned value must be in Korean.\n
        """),
        ("human", "Input: {human_input}\n Retrieve: {context}"),
    ])
    prompt_with_tools = ChatPromptTemplate.from_messages([
        ("system","""
        You are an expert in creating characters for fiction.\n
        Whatever input the user is presented with, you must return a description of the completed character.\n
        If no information is available, randomly generate and return the character's attributes.\n
        Based on the values entered by the user, envision and present the character, including the character's age, gender, job, location, interests, hobbies and etc.\n
        If you have difficulty creating an appropriate character, use an online search to solve the problem.\n
        The returned value must be in Korean.\n
        """),
        ("human", "Input: {human_input}\n Retrieve: {context}"),
    ])
    messages_list = state['messages']
    last_human_message = next((msg for msg in reversed(messages_list) if isinstance(msg, HumanMessage)), None).content
    last_msg = state['messages'][-1].content
    
    if last_human_message == last_msg:
        last_msg = ""
        print(f"==================================== INPUT ====================================\nHuman Input: {last_human_message}")
    else:
        try:
            last_msg_data = json.loads(state['messages'][-1].content)
            last_msg = "\n\n".join([d["content"] for d in last_msg_data])
        except:
            ...
        print(f"==================================== INPUT ====================================\nHuman Input: {last_human_message}\nContext: {last_msg}")
    
    if state['tools_call_switch']:
        chain_with_tools = prompt_with_tools | character_model_with_tools
        response = chain_with_tools.invoke({"human_input": last_human_message, "context": last_msg})
        
        if hasattr(response, "tool_calls") and len(response.tool_calls) > 0 and (response.tool_calls[0]["name"]) == "tavily_search_results_json":
            print("================================ Search Online ================================")
            tool_switch = False
        elif hasattr(response, "tool_calls") and len(response.tool_calls) > 0 and (response.tool_calls[0]["name"]) == "retrieve_trends":
            print("=============================== Search Retrieval ===============================")
            tool_switch = False
        else:
            print("============================= Chracter Information =============================")
            tool_switch = False
            print(response.content)
            
    else:
        chain = prompt | character_model
        response = chain.invoke({"human_input": last_human_message, "context": last_msg})
        print("============================= Chracter Information =============================")
        tool_switch = False
        print(response.content)

    return {"messages": [response], "user_input": last_human_message, "tools_call_switch": tool_switch}

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# 노드2 - 입력된 문장으로부터 페르소나에 관한 정보를 추출하고, 정보가 없는 경우 이를 채워넣는 노드.
class Persona_Output(TypedDict):
    """
    Sturctured_output을 생성하기위한 클래스
    """
    character_age: Annotated[str, ..., "An age of the Persona"]
    character_sex: Annotated[str, ..., "A sex of the Persona"]
    character_location: Annotated[str, ..., "A place where the persona might live"]
    character_interest: Annotated[str, ..., "Interests that the persona might have"]
    character_hobby: Annotated[str, ..., "Hobbies that the persona might have"]
    character_job: Annotated[str, ..., "Job that the persona might have"]
    character_information: Annotated[str, ..., "Additional information to describe the persona"]
    
persona_model = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)
persona_model = persona_model.with_structured_output(Persona_Output)

# 페르소나를 반환하는 매우 경직된 LLM.
# 정보가 없는 경우 임의의 값을 채워넣도록 되어있음.
def persona_setup_node(state: PersonaState):
    messages = [
        ("system", """
         You are the expert in determining your character's persona.
        Extract the character's 'age', 'sex', 'job', 'location', 'interest', and 'hobbies' from the values entered by the user.
        If no information is available, it will return a randomised set of appropriate information that must be entered.
        The returned value must be in Korean.
        """),
        ("human", state['messages'][-1].content)
    ]
    response = persona_model.invoke(messages)
    
    print("================================= Persona Setup =================================")
    print(f"성별: {response['character_sex']}")
    print(f"나이: {response['character_age']}")
    print(f"거주지: {response['character_location']}")
    print(f"흥미: {response['character_interest']}")
    print(f"취미: {response['character_hobby']}")
    print(f"직업: {response['character_job']}")
    print(f"추가정보: {response['character_information']}")
    
    return {"character_persona_dict": response}

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# 노드 3 - 페르소나를 토대로 적절한 검색 키워드를 생성하는 놈.

class Search_Output(TypedDict):
    """
    Sturctured_output을 생성하기위한 클래스
    """
    query_list: Annotated[list, ..., "List of queries that customers have entered in your shop"]

search_model = ChatOpenAI(model="gpt-4o")
search_model = search_model.with_structured_output(Search_Output)

examples = [
    {"input": 
        """
            User Sex: 여자,
            User Age: 20대,
            User Location: 서울 강남,
            User Interest: 최신 화장법,
            User Hobby: 공원 산책,
            User Job: 그래픽 디자이너,
            User Information: 강아지를 기르고 있음, 피부에 관심이 많음
        """, 
    "output": 
        ['피부진정용 필링패드', '수분에센스', '스틱형 파운데이션', '강아지 간식', '강아지용 배변패드', '강아지 장난감']
    },
]

example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("ai", "{output}"),
    ]
)

few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)

def search_setence_node(state: SearchQueryState):
    prompt = ChatPromptTemplate.from_messages([
        ("system","""
        You're a great marketing manager, and you're working on inferring customer search queries.
        Given the customer information, generate appropriate search quries that customers might enter to find products in your shopping mall.
        Make sure to clearly present the actual product names that a user with that persona would search for in your retail mall.
        """),
        few_shot_prompt,
        ("human", """
         User Sex: {sex},
         User Age: {age},
         User Location: {location},
         User Interest: {interest},
         User Hobby: {hobby},
         User Job: {job},
         User Information: {information}
         """),
    ])
    
    chain = prompt | search_model
    response = chain.invoke(
        {
            "sex": state['character_persona_dict']['character_sex'],
            "age": state['character_persona_dict']['character_age'],
            "location": state['character_persona_dict']['character_location'],
            "interest": state['character_persona_dict']['character_interest'],
            "hobby": state['character_persona_dict']['character_hobby'],
            "job": state['character_persona_dict']['character_job'],
            "information": state['character_persona_dict']['character_information'],
        }
    )
    print("=============================== Search Queries ===============================")
    print(response['query_list'])
    
    return {"query_list": response}

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# 노드 4, revise_tool - 반환된 서치쿼리가 적당한지 검증하는 노드임.
class QueryReviseAssistance(BaseModel):
    """Escalate the conversation. 
    Use only if the given search query is a strong mismatch with the customer's information.
    Use this tool even if given search query is seriously inappropriate to enter into the search bar of an online retailer like Amazon.
    Never call the tool if the same input is still being given as before.
    To use this function, return 'query_list'.
    """
    query_list: list
    
query_check_model = ChatOpenAI(model="gpt-4o", temperature=0.5, streaming=True)
query_check_model = query_check_model.bind_tools([QueryReviseAssistance])

def query_check_node(state: SearchQueryState):
    print("=============================== Query Check ===============================")
    prompt = ChatPromptTemplate.from_messages([
        ("system","""
        You are a search manager.
        If you think that the given customer's information and the search query that they used on your online store are relevant, then return the query as it is.
        Never invoke the tool if you are still being given the same query that was entered in the previous dialogue.
        """),
        ("human", """
            User Sex: {sex},
            User Age: {age},
            User Location: {location},
            User Interest: {interest},
            User Hobby: {hobby},
            User Job: {job},
            User Information: {information},
            Queries: {queries}
            """),
        ])
    chain = prompt | query_check_model
    
    response = chain.invoke(
        {
            "sex": state['character_persona_dict']['character_sex'],
            "age": state['character_persona_dict']['character_age'],
            "location": state['character_persona_dict']['character_location'],
            "interest": state['character_persona_dict']['character_interest'],
            "hobby": state['character_persona_dict']['character_hobby'],
            "job": state['character_persona_dict']['character_job'],
            "information": state['character_persona_dict']['character_information'],
            "queries": state['query_list']['query_list'],
        }
    )
    is_revise = False
        
    if (
        response.tool_calls
        and response.tool_calls[0]["name"] == QueryReviseAssistance.__name__
    ):
        print("Revise Requires")
        is_revise = True
    
    return {"messages": [response], "is_revise": is_revise}

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# 노드 4-1. 쿼리를 수정하도록 요청받은 경우 이를 수행하는 노드임.

class QueryCheck_Output(TypedDict):
    """
    Sturctured_output을 생성하기위한 클래스
    """
    query_list: Annotated[list, ..., "List of queries that customers might have entered in search-bar of your online retail shop"]
    
query_revise_model = ChatOpenAI(model="gpt-4o")
query_revise_model = query_revise_model.with_structured_output(QueryCheck_Output)

def query_revise_node(state: SearchQueryState):
    print("=============================== Query Revise ===============================")
    prompt = ChatPromptTemplate.from_messages([
        ("system",
            """
                You are a validator who fixes errors in a given query.
                From the list of queries given, remove or modify the queries that do not match the user's information appropriately.
                Be sure to delete highly irrelevant data.
                Be sure to remove search terms that you wouldn't use on a shopping site like Amazon.
                Return the modified queries as a list.
            """
        ),
        ("human", 
            """
                User Sex: {sex},
                User Age: {age},
                User Location: {location},
                User Interest: {interest},
                User Hobby: {hobby},
                User Job: {job},
                User Information: {information},
                Queries: {queries}
            """
        )])
    
    chain = prompt | query_revise_model
    response = chain.invoke(
        {
            "sex": state['character_persona_dict']['character_sex'],
            "age": state['character_persona_dict']['character_age'],
            "location": state['character_persona_dict']['character_location'],
            "interest": state['character_persona_dict']['character_interest'],
            "hobby": state['character_persona_dict']['character_hobby'],
            "job": state['character_persona_dict']['character_job'],
            "information": state['character_persona_dict']['character_information'],
            "queries": state['query_list']['query_list'],
        }
    )
    
    print(response['query_list'])
    
    return {"query_list": response, "is_revise": False}
    