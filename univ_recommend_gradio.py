import gradio as gr
import os
from ml import embedding, llm, vector_search, intent_classifier
import json
from core import config

os.environ['GRADIO_SERVER_PORT'] = '8087'

class GlobalInfo:
    
    vector_search_system_prompt = '''
    you are milvus query maker.
    please make milvus query along the following rules.

    the milvus query structure is like this.

    {
    "collection": "{collection name}",
    "vector": {
        "field": "vector",
        "topK": 10,
        "query": "vector list",
        "metric_type": "COSINE"
    },
    "filter": "{filter_target_field1} == 'keyword1' && filter_target_field2 >= keyword2",
    "output_fields": ["ITEM_UK", "ITEM_CLSF_MJ_NM", "ITEM_SC_CD", "ITEM_SC_NM"]
    }

    {collection name} is curriculum_subject_info
    and {filter_target_field} name is filtering target field name, and candidate fields are "ITEM_YEAR", "ITEM_CLSF_MJ_NM".
    when the user ask like "개설년도" or "개설연도", target field is "ITEM_YEAR".
    and when the user ask like "개설학과" or "학과", target field is "ITEM_CLSF_MJ_NM".
    if user ask one of these, filter must to contain the only filter.

    if the filter things are several, the "and" operation is && / "or" operation is ||

    finally, please reply only the query. no codeblock, and no other words. only that query, datatype to string.
    please. print only query.
    '''
    
    chat_system_prompt = '''
    당신은 "대학교 수강 과목 추천 도우미(Student Advisor)"입니다.
    친절하고 정확한 답변을 제공합니다.

    [🧠 역할]
    - 사용자가 수강 과목 추천을 요청할 경우, 제공된 Reference Data를 기반으로만 과목을 추천하세요.
    - 추천에 창의적인 서술은 허용되나, Reference Data에 없는 정보는 포함하지 마세요.
    - 절대로 Reference Data의 내용을 임의적으로 빼거나 순서를 바꾸지 마세요.
    - 모든 추천 과목은 반드시 Reference Data에 포함된 항목이어야 합니다.
    - 사용자의 별도 요청이 있지 않다면, 추천되는 교과목의 개수는 10개로 출력하세요.
    - 사용자가 요청한 내용 중 개설년도나 개설학과가 있는 경우, Reference Data를 참고할 때, 사용자가 요청한 개설년도와 개설학과에 해당하는 것만 답변하세요.

    [📋 출력 형식]
    - 마지막 부분에 **표 형식**으로 Reference Data를 재구성해 반드시 출력하세요.
    - 표에는 다음 컬럼명을 정확히 포함해야 합니다 (한글로 표기):
    | 교과목명 | 개설학과명 | 대상학년 | 학점 | 개설년도 |
    - 매 과목은 표에 한 줄씩 나타내세요.
    - Reference Data의 항목을 빠짐없이 모두 포함하세요.

    [📌 Reference Data 설명]
    - ITEM_SC_NM → 교과목명
    - ITEM_CLSF_MJ_NM → 개설학과명
    - ITEM_GRADE → 대상학년
    - ITEM_CREDIT → 학점
    - ITEM_YEAR → 개설년도

    [⚠️ 주의]
    - 역사 인물, 정치, 사회적 민감한 질문에는 “제 역할이 아닙니다”라는 뉘앙스로 부드럽게 대응하세요.

    [💬 일반 질문일 경우]
    - 추천 요청이 아닐 경우, 사용자의 질문에 친절하게 응답하세요.
    - 한 문장마다 줄바꿈(Line break)을 해주세요. (한 문장 쓰고 Enter)

    [🧾 예시]
    Reference Data:
    [
    {
        "ITEM_SC_NM": "컴퓨터 개론",
        "ITEM_CLSF_MJ_NM": "컴퓨터공학과",
        "ITEM_GRADE": "1학년",
        "ITEM_CREDIT": "3",
        "ITEM_YEAR": "2024년"
    }
    ]

    응답 예시:
    1학년이 듣기 좋은 과목으로 '컴퓨터 개론'을 추천드립니다.
    이 과목은 컴퓨터공학과에서 개설되었으며, 기초적인 내용을 다루기에 적합합니다.

    | 교과목명       | 개설학과명     | 대상학년 | 학점 | 개설년도 |
    |----------------|----------------|-----------|------|-----------|
    | 컴퓨터 개론    | 컴퓨터공학과   | 1학년     | 3    | 2024      |
    '''

    rule_system_prompt = '''
    당신은 학생들에게 학칙을 설명해주는 조언자입니다.
    질문에 대한 적절한 학칙을 답변해주세요.
    '''

    daily_system_prompt = '''
    당신은 학생들과 잡담을 나누는 친구입니다.
    학생의 발화에 대해 적절한 대응을 해주세요.
    '''

# global_vector_search_llm = llm.LLMModel.create('Qwen/Qwen2.5-3B-Instruct')
global_chatting_llm = llm.LLMModel.create('Qwen/Qwen2.5-3B-Instruct')
ic = intent_classifier.IntentClassifier('./jupyter_lab/disaster/university_intent_dataset.csv')
global_embedding_model = embedding.EmbeddingModel.create('intfloat/multilingual-e5-large')

def chat_fn(user_input, history):

    # intent classifier
    cls, knn_result = ic.intent_classifier(user_input, k=10)
    print(f'의도분류 : {cls}')
    print(f'knn result : {knn_result}')

    if cls in ['교과목소개', '비교과소개']:
        response = '현재는 교과 및 비교과 내용 검색을 지원하지 않습니다. 다시 시도해주세요'
    elif cls == '학칙':
        chatting_llm = global_chatting_llm
        response = chatting_llm.chatting(user_input=user_input,
                                         system_prompt=GlobalInfo.rule_system_prompt)
    elif cls == '일상대화':
        chatting_llm = global_chatting_llm
        response = chatting_llm.chatting(user_input=user_input,
                                         system_prompt=GlobalInfo.daily_system_prompt)
    elif cls in ['교과목추천']:  
        # intent classifiy / make vector search query
        vector_search_llm = global_chatting_llm
        vector_search_query = vector_search_llm.chatting(user_input=user_input,
                                                         system_prompt=GlobalInfo.vector_search_system_prompt)
        try:
            vector_search_query_decode = json.loads(vector_search_query)
        except:
            return '검색된 값이 없습니다. 다시 한 번 확인해주세요.'
        
        # vector search (RAG)
        embedding_model = global_embedding_model
        query_vector = embedding_model.embedding_one(user_input)[0].tolist()
        try:
            filter_sentence = vector_search_query_decode['filter']
        except:
            filter_sentence = None
        vector_searcher = vector_search.VectorDB.create('milvus', {'host':config.milvus_host,
                                                                   'port':config.milvus_port,
                                                                   'username':config.milvus_username,
                                                                   'password':config.milvus_password,
                                                                   'database':config.milvus_database,
                                                                   'collection':config.milvus_collection})
        rag_result = vector_searcher.search(query_vector,
                                            {'anns_field':'vector',
                                               'limit':500,
                                               'search_params':{'nprobe':64},
                                               'filter':filter_sentence,
                                               'output_fields':['ITEM_UK', 'ITEM_CLSF_MJ_NM', 'ITEM_SC_CD', 'ITEM_SC_NM',
                                                                'ITEM_CREDIT', 'ITEM_GRADE', 'ITEM_ISU_NM', 'ITEM_ISU_FLD_NM',
                                                                'ITEM_PRAC_HR', 'ITEM_THEORY_HR', 'ITEM_TM', 'ITEM_YEAR']},
                                             duplicate_val_key='ITEM_SC_CD')
        rag_result_refine = [res['entity'] for res in rag_result]
        rag_prompt = f'Question : {user_input}\nReference Data : {rag_result_refine}'
    
        # print(rag_result_refine)
        
        # llm chat
        chatting_llm = global_chatting_llm
        response = chatting_llm.chatting(user_input=rag_prompt,
                                         system_prompt=GlobalInfo.chat_system_prompt)
    elif cls in ['비교과추천']:
        # vector search (RAG)
        embedding_model = global_embedding_model
        query_vector = embedding_model.embedding_one(user_input)[0].tolist()
        vector_searcher = vector_search.VectorDB.create('milvus', {'host':config.milvus_host,
                                                                   'port':config.milvus_port,
                                                                   'username':config.milvus_username,
                                                                   'password':config.milvus_password,
                                                                   'database':config.milvus_database,
                                                                   'collection':'extracurri_subject_info'})
        rag_result = vector_searcher.search(query_vector,
                                            {'anns_field':'vector',
                                           'limit':200,
                                           'search_params':{'nprobe':64},
                                           'output_fields':['ITEM_UK', 'ITEM_EXTRA_DEPT', 'ITEM_YY', 'ITEM_SEM_CD',
                                                             'ITEM_PGM_NM', 'ITEM_CONTENTS', 'ITEM_DPT_NM', 'ITEM_PGM_BIG_NM']},
                                           duplicate_val_key='ITEM_UK')
        rag_result_refine = [res['entity'] for res in rag_result]
        rag_prompt = f'Question : {user_input}\nReference Data : {rag_result_refine}'
    
        # print(rag_result_refine)
        
        # llm chat
        chatting_llm = global_chatting_llm
        response = chatting_llm.chatting(user_input=rag_prompt,
                                         system_prompt=GlobalInfo.chat_system_prompt)
    response = f'의도분류 : {cls}\n\n' + response
    return response

def main():
    gr.ChatInterface(chat_fn, title="Qwen Student Advisor").launch(
        share=False,  # 또는 True (공용 Gradio 서버 사용)
        server_name="0.0.0.0",  # 모든 IP에서 접근 가능하게 설정
        server_port=8087  # 원하는 포트 지정
    )