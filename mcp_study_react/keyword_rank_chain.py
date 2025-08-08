from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from typing import TypedDict, Dict

# output 결과 형식 정의
class KeywordsrankOutput(TypedDict):
    Dict[str, int]

# 텍스트 파일에서 프롬프트 로딩
def load_prompt_from_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()
    
# 프롬프트 파일 경로
prompt_text = load_prompt_from_file("chain/prompt/keywords_rank_prompt.txt")
keywords_rank_prompt = ChatPromptTemplate.from_template(prompt_text)

keyword_rank_parser = JsonOutputParser(pydantic_object=KeywordsrankOutput)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
keyword_rank_chain = keywords_rank_prompt | llm | keyword_rank_parser