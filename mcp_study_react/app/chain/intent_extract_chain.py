from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from typing import TypedDict, List, Dict

# 음식 추천
class IntentExtractOutput(TypedDict):
    location: str
    conditions: list[str]
    condition_food_map: Dict[str, List[str]]

# 텍스트 파일에서 프롬프트 로딩
def load_prompt_from_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()
    
# 프롬프트 파일 경로
prompt_text = load_prompt_from_file("chain/prompt/intent_extract_prompt.txt")
intent_extract_prompt = ChatPromptTemplate.from_template(prompt_text)

parser = JsonOutputParser(pydantic_object=IntentExtractOutput)
llm = ChatOpenAI(model="gpt-4o-mini")
intent_extract_chain = intent_extract_prompt | llm | parser