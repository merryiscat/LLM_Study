from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from typing import TypedDict

# 의도 분류
class IntentclassifyOutput(TypedDict):
    intent: str

# 텍스트 파일에서 프롬프트 로딩
def load_prompt_from_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()
    
# 프롬프트 파일 경로
prompt_text = load_prompt_from_file("chain/prompt/intent_classify_prompt.txt")
intent_classify_prompt = ChatPromptTemplate.from_template(prompt_text)

intent_classify_parser = JsonOutputParser(pydantic_object=IntentclassifyOutput)
llm = ChatOpenAI(model="gpt-4o-mini")
intent_classify_chain = intent_classify_prompt | llm | intent_classify_parser
