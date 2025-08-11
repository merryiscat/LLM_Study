from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from typing import TypedDict, Literal

# 파서정의
class TalkOutput(TypedDict):
    role: Literal["assistant"]
    content: str

# 텍스트 파일에서 프롬프트 로딩
def load_prompt_from_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()
    
# 프롬프트 파일 경로
prompt_text = load_prompt_from_file("chain/prompt/talk_prompt.txt")
talk_prompt = ChatPromptTemplate.from_template(prompt_text)

talk_parser = JsonOutputParser(pydantic_object=TalkOutput)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=1.0)
talk_chain = talk_prompt | llm | talk_parser