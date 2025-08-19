from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from typing import TypedDict, Literal

# 정체성 정의
class IdentityOutput(TypedDict):
    role: Literal["assistant"]
    content: str

# 텍스트 파일에서 프롬프트 로딩
def load_prompt_from_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()
    
# 프롬프트 파일 경로
prompt_text = load_prompt_from_file("chain/prompt/identity_prompt.txt")
identity_prompt = ChatPromptTemplate.from_template(prompt_text)

identity_parser = JsonOutputParser(pydantic_object=IdentityOutput)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=1.0)
identity_chain = identity_prompt | llm | identity_parser