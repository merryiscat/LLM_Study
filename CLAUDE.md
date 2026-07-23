# LLM_Study - LLM 학습 및 실험 저장소

다양한 LLM 프레임워크와 기법을 탐구하는 학습/사이드프로젝트 모음.

## Tech Stack
- **Language**: Python 3.11+
- **Frameworks**: LangChain, LangGraph, MCP, Streamlit, Gradio, FastAPI
- **LLM**: OpenAI API
- **Vector DB**: ChromaDB
- **패키지관리**: uv (pyproject.toml)

## Project Structure
```
mcp_study_react/          MCP 기반 LangGraph 챗봇 (레스토랑 추천)
  Graph.py                그래프 워크플로우 (조건부 라우팅)
  Node.py                 10+ 노드 구현
  app/chain/              LangChain 체인들
TTD-DR/                   Self-Evolving Prompt 최적화
  self_evolution.py       진화 알고리즘 기반 프롬프트 개선
  app.py                  Streamlit 리포트 생성
LLM주피터노트실패모음/      RAG 챗봇 실험 (재무제표)
Json_excel_inverter/      JSON ↔ Excel 변환 유틸
intent_classifier.py      KNN 기반 의도 분류 (임베딩)
univ_recommend_gradio.py  대학 강좌 추천 (Gradio UI)
```

## Development Commands
```bash
cd LLM_Study
uv run python main.py                          # 메인 진입점
uv run streamlit run mcp_study_react/app.py     # MCP 챗봇
uv run streamlit run TTD-DR/app.py              # Self-Evolution
```

## Key Concepts
- **의도분류**: KNN + 다국어 임베딩
- **RAG**: ChromaDB + LangChain 문서 체인
- **에이전트**: LangGraph StateGraph 기반 조건부 워크플로우
- **Self-Evolution**: Judge→Revise 루프, 인구 기반 최적화
- **MCP**: Model Context Protocol 도구 통합
