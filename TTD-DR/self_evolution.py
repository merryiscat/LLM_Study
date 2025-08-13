"""
Self-Evolution (TTD-DR component) — Practical Python Template

What this file gives you
- End-to-end self-evolution loop you can drop into any agent stage (plan / search-question / answer / final-report).
- Pluggable LLM client (OpenAI example included; swap with your provider easily).
- Judge → Revise loop with JSON-robust parsing and merge step (cross-over).
- Sensible defaults for k variants, T episodes, population n.

How to use quickly
1) Set OPENAI_API_KEY in your env, or replace OpenAIClient with your stack.
2) Run the __main__ example to see a Stage 2b (answer synthesis) evolution using mock docs.
3) Call SelfEvolution.run_for_stage(...) from your agent code.

Author’s note
- Keep dependencies stdlib-only for portability. If you add aiohttp/async, you can parallelize calls.

Self-Evolution (TTD-DR 구성요소) — 실전용 파이썬 템플릿
이 파일이 제공하는 것
에이전트의 어떤 단계(계획 / 검색 질문 / 답변 / 최종 보고)에도 끼워 넣을 수 있는 엔드 투 엔드 self-evolution 루프

교체 가능한 LLM 클라이언트(OpenAI 예시 포함, 원하는 제공자로 쉽게 교체 가능)

Judge → Revise 루프(견고한 JSON 파싱)와 병합 단계(크로스오버)

k 변형 수, T 에피소드 수, population n에 대한 합리적인 기본값

빠르게 사용하는 법
환경 변수에 OPENAI_API_KEY를 설정하거나, OpenAIClient를 당신의 스택으로 교체하세요.

__main__ 예제를 실행해 **2b 단계(답변 합성)**가 모의 문서로 어떻게 진화하는지 확인하세요.

에이전트 코드에서 SelfEvolution.run_for_stage(...)를 호출하세요.

작성자 주
이식성을 위해 표준 라이브러리만 사용하세요. aiohttp/async 등을 추가하면 호출을 병렬화할 수 있습니다.
"""
from __future__ import annotations

import os
import json
import math
import random
import time
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Callable, Protocol

# =============================
# LLM Client Interface & OpenAI impl
# =============================
class LLMClient(Protocol):
    def generate(self, prompt: str, *, temperature: float = 0.7, top_p: float = 0.95, max_tokens: Optional[int] = None) -> str:  # pragma: no cover
        ...


class OpenAIClient:
    """Minimal OpenAI client wrapper (requires: pip install openai>=1.0.0)
    - Set env var OPENAI_API_KEY
    - Model default: gpt-4o-mini (change as needed)
    """

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        try:
            from openai import OpenAI  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("Install openai>=1.0.0 to use OpenAIClient (pip install openai)") from e
        self._OpenAI = OpenAI
        self._client = OpenAI()
        self._model = model

    def generate(self, prompt: str, *, temperature: float = 0.7, top_p: float = 0.95, max_tokens: Optional[int] = None) -> str:  # pragma: no cover
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""


# =============================
# Data structures
# =============================
@dataclass
class Candidate:
    text: str
    score: float = 0.0
    helpfulness: float = 0.0
    comprehensiveness: float = 0.0
    feedback: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvolutionConfig:
    variants_k: int = 4
    episodes_T: int = 3
    population_n: int = 2
    weight_helpfulness: float = 0.6
    weight_comprehensiveness: float = 0.4
    temperature_candidates: Tuple[float, ...] = (0.7, 0.9, 1.0, 1.1)
    top_p_candidates: Tuple[float, ...] = (0.9, 0.95, 0.97, 0.98)
    max_tokens: Optional[int] = None  # set if your provider needs a hard cap
    early_stop_delta: float = 0.25  # stop if improvement below this (absolute) for last episode


# =============================
# Prompt templates
# =============================

def prompt_generate_variant(stage: str, context: Dict[str, Any]) -> str:
    """Stage-specific generation prompt. Extend per your schema.
    stage in {"plan", "search_question", "answer", "report"}
    """
    if stage == "plan":
        return f"""
You are a research planner. Given the user query, produce a structured plan (sections + bullets).
- Be specific about evidence to gather, and list potential risks/open questions.
- Output in markdown.

<USER_QUERY>\n{context.get('user_query','')}\n</USER_QUERY>
"""
    elif stage == "search_question":
        return f"""
You are a query strategist. Using the current plan and draft history, propose a high-novelty search question.
- Avoid duplicates of past queries. Explain why this query advances coverage.
- Output JSON: {{"query": str, "rationale": str}}

<PLAN>\n{context.get('plan','')}\n</PLAN>
<PAST_QA>\n{json.dumps(context.get('past_qa', []), ensure_ascii=False)}\n</PAST_QA>
<REPORT_DRAFT>\n{context.get('report_draft','')}\n</REPORT_DRAFT>
"""
    elif stage == "answer":
        docs = context.get("retrieved_docs", [])
    # 번호 달아 나열 (모델이 [1], [2]로 인용하게)
        docs_md = "\n".join(
            f"[{i+1}] {d.get('title','')} ({d.get('url','')}) — {d.get('snippet','')}"
            for i, d in enumerate(docs) 
        )
        return f"""
You are an evidence-first synthesizer. Given the user query and retrieved documents, write a concise, accurate answer.
- Cite sources inline as [#] by index in the list below.
- Cite sources inline as [1], [2] by index in the list below.
- If insufficient evidence, state what's missing and propose the next best query.

<USER_QUERY>\n{context.get('user_query','')}\n</USER_QUERY>
<RETRIEVED_DOCS>\n{docs_md}\n</RETRIEVED_DOCS>

Return ONLY the answer text (no preamble).
"""
    else:  # report
        return f"""
You are a report writer. Synthesize the plan and Q/A pairs into a coherent report with clear structure.
- Keep sections tight, remove redundancy, and preserve citations.

<PLAN>\n{context.get('plan','')}\n</PLAN>
<QA_PAIRS>\n{json.dumps(context.get('qa_pairs', []), ensure_ascii=False)}\n</QA_PAIRS>
"""


def prompt_judge(criteria: Dict[str, Any], candidate_text: str) -> str:
    return f"""
You are a meticulous evaluator (LLM-as-a-judge). Score the candidate by two metrics:
- Helpfulness: intent satisfaction, accuracy, fluency, appropriate language (1–10)
- Comprehensiveness: no key omissions (1–10)
Provide actionable feedback.

Return STRICT JSON with keys: helpfulness (int), comprehensiveness (int), feedback (string).

<CRITERIA>\n{json.dumps(criteria, ensure_ascii=False)}\n</CRITERIA>
<CANDIDATE>\n{candidate_text}\n</CANDIDATE>
"""


def prompt_revise(candidate_text: str, feedback: str, hard_rules: Optional[str] = None) -> str:
    return f"""
Revise the candidate using the feedback. Keep strengths; fix weaknesses.
- Preserve factual accuracy and citations; do not invent sources.
- If evidence is insufficient, add a TODO note for required data.
{{('HARD RULES:\n'+hard_rules) if hard_rules else ''}}

Return ONLY the revised text.

<CANDIDATE>\n{candidate_text}\n</CANDIDATE>
<FEEDBACK>\n{feedback}\n</FEEDBACK>
"""


def prompt_merge(finalists: List[str], merge_rules: Optional[str] = None) -> str:
    joined = "\n\n---\n\n".join(finalists)
    return f"""
Merge the following top candidates into ONE superior output:
- Remove redundancy, resolve contradictions (prefer statements supported by citations).
- Keep all unique, well-supported insights; preserve citations; unify style and terminology.
{{('MERGE RULES:\n'+merge_rules) if merge_rules else ''}}

Return ONLY the merged text (no bullet list of differences).

<CANDIDATES>\n{joined}\n</CANDIDATES>
"""


# =============================
# JSON parsing helpers for judge
# =============================
JSON_BLOCK = re.compile(r"\{[\s\S]*\}")


def parse_judge_json(s: str) -> Tuple[float, float, str]:
    """Extract judge JSON robustly: returns (helpfulness, comprehensiveness, feedback).
    Fallback: attempt heuristic extraction if JSON invalid.
    """
    try:
        m = JSON_BLOCK.search(s)
        data = json.loads(m.group(0) if m else s)
        h = float(data.get("helpfulness", 0))
        c = float(data.get("comprehensiveness", 0))
        fb = str(data.get("feedback", ""))
        return h, c, fb
    except Exception:
        # Heuristic fallback
        h = _extract_first_number(r"helpfulness\D+(\d+(?:\.\d+)?)", s) or 0.0
        c = _extract_first_number(r"comprehensiveness\D+(\d+(?:\.\d+)?)", s) or 0.0
        fb = s.strip()
        return float(h), float(c), fb


def _extract_first_number(pattern: str, s: str) -> Optional[float]:
    m = re.search(pattern, s, re.IGNORECASE)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


# =============================
# Core Self-Evolution Engine
# =============================
class SelfEvolution:
    def __init__(self, llm: LLMClient, config: EvolutionConfig = EvolutionConfig()) -> None:
        self.llm = llm
        self.cfg = config

    def _score(self, h: float, c: float) -> float:
        return self.cfg.weight_helpfulness * h + self.cfg.weight_comprehensiveness * c

    def _make_variants(self, stage: str, context: Dict[str, Any]) -> List[Candidate]:
        variants: List[Candidate] = []
        temps = self.cfg.temperature_candidates
        topps = self.cfg.top_p_candidates
        for i in range(self.cfg.variants_k):
            t = temps[i % len(temps)]
            p = topps[i % len(topps)]
            text = self.llm.generate(
                prompt_generate_variant(stage, context),
                temperature=t,
                top_p=p,
                max_tokens=self.cfg.max_tokens,
            )
            variants.append(Candidate(text=text, meta={"temperature": t, "top_p": p}))
        return variants

    def _judge(self, cand: Candidate, criteria: Dict[str, Any]) -> Candidate:
        j = self.llm.generate(
            prompt_judge(criteria, cand.text),
            temperature=0.0,
            top_p=1.0,
            max_tokens=self.cfg.max_tokens,
        )
        h, c, fb = parse_judge_json(j)
        cand.helpfulness = h
        cand.comprehensiveness = c
        cand.score = self._score(h, c)
        cand.feedback = fb
        return cand

    def _revise(self, cand: Candidate, hard_rules: Optional[str] = None) -> Candidate:
        revised = self.llm.generate(
            prompt_revise(cand.text, cand.feedback, hard_rules),
            temperature=0.7,
            top_p=0.95,
            max_tokens=self.cfg.max_tokens,
        )
        return Candidate(text=revised, meta=cand.meta.copy())

    def _merge(self, finalists: List[Candidate], merge_rules: Optional[str] = None) -> Candidate:
        merged = self.llm.generate(
            prompt_merge([c.text for c in finalists], merge_rules),
            temperature=0.3,
            top_p=0.9,
            max_tokens=self.cfg.max_tokens,
        )
        return Candidate(text=merged)

    def run_for_stage(
        self,
        stage: str,
        context: Dict[str, Any],
        *,
        judge_criteria: Optional[Dict[str, Any]] = None,
        hard_rules: Optional[str] = None,
        merge_rules: Optional[str] = None,
        seed: Optional[int] = 42,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Run self-evolution for a given stage, returning best text + telemetry.

        Returns dict: {
          "best_text": str,
          "finalists": List[Candidate],
          "history": List[Dict[str, Any]]  # episode-wise stats
        }
        """
        if seed is not None:
            random.seed(seed)
        criteria = judge_criteria or {
            "definition": {
                "helpfulness": [
                    "intent satisfaction", "accuracy", "fluency", "appropriate language"
                ],
                "comprehensiveness": ["no key omissions"],
            }
        }

        # 1) Initialize with k variants
        population = self._make_variants(stage, context)
        # initial judge
        population = [self._judge(c, criteria) for c in population]
        population.sort(key=lambda x: x.score, reverse=True)
        population = population[: self.cfg.population_n]

        history: List[Dict[str, Any]] = []
        last_best = population[0].score if population else 0.0

        # 2) Episodes loop
        for ep in range(self.cfg.episodes_T):
            # Revise each candidate using its feedback
            revised: List[Candidate] = []
            for c in population:
                rc = self._revise(c, hard_rules)
                rc = self._judge(rc, criteria)
                revised.append(rc)

            # Keep top-n (elitism)
            revised.sort(key=lambda x: x.score, reverse=True)
            population = revised[: self.cfg.population_n]

            best = population[0].score
            history.append({
                "episode": ep + 1,
                "best_score": best,
                "best_helpfulness": population[0].helpfulness,
                "best_comprehensiveness": population[0].comprehensiveness,
            })

            if verbose:
                print(f"[EP {ep+1}] best score={best:.2f} (H={population[0].helpfulness:.1f}, C={population[0].comprehensiveness:.1f})")

            if abs(best - last_best) < self.cfg.early_stop_delta:
                if verbose:
                    print("Early stop: improvement below delta.")
                break
            last_best = best

        # 3) Merge finalists
        merged = self._merge(population, merge_rules)
        merged = self._judge(merged, criteria)

        return {
            "best_text": merged.text,
            "finalists": population,
            "history": history,
        }


# =============================
# Example wiring (Stage 2b: answer synthesis)
# =============================
if __name__ == "__main__":  # pragma: no cover
    # 0) Prepare client
    llm = OpenAIClient(model=os.getenv("SELF_EVOLUTION_MODEL", "gpt-4o-mini"))

    # 1) Mock context for Stage 2b (answer synthesis)
    context = {
        "user_query": "What are the key safety considerations when deploying retrieval-augmented agents in healthcare?",
        "retrieved_docs": [
            {
                "title": "FDA Guidance on Clinical Decision Support Software",
                "url": "https://www.fda.gov/medical-devices",
                "snippet": "Outlines recommendations for software functions that support clinical decision-making, risk categorization, and labeling.",
            },
            {
                "title": "HIPAA Security Rule Summary",
                "url": "https://www.hhs.gov/hipaa",
                "snippet": "Administrative, physical, and technical safeguards to ensure confidentiality, integrity and availability of ePHI.",
            },
            {
                "title": "Best Practices for RAG Systems in Sensitive Domains",
                "url": "https://example.org/rag-healthcare",
                "snippet": "Discusses data minimization, audit trails, retrieval validation, and human-in-the-loop protocols.",
            },
        ],
    }

    # 2) Config & engine
    cfg = EvolutionConfig(
    variants_k=4,
    episodes_T=3,         # 라운드 수 (그대로 OK)
    early_stop_delta=0.05 # 더 작게 (또는 0.0)
    )
    engine = SelfEvolution(llm, cfg)

    # 3) Optional rules / criteria
    hard_rules = (
        "- Do not include protected health information (PHI) in examples.\n"
        "- Cite sources inline as [#] using the provided document list order.\n"
    )
    merge_rules = (
        "- Prefer statements supported by multiple sources; mark uncertain claims as TBD.\n"
    )

    # 4) Run
    result = engine.run_for_stage(
        stage="answer",
        context=context,
        hard_rules=hard_rules,
        merge_rules=merge_rules,
        verbose=True,
    )

    print("\n====== MERGED BEST TEXT ======")
    print(result["best_text"])
