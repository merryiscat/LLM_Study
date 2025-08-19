# log/observability.py
import time, uuid, json, traceback
from functools import wraps
from typing import Dict, Any, Callable, Iterable, Optional
from logger.logger import get_logger

logger = get_logger("graph")

def _summarize(value, max_str=200, max_items=5):
    try:
        if isinstance(value, str):
            return value if len(value) <= max_str else value[:max_str] + f"...({len(value)} chars)"
        if isinstance(value, (int, float, bool)) or value is None:
            return value
        if isinstance(value, list):
            return {"type":"list", "len":len(value), "head": value[:max_items]}
        if isinstance(value, tuple):
            return {"type":"tuple","len":len(value), "head": list(value[:max_items])}
        if isinstance(value, dict):
            keys = list(value.keys())
            head_keys = keys[:max_items]
            return {"type":"dict","len":len(keys),
                    "keys_head": {k: _summarize(value[k], max_str, max_items) for k in head_keys}}
        return repr(value)[:max_str]
    except Exception:
        return "<unserializable>"

def _mask_text(s: Any, rules: Iterable[str]):
    if not isinstance(s, str):
        return s
    out = s
    for r in rules:
        out = out.replace(r, "***")
    return out

def snapshot_outputs(state: Dict[str, Any],
                     include: Optional[Iterable[str]],
                     exclude: Optional[Iterable[str]],
                     mask_rules: Iterable[str],
                     max_str: int, max_items: int) -> Dict[str, Any]:
    keys = set(state.keys()) if include is None else set(include)
    if exclude:
        keys -= set(exclude)
    snap = {}
    for k in keys:
        v = state.get(k)
        if isinstance(v, str):
            v = _mask_text(v, mask_rules)
        snap[k] = _summarize(v, max_str=max_str, max_items=max_items)
    return snap

def log_node_outputs(node_name: str,
                     *,
                     include_keys: Optional[Iterable[str]] = None,  # None이면 모든 키 요약
                     exclude_keys: Optional[Iterable[str]] = None,
                     mask_rules: Iterable[str] = ("sk-", "Bearer "),
                     max_str: int = 200,
                     max_items: int = 5) -> Callable:
    """노드 return 값만 로그(JSONL)로 남깁니다."""
    def deco(fn: Callable):
        @wraps(fn)
        def wrapper(state: Dict[str, Any]) -> Dict[str, Any]:
            run_id = state.get("run_id") or str(uuid.uuid4())
            t0 = time.time()
            try:
                out = fn(state)
                out.setdefault("run_id", run_id)
                dt = int((time.time() - t0) * 1000)

                out_snap = snapshot_outputs(out, include_keys, exclude_keys, mask_rules, max_str, max_items)

                logger.info(f"{node_name} return", extra={"_extra": {
                    "event": "node_return",
                    "node": node_name,
                    "run_id": run_id,
                    "duration_ms": dt,
                    "outputs": out_snap
                }})
                return out
            except Exception as e:
                dt = int((time.time() - t0) * 1000)
                logger.error(f"{node_name} error", extra={"_extra":{
                    "event":"node_error","node":node_name,"run_id":run_id,
                    "duration_ms":dt,"error":str(e),"trace": traceback.format_exc()
                }})
                raise
        return wrapper
    return deco
