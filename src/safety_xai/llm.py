"""Calling local models through Ollama, with a cache, plus the SLM analyser.

Everything runs locally, so no student text leaves the machine. This is true
for the SLM under test and also for the judge and the LLM annotators.

Cache: every successful answer is stored in results/llm_cache.jsonl, keyed by
model + prompt + options. Each text is therefore generated ONCE and reused by
both classifier runs, the judge and the report. (Ollama at temperature 0 is not
always bit-for-bit deterministic, so regenerating could change a narrative.)
"""

import hashlib
import json
import time

from .config import LLM_CACHE, LLM_OPTIONS, RISK_CATEGORIES, SLM_MODEL

_cache = None


def _load_cache():
    global _cache
    if _cache is None:
        _cache = {}
        if LLM_CACHE.exists():
            for line in LLM_CACHE.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    _cache[rec["key"]] = rec
    return _cache


def ask_json(model, prompt, check=None, schema=None):
    """Ask a local model for a JSON answer.

    `check(dict) -> dict` validates/cleans the answer and raises on bad output.
    Returns {"ok": bool, "answer": dict or None, "error": str, "seconds": float, "cached": bool}.
    Failed calls are retried once and never cached.
    `schema` (optional JSON schema) makes Ollama only produce answers of that shape.
    """
    import ollama

    key = hashlib.sha256(json.dumps([model, prompt, LLM_OPTIONS], sort_keys=True).encode()).hexdigest()
    cache = _load_cache()
    if key in cache:
        rec = cache[key]
        return {"ok": True, "answer": rec["answer"], "error": "", "seconds": rec["seconds"], "cached": True}

    error = ""
    for _ in range(2):
        start = time.perf_counter()
        try:
            resp = ollama.generate(model=model, prompt=prompt, format=schema or "json", options=LLM_OPTIONS)
            answer = json.loads(resp["response"])
            if check is not None:
                answer = check(answer)
            seconds = time.perf_counter() - start
            rec = {"key": key, "model": model, "prompt": prompt, "answer": answer, "seconds": seconds}
            cache[key] = rec
            LLM_CACHE.parent.mkdir(parents=True, exist_ok=True)
            with LLM_CACHE.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            return {"ok": True, "answer": answer, "error": "", "seconds": seconds, "cached": False}
        except Exception as e:  # server down, refusal, or malformed JSON
            error = f"{type(e).__name__}: {e}"[:300]
    return {"ok": False, "answer": None, "error": error, "seconds": None, "cached": False}


def clean_category(value):
    value = str(value).strip()
    return value if value in RISK_CATEGORIES else "none"


# ---------------------------------------------------------------- SLM semantic analyser
# Changes from v1, from the reviews:
#  * no sentence saying the query "was flagged as high-risk" (that leaked the label)
#  * the SLM makes its OWN decision: risk + category
#  * explain-then-predict (Ludan et al.): the "intent" sentence is written first,
#    then the decision, so the decision is conditioned on the explanation
#  * it runs on EVERY query, in parallel with the classifier, never after it
SLM_PROMPT = """You help a school safety team review search queries typed by students on school computers.

Query: "{query}"

Judge only what the query itself says. It may be harmful, harmless, or unclear.
First explain, then decide. Return a JSON object with these keys, in this order:
"intent": one sentence describing what the student most likely wants to do.
"risk": "harmful", "benign" or "unclear".
"category": the best matching category from {categories}, or "none" if the query is not harmful."""


# Forces valid keys and labels. Without it, Llama sometimes put the category in "risk"
# or returned junk like {".ping": "..."} (31 of 287 answers in the first run).
SLM_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string"},
        "risk": {"type": "string", "enum": ["harmful", "benign", "unclear"]},
        "category": {"type": "string", "enum": RISK_CATEGORIES + ["none"]},
    },
    "required": ["intent", "risk", "category"],
}


def check_slm_answer(answer):
    risk = str(answer.get("risk", "")).strip().lower()
    intent = str(answer.get("intent", "")).strip()
    if risk not in ("harmful", "benign", "unclear") or not intent:
        raise ValueError(f"bad SLM answer: {answer}")
    return {"intent": intent, "risk": risk, "category": clean_category(answer.get("category", "none"))}


def analyse_query(query, model=SLM_MODEL):
    prompt = SLM_PROMPT.format(query=query.replace('"', "'"), categories=json.dumps(RISK_CATEGORIES))
    return ask_json(model, prompt, check_slm_answer, SLM_SCHEMA)


def ollama_memory():
    """Memory used by the models Ollama has loaded right now (from `ollama ps`)."""
    import ollama

    out = []
    for m in ollama.ps().models:
        out.append({"model": m.model, "total_gb": round(m.size / 1e9, 2), "gpu_gb": round(m.size_vram / 1e9, 2)})
    return out


def require_models(models):
    """Stop early with a clear message if a model is not pulled in Ollama."""
    import ollama

    try:
        installed = {m.model for m in ollama.list().models}
    except Exception as e:
        raise SystemExit(f"Cannot reach Ollama ({e}). Open the Ollama app and try again.")
    missing = [m for m in models if m not in installed and f"{m}:latest" not in installed]
    if missing:
        pulls = "; ".join(f"ollama pull {m}" for m in missing)
        raise SystemExit(f"Not installed in Ollama: {missing}. Run: {pulls}")
