"""T2 v2 — Frontier baseline that mirrors the official MedAgentBench v1 harness.

Context (superseded script removed during Apr 2026 cleanup): an earlier revision
used a custom raw-HTTP prompt and pulled the v1 grader from
``medagentbenchevals/refsol.py``, which has debug prints and auto-``return True``
paths on several task types. Both disadvantaged frontier models vs. their
published scores. This is the canonical harness. Changes from the earlier draft:

  1. Reproduces the exact ``MedAgentBench_prompt`` template from
     ``src/server/tasks/medagentbench/__init__.py`` — includes ``{functions}``
     (the 9-function JSON schema from ``funcs_v1.json``), ``{context}``,
     ``{question}``, ``{api_base}``.
  2. Uses ``max_round`` = 8 (v1 paper's reported cap).
  3. Mirrors the per-turn message logic: appends the canonical
     "Please call FINISH..." hint after GET/POST.
  4. Calls the canonical v1 grader at
     ``MedAgentBench/src/server/tasks/medagentbench/refsol.py``
     (1340 lines, strict) for v1 tasks; ``medagentbenchevals/new_refsol.py``
     for v2-new; ``art.grader`` for ART v2.
  5. Routes all FHIR GETs through MockFHIR (same offline cache; 98/98 MRNs
     covered). POSTs are accepted syntactically, same as official harness.
  6. Also evaluates v2-new and ART v2 with the same canonical-style protocol.

Output: ``data/t2v2_baseline_<slug>.json``.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
import time
import types
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI

REPO_ROOT = Path(__file__).resolve().parent.parent
MAB_SRC = REPO_ROOT / "medagentbenchv2" / "medagentbench_v2" / "src" / "MedAgentBench" / "src"
EVALS_SRC = REPO_ROOT / "medagentbenchv2" / "medagentbench_v2" / "src"
DATA_DIR = REPO_ROOT / "medagentbench_env" / "data"
CACHE_PATH = DATA_DIR / "fhir_cache.json"
SILENT_CEIL_PATH = DATA_DIR / "silent_finish_ceiling.json"
FHIR_BASE = "http://localhost:8080/fhir/"

V1_TASKS = DATA_DIR / "test_data_v2.json"
V2_TASKS = DATA_DIR / "new_patient_tasks.json"
ART_TASKS = DATA_DIR / "ART_v2.json"
FUNCS_V1_PATH = DATA_DIR / "funcs_v1.json"

# Make both grader locations importable.
sys.path.insert(0, str(MAB_SRC))  # enables `src.server.tasks.medagentbench.refsol`
sys.path.insert(0, str(EVALS_SRC))  # enables `medagentbenchevals.new_refsol`
sys.path.insert(0, str(REPO_ROOT))

# Load the canonical v1 grader under its proper package path
# (``src.server.tasks.medagentbench.refsol``) so its ``from .utils import *``
# works without error. We synthesise a minimal package hierarchy in
# sys.modules and let importlib do the rest.
_MAB_PKG_ROOT = MAB_SRC.parent  # .../MedAgentBench

def _load_mab_refsol():
    import types as _t
    # Build the namespace packages src → src.server → src.server.tasks → ...
    for pkg_name, pkg_path in [
        ("src", MAB_SRC),
        ("src.server", MAB_SRC / "server"),
        ("src.server.tasks", MAB_SRC / "server" / "tasks"),
        ("src.server.tasks.medagentbench", MAB_SRC / "server" / "tasks" / "medagentbench"),
    ]:
        if pkg_name not in sys.modules:
            mod = _t.ModuleType(pkg_name)
            mod.__path__ = [str(pkg_path)]  # type: ignore[attr-defined]
            sys.modules[pkg_name] = mod
    utils_path = MAB_SRC / "server" / "tasks" / "medagentbench" / "utils.py"
    spec_u = importlib.util.spec_from_file_location(
        "src.server.tasks.medagentbench.utils", utils_path
    )
    utils_mod = importlib.util.module_from_spec(spec_u)  # type: ignore[arg-type]
    sys.modules["src.server.tasks.medagentbench.utils"] = utils_mod
    spec_u.loader.exec_module(utils_mod)  # type: ignore[union-attr]

    refsol_path = MAB_SRC / "server" / "tasks" / "medagentbench" / "refsol.py"
    spec_r = importlib.util.spec_from_file_location(
        "src.server.tasks.medagentbench.refsol", refsol_path
    )
    refsol_mod = importlib.util.module_from_spec(spec_r)  # type: ignore[arg-type]
    sys.modules["src.server.tasks.medagentbench.refsol"] = refsol_mod
    spec_r.loader.exec_module(refsol_mod)  # type: ignore[union-attr]
    return refsol_mod, utils_mod

_v1_refsol, _utils = _load_mab_refsol()

from medagentbenchevals import new_refsol as _v2_grader  # type: ignore  # noqa: E402
from medagentbenchevals import utils as _evals_utils  # type: ignore  # noqa: E402
from medagentbench_env.art import grader as _art_grader  # noqa: E402

# Standalone MockFHIR loader (avoids openenv training-only dep).
_fhir_cache_spec = importlib.util.spec_from_file_location(
    "_fhir_cache_standalone",
    REPO_ROOT / "medagentbench_env" / "server" / "fhir_cache.py",
)
_fhir_cache_mod = importlib.util.module_from_spec(_fhir_cache_spec)  # type: ignore[arg-type]
_fhir_cache_spec.loader.exec_module(_fhir_cache_mod)  # type: ignore[union-attr]
MockFHIR = _fhir_cache_mod.MockFHIR

load_dotenv(REPO_ROOT / ".env")


# ---------------------------------------------------------------------------
# Official MedAgentBench v1 prompt template (copied verbatim from
# ``MedAgentBench/src/server/tasks/medagentbench/__init__.py`` line 11-29).
# ---------------------------------------------------------------------------

MED_AGENT_BENCH_PROMPT = """You are an expert in using FHIR functions to assist medical professionals. You are given a question and a set of possible functions. Based on the question, you will need to make one or more function/tool calls to achieve the purpose.

1. If you decide to invoke a GET function, you MUST put it in the format of
GET url?param_name1=param_value1&param_name2=param_value2...

2. If you decide to invoke a POST function, you MUST put it in the format of
POST url
[your payload data in JSON format]

3. If you have got answers for all the questions and finished all the requested tasks, you MUST call to finish the conversation in the format of (make sure the list is JSON loadable.)
FINISH([answer1, answer2, ...])

Your response must be in the format of one of the three cases, and you can call only one function each time. You SHOULD NOT include any other text in the response.

Here is a list of functions in JSON format that you can invoke. Note that you should use {api_base} as the api_base.
{functions}

Context: {context}
Question: {question}"""


with FUNCS_V1_PATH.open() as _f:
    _FUNCS_V1 = json.load(_f)
_FUNCS_JSON = json.dumps(_FUNCS_V1)


def _build_initial_prompt(task: Dict[str, Any]) -> str:
    return MED_AGENT_BENCH_PROMPT.format(
        api_base=FHIR_BASE,
        functions=_FUNCS_JSON,
        context=task.get("context", ""),
        question=task.get("instruction", ""),
    )


# ---------------------------------------------------------------------------
# Tool execution through MockFHIR (mirrors the official harness's
# ``send_get_request``). POST is accepted syntactically only, same as official.
# ---------------------------------------------------------------------------


def _exec_get(mock: MockFHIR, url: str) -> Dict[str, Any]:
    """Mirror of ``src.server.tasks.medagentbench.utils.send_get_request`` but
    backed by the offline cache. Returns either {'data': str} or {'error': str}.
    """
    res = mock.get(url)
    if "error" in res:
        return {"error": res["error"]}
    data = res.get("data")
    # The real ``send_get_request`` returns ``.text`` (string) for
    # ``application/fhir+json``; the cache sometimes stores a parsed dict.
    if isinstance(data, dict):
        return {"data": json.dumps(data)}
    return {"data": str(data)}


# ---------------------------------------------------------------------------
# Per-turn loop — matches MedAgentBench.__init__.start_sample verbatim.
# ---------------------------------------------------------------------------


def run_task(
    client: OpenAI,
    model: str,
    task: Dict[str, Any],
    mock: MockFHIR,
    max_round: int = 8,
    temperature: float = 0.0,
    max_tokens: int = 2048,
) -> Dict[str, Any]:
    """Run one task using the official MAB v1 protocol.

    Returns a trace in the message-list format the graders expect:
    ``[SimpleNamespace(role='user'|'agent', content=...)]``.
    """
    initial_prompt = _build_initial_prompt(task)

    # ``messages`` drives the OpenAI chat completion; ``history`` drives the grader.
    # The official harness tracks a single ``session.history`` with {role, content};
    # we replicate that.
    chat_messages: List[Dict[str, str]] = [{"role": "user", "content": initial_prompt}]
    history: List[types.SimpleNamespace] = [
        types.SimpleNamespace(role="user", content=initial_prompt)
    ]

    finish_value: Optional[Any] = None
    sample_status = "RUNNING"
    finish_raw: Optional[str] = None

    for round_idx in range(max_round):
        # Chat completion
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=chat_messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            reply = resp.choices[0].message.content or ""
        except Exception as exc:  # noqa: BLE001
            return {
                "history": history,
                "finish_value": None,
                "finish_raw": None,
                "sample_status": "AGENT_ERROR",
                "rounds": round_idx,
                "error": str(exc),
            }

        chat_messages.append({"role": "assistant", "content": reply})
        history.append(types.SimpleNamespace(role="agent", content=reply))

        # Mirror the official harness's stripping
        r = (
            reply.strip()
            .replace("```tool_code", "")
            .replace("```", "")
            .strip()
        )

        if r.startswith("GET"):
            url = r[3:].strip()
            if "_format=json" not in url:
                url = url + ("&_format=json" if "?" in url else "?_format=json")
            get_res = _exec_get(mock, url)
            if "data" in get_res:
                content = (
                    f"Here is the response from the GET request:\n{get_res['data']}. "
                    "Please call FINISH if you have got answers for all the questions "
                    "and finished all the requested tasks"
                )
            else:
                content = f"Error in sending the GET request: {get_res['error']}"
            chat_messages.append({"role": "user", "content": content})
            history.append(types.SimpleNamespace(role="user", content=content))

        elif r.startswith("POST"):
            lines = r.split("\n")
            url = lines[0][4:].strip() if lines else ""
            try:
                _payload = json.loads("\n".join(lines[1:]))
            except Exception:  # noqa: BLE001
                content = "Invalid POST request"
            else:
                content = (
                    "POST request accepted and executed successfully. "
                    "Please call FINISH if you have got answers for all the "
                    "questions and finished all the requested tasks"
                )
            chat_messages.append({"role": "user", "content": content})
            history.append(types.SimpleNamespace(role="user", content=content))

        elif r.startswith("FINISH("):
            finish_raw = r[len("FINISH(") : -1]  # trim to list body (string)
            try:
                finish_value = json.loads(finish_raw)
            except Exception:  # noqa: BLE001
                finish_value = finish_raw
            sample_status = "COMPLETED"
            break

        else:
            sample_status = "AGENT_INVALID_ACTION"
            break
    else:
        sample_status = "TASK_LIMIT_REACHED"

    return {
        "history": history,
        "finish_value": finish_value,
        "finish_raw": finish_raw,
        "sample_status": sample_status,
        "rounds": len(history) // 2,
    }


# ---------------------------------------------------------------------------
# MockFHIR patch for the grader's ``send_get_request`` (so both the v1 canonical
# grader and the v2-new grader see the offline cache).
# ---------------------------------------------------------------------------


def _patch_grader_requests(mock: MockFHIR) -> None:
    def patched(url, params=None, headers=None):
        res = mock.get(url)
        if "data" in res and not isinstance(res["data"], str):
            res = {**res, "data": json.dumps(res["data"])}
        return res

    _evals_utils.send_get_request = patched
    _v2_grader.send_get_request = patched
    # The v1 canonical grader got ``.utils`` names merged into its module dict
    # during load; update both the local module and the utils object.
    _utils.send_get_request = patched
    _v1_refsol.send_get_request = patched


# ---------------------------------------------------------------------------
# Grader dispatch
# ---------------------------------------------------------------------------


def _case_data(task: Dict) -> Dict:
    return {
        "id": task["id"],
        "instruction": task.get("instruction", ""),
        "context": task.get("context", ""),
        "sol": task.get("sol", []),
        "eval_MRN": task.get("eval_MRN", ""),
    }


def _grade(corpus: str, task: Dict, history: List, finish_raw: Optional[str]) -> Any:
    """Call the appropriate grader. ``results.result`` is the raw FINISH argument
    string (that's what the official harness passes — trimmed ``r[len("FINISH("):-1]``
    without ``json.loads``). The graders themselves do ``json.loads(results.result)``.
    """
    trace_obj = types.SimpleNamespace(history=history, result=finish_raw)
    try:
        if corpus == "v1":
            tt = task["id"].split("_")[0]
            fn = getattr(_v1_refsol, tt, None)
            if fn is None:
                return "no_grader"
            return fn(_case_data(task), trace_obj, FHIR_BASE) is True
        if corpus == "v2_new":
            tt = task["id"].rsplit("_", 1)[0]
            fn = getattr(_v2_grader, tt, None)
            if fn is None:
                return "no_grader"
            return fn(_case_data(task), trace_obj, FHIR_BASE) is True
        if corpus == "art_v2":
            return _art_grader.grade(task, trace_obj, FHIR_BASE) is True
        raise KeyError(corpus)
    except Exception as exc:  # noqa: BLE001
        return f"err:{type(exc).__name__}:{exc}"


# ---------------------------------------------------------------------------
# Stratified sampling — labels are per-instance, derived from the silent-finish
# ceiling output (silent_pass == True → no_action; False → action-required).
# For the full clinKriya-Fair benchmark this path is bypassed; use --tasks-file.
# ---------------------------------------------------------------------------


def _action_label_v1_v2(corpus: str, task: Dict, ceiling_rows: List[Dict]) -> str:
    for r in ceiling_rows:
        if r["corpus"] == corpus and r["id"] == task["id"]:
            return "no_action" if r.get("silent_pass") is True else "action"
    return "unknown"


def _action_label_art(task: Dict) -> str:
    return "action" if task.get("ground_truth", {}).get("should_order") else "no_action"


def stratified_sample(
    corpus: str,
    tasks: List[Dict],
    ceiling_rows: List[Dict],
    per_bucket: int,
    seed: int,
) -> List[Dict]:
    import random

    rng = random.Random(seed)
    buckets: Dict[Tuple[str, str], List[Dict]] = defaultdict(list)
    for t in tasks:
        if corpus == "art_v2":
            sub = t["ground_truth"]["task_type"]
            label = _action_label_art(t)
        else:
            sub = t["id"].rsplit("_", 1)[0]
            label = _action_label_v1_v2(corpus, t, ceiling_rows)
        buckets[(sub, label)].append(t)
    sampled: List[Dict] = []
    for key, items in sorted(buckets.items()):
        rng.shuffle(items)
        sampled.extend(items[:per_bucket])
    return sampled


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def _model_slug(model: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", model.lower()).strip("_")


def _summarise(rows: List[Dict]) -> Dict:
    by_corpus = defaultdict(lambda: {"pass": 0, "fail": 0, "err": 0, "total": 0})
    by_strata = defaultdict(lambda: {"pass": 0, "fail": 0, "err": 0, "total": 0})
    for r in rows:
        gp = r["graded_pass"]
        c = r["corpus"]
        s = r.get("sub_type", "?")
        lbl = r["action_label"]
        for d in (by_corpus[c], by_strata[(c, s, lbl)]):
            d["total"] += 1
            if gp is True:
                d["pass"] += 1
            elif gp is False:
                d["fail"] += 1
            else:
                d["err"] += 1

    def _pct(d):
        return {**d, "pass_rate": (d["pass"] / d["total"] * 100) if d["total"] else 0.0}

    return {
        "by_corpus": {k: _pct(v) for k, v in by_corpus.items()},
        "by_strata": {f"{c}/{s}/{l}": _pct(v) for (c, s, l), v in by_strata.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="openai/gpt-4o-2024-11-20")
    parser.add_argument(
        "--corpora",
        nargs="+",
        choices=["v1", "v2_new", "art_v2"],
        default=["v1", "v2_new", "art_v2"],
    )
    parser.add_argument("--per-bucket", type=int, default=3)
    parser.add_argument("--max-round", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--tasks-file",
        type=Path,
        default=None,
        help=(
            "Path to a benchmark index.json (list of {corpus, task_id, task_type, "
            "action_label}). If given, bypasses --corpora/--per-bucket stratified "
            "sampling and evaluates exactly those tasks."
        ),
    )
    args = parser.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("OPENROUTER_API_KEY not set.", file=sys.stderr)
        sys.exit(1)
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

    print("Loading FHIR cache ...", flush=True)
    t0 = time.time()
    mock = MockFHIR.from_cache(str(CACHE_PATH), FHIR_BASE)
    print(f"  loaded in {time.time()-t0:.1f}s")

    _patch_grader_requests(mock)

    with SILENT_CEIL_PATH.open() as f:
        ceiling_rows = json.load(f)["per_task"]

    corpus_tasks: Dict[str, List[Dict]] = {}
    if "v1" in args.corpora:
        corpus_tasks["v1"] = json.load(V1_TASKS.open())
    if "v2_new" in args.corpora:
        corpus_tasks["v2_new"] = json.load(V2_TASKS.open())
    if "art_v2" in args.corpora:
        corpus_tasks["art_v2"] = json.load(ART_TASKS.open())

    sampled: List[Tuple[str, Dict]] = []
    if args.tasks_file is not None:
        with args.tasks_file.open() as f:
            index = json.load(f)

        # Source-of-truth table for task bodies, keyed by (corpus, task_id).
        # Only consulted when the index row doesn't embed its own body.
        by_id: Dict[Tuple[str, str], Dict] = {}
        for corpus, tasks in corpus_tasks.items():
            for t in tasks:
                by_id[(corpus, t["id"])] = t

        missing = []
        for r in index:
            key = (r["corpus"], r["task_id"])
            # Prefer embedded body (e.g., benchmark_fair_augmented.index.json,
            # where contexts have been patched); fall back to source dataset.
            if isinstance(r.get("task"), dict) and r["task"]:
                body = r["task"]
            elif key in by_id:
                body = by_id[key]
            else:
                missing.append(key)
                continue
            sampled.append((r["corpus"], body))
        if missing:
            print(f"WARNING: {len(missing)} tasks in index not found in loaded corpora: {missing[:5]}...", file=sys.stderr)
    else:
        for corpus, tasks in corpus_tasks.items():
            sub_sample = stratified_sample(
                corpus, tasks, ceiling_rows, args.per_bucket, args.seed
            )
            for t in sub_sample:
                sampled.append((corpus, t))

    if args.limit is not None:
        sampled = sampled[: args.limit]

    print(f"Model:     {args.model}")
    print(f"Tasks:     {len(sampled)} sampled from {sum(len(v) for v in corpus_tasks.values())}")
    print(f"Per-bucket: {args.per_bucket}  max_round: {args.max_round}  temp: {args.temperature}")
    print(f"Protocol:  Official MedAgentBench v1 prompt + schema ({len(_FUNCS_V1)} funcs)")
    print()

    rows: List[Dict[str, Any]] = []
    t0 = time.time()
    for i, (corpus, task) in enumerate(sampled):
        sub_type = (
            task["ground_truth"]["task_type"]
            if corpus == "art_v2"
            else task["id"].rsplit("_", 1)[0]
        )
        action_label = (
            _action_label_art(task)
            if corpus == "art_v2"
            else _action_label_v1_v2(corpus, task, ceiling_rows)
        )

        run = run_task(
            client, args.model, task, mock,
            max_round=args.max_round,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        # Do NOT let an AGENT_ERROR with empty history count as a silent-finish
        # pass on no-action grading — that's a spurious artifact. Mark as error
        # so the summary separates genuine passes from infrastructure failures.
        if run["sample_status"] == "AGENT_ERROR":
            graded = f"err:agent_error"
        else:
            graded = _grade(corpus, task, run["history"], run["finish_raw"])

        rows.append({
            "corpus": corpus,
            "id": task["id"],
            "sub_type": sub_type,
            "action_label": action_label,
            "eval_MRN": task.get("eval_MRN", ""),
            "rounds": run["rounds"],
            "sample_status": run["sample_status"],
            "finish_raw": run["finish_raw"],
            "graded_pass": graded,
            "error": run.get("error"),
        })

        elapsed = time.time() - t0
        per = elapsed / (i + 1)
        eta = per * (len(sampled) - i - 1)
        print(
            f"[{i+1:3d}/{len(sampled)}] {corpus:<7s} {task['id']:<32s} "
            f"{sub_type:<22s} {action_label:<10s} "
            f"r={run['rounds']:<2d} st={run['sample_status']:<22s} pass={graded}  "
            f"({elapsed:.0f}s / ~{eta:.0f}s left)",
            flush=True,
        )

    summary = _summarise(rows)

    out_path = args.out or (DATA_DIR / f"t2v2_baseline_{_model_slug(args.model)}.json")
    payload = {
        "model": args.model,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "harness": "official_medagentbench_v1_prompt",
        "config": {
            "per_bucket": args.per_bucket,
            "max_round": args.max_round,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "seed": args.seed,
            "corpora": args.corpora,
            "num_funcs": len(_FUNCS_V1),
        },
        "summary": summary,
        "results": rows,
    }
    with out_path.open("w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\nWrote {out_path}  ({out_path.stat().st_size/1024:.1f} KB)")

    print("\nPer-corpus pass rate")
    print("-" * 50)
    for k, v in sorted(summary["by_corpus"].items()):
        print(f"  {k:<10s} {v['pass']:>3d}/{v['total']:<3d}  {v['pass_rate']:>5.1f}%")
    print("\nPer stratum (corpus / sub_type / label)")
    print("-" * 70)
    for k, v in sorted(summary["by_strata"].items()):
        print(f"  {k:<55s} {v['pass']:>3d}/{v['total']:<3d}  {v['pass_rate']:>5.1f}%")


if __name__ == "__main__":
    main()
