"""Silent-finish ceiling measurement on the full MedAgentBench corpus (600 tasks).

Runs the reference graders (refsol.py for v1, new_refsol.py for v2-new) against
a null trace (no tool calls, no ``finish`` value). A grader that accepts the null
trace is labelling the task as a "no-action" branch — silently doing nothing was
the correct behaviour. The pass rate of the null trace across the corpus is the
silent-finish ceiling used in the paper.

FHIR GETs inside graders are routed through the offline MockFHIR cache; no live
FHIR server is required. Graders are normally called against a live HAPI server,
and expect ``send_get_request(url)["data"]`` to be a JSON string (HAPI sends
``application/fhir+json`` which falls through to ``.text`` in the real utils);
we marshal MockFHIR's parsed dicts back to JSON to keep the contract identical.

Usage:
    python -m medagentbench_env.silent_finish_ceiling \\
        [--out data/silent_finish_ceiling.json]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import types
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
GRADER_SRC = REPO_ROOT / "medagentbenchv2" / "medagentbench_v2" / "src"
DATA_DIR = REPO_ROOT / "medagentbench_env" / "data"
CACHE_PATH = DATA_DIR / "fhir_cache.json"
V1_TASKS_PATH = DATA_DIR / "test_data_v2.json"
V2_TASKS_PATH = DATA_DIR / "new_patient_tasks.json"
ART_V2_TASKS_PATH = DATA_DIR / "ART_v2.json"
FHIR_BASE = "http://localhost:8080/fhir/"

sys.path.insert(0, str(GRADER_SRC))

# Avoid importing ``medagentbench_env.server`` (its __init__ pulls in openenv,
# which is a training-only dep). Load fhir_cache as a standalone module.
import importlib.util as _ilu  # noqa: E402

_fhir_cache_spec = _ilu.spec_from_file_location(
    "_fhir_cache_standalone",
    REPO_ROOT / "medagentbench_env" / "server" / "fhir_cache.py",
)
_fhir_cache_mod = _ilu.module_from_spec(_fhir_cache_spec)  # type: ignore[arg-type]
assert _fhir_cache_spec and _fhir_cache_spec.loader
_fhir_cache_spec.loader.exec_module(_fhir_cache_mod)
MockFHIR = _fhir_cache_mod.MockFHIR


def _task_type(task_id: str) -> str:
    parts = task_id.rsplit("_", 1)
    return parts[0] if len(parts) == 2 and parts[1].isdigit() else task_id


def _mock_send_get_request(mock: MockFHIR):
    def _inner(url: str, params=None, headers=None):
        res = mock.get(url)
        if "data" in res and not isinstance(res["data"], str):
            res = {**res, "data": json.dumps(res["data"])}
        return res

    return _inner


def _install_mock(mock: MockFHIR):
    """Monkey-patch send_get_request in the grader modules' namespaces.

    v1 path: use the **canonical strict** grader at
    ``medagentbenchv2/.../MedAgentBench/src/server/tasks/medagentbench/refsol.py``.
    The bundled ``medagentbenchevals/refsol.py`` is a weakened copy (debug
    prints, auto-``return True`` on task5/task9) that mislabels silent-finish
    ceilings; using it here caused v1 task9 to appear as 29/30 silent-pass when
    under the strict grader (which is what the frontier eval uses) it is 0/30.
    Verified 2026-04-24.

    v2-new path: ``medagentbenchevals/new_refsol.py`` — this is the only grader
    for v2-new tasks; no strict/weak duality here.

    ``refsol.py`` does ``from .utils import *`` which binds ``send_get_request``
    as a module-level name, so we patch all the relevant bindings.
    """
    patched = _mock_send_get_request(mock)

    # Canonical strict v1 grader — same loader t2_baseline_v2 uses
    from medagentbench_env.t2_baseline_v2 import _v1_refsol as v1_grader  # type: ignore

    from medagentbenchevals import utils as grader_utils  # type: ignore
    from medagentbenchevals import new_refsol as v2_grader  # type: ignore

    grader_utils.send_get_request = patched
    v1_grader.send_get_request = patched
    v2_grader.send_get_request = patched

    return v1_grader, v2_grader


def _null_results() -> types.SimpleNamespace:
    """Silent-finish trace: no history, no final value."""
    return types.SimpleNamespace(history=[], result=None)


def _score_one(grader_module, task: Dict[str, Any]) -> Any:
    task_id = task["id"]
    task_type = _task_type(task_id)
    grader_fn = getattr(grader_module, task_type, None)
    if grader_fn is None:
        return "no_grader"

    case_data = {
        "id": task_id,
        "instruction": task.get("instruction", ""),
        "context": task.get("context", ""),
        "sol": task.get("sol", []),
        "eval_MRN": task.get("eval_MRN", ""),
    }

    import contextlib
    import io

    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            out = grader_fn(case_data, _null_results(), FHIR_BASE)
        return bool(out is True)
    except Exception as exc:  # noqa: BLE001
        return f"error:{type(exc).__name__}:{exc}"


def _run_corpus(label: str, tasks: List[Dict[str, Any]], grader_module) -> List[Dict[str, Any]]:
    per_task = []
    t0 = time.time()
    for i, task in enumerate(tasks):
        result = _score_one(grader_module, task)
        per_task.append(
            {
                "corpus": label,
                "id": task["id"],
                "task_type": _task_type(task["id"]),
                "eval_MRN": task.get("eval_MRN", ""),
                "silent_pass": result,
            }
        )
        if (i + 1) % 50 == 0:
            print(f"  [{label}] {i+1}/{len(tasks)}  ({time.time()-t0:.1f}s)", flush=True)
    print(f"  [{label}] done {len(tasks)} tasks in {time.time()-t0:.1f}s", flush=True)
    return per_task


def _summarise(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_corpus_type = defaultdict(lambda: {"pass": 0, "fail": 0, "error": 0, "total": 0})
    by_corpus = defaultdict(lambda: {"pass": 0, "fail": 0, "error": 0, "total": 0})

    for r in rows:
        key_ct = (r["corpus"], r["task_type"])
        key_c = r["corpus"]
        for d in (by_corpus_type[key_ct], by_corpus[key_c]):
            d["total"] += 1
            if r["silent_pass"] is True:
                d["pass"] += 1
            elif r["silent_pass"] is False:
                d["fail"] += 1
            else:
                d["error"] += 1

    return {
        "by_corpus": {k: v for k, v in by_corpus.items()},
        "by_corpus_task": {f"{c}/{t}": v for (c, t), v in by_corpus_type.items()},
    }


def _print_table(summary: Dict[str, Any]) -> None:
    print()
    print("Silent-finish ceiling — per task type")
    print("=" * 72)
    print(f"{'corpus':<8}{'task':<12}{'pass':>6}{'fail':>6}{'err':>5}{'total':>7}{'ceil':>9}")
    print("-" * 72)
    for key, stats in sorted(summary["by_corpus_task"].items()):
        corpus, tt = key.split("/", 1)
        pct = (stats["pass"] / stats["total"] * 100) if stats["total"] else 0.0
        print(
            f"{corpus:<8}{tt:<12}{stats['pass']:>6}{stats['fail']:>6}"
            f"{stats['error']:>5}{stats['total']:>7}{pct:>8.1f}%"
        )
    print("-" * 72)
    for corpus, stats in sorted(summary["by_corpus"].items()):
        pct = (stats["pass"] / stats["total"] * 100) if stats["total"] else 0.0
        print(
            f"{corpus:<8}{'TOTAL':<12}{stats['pass']:>6}{stats['fail']:>6}"
            f"{stats['error']:>5}{stats['total']:>7}{pct:>8.1f}%"
        )
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=DATA_DIR / "silent_finish_ceiling.json",
        help="Output JSON path (default: data/silent_finish_ceiling.json)",
    )
    args = parser.parse_args()

    print(f"Loading FHIR cache from {CACHE_PATH} ...", flush=True)
    t0 = time.time()
    mock = MockFHIR.from_cache(str(CACHE_PATH), FHIR_BASE)
    print(f"  loaded in {time.time()-t0:.1f}s", flush=True)

    v1_grader, v2_grader = _install_mock(mock)

    with V1_TASKS_PATH.open() as f:
        v1_tasks = json.load(f)
    with V2_TASKS_PATH.open() as f:
        v2_tasks = json.load(f)
    print(f"v1 corpus: {len(v1_tasks)} tasks — {V1_TASKS_PATH.name}")
    print(f"v2 corpus: {len(v2_tasks)} tasks — {V2_TASKS_PATH.name}")

    rows: List[Dict[str, Any]] = []
    rows.extend(_run_corpus("v1", v1_tasks, v1_grader))
    rows.extend(_run_corpus("v2_new", v2_tasks, v2_grader))

    # ART v2 — uses our own grader, so we score it inline here rather than
    # going through the MAB grader dispatcher.
    if ART_V2_TASKS_PATH.exists():
        from medagentbench_env.art.grader import grade as art_grade  # noqa: WPS433

        with ART_V2_TASKS_PATH.open() as f:
            art_tasks = json.load(f)
        print(f"ART v2 corpus: {len(art_tasks)} tasks — {ART_V2_TASKS_PATH.name}")
        t0 = time.time()
        for i, task in enumerate(art_tasks):
            try:
                ok = art_grade(task, _null_results(), FHIR_BASE)
            except Exception as exc:  # noqa: BLE001
                ok = f"error:{type(exc).__name__}:{exc}"
            rows.append(
                {
                    "corpus": "art_v2",
                    "id": task["id"],
                    "task_type": task["ground_truth"]["task_type"],
                    "eval_MRN": task.get("eval_MRN", ""),
                    "silent_pass": ok,
                }
            )
            if (i + 1) % 150 == 0:
                print(f"  [art_v2] {i+1}/{len(art_tasks)}  ({time.time()-t0:.1f}s)", flush=True)
        print(f"  [art_v2] done {len(art_tasks)} tasks in {time.time()-t0:.1f}s", flush=True)

    summary = _summarise(rows)

    payload = {
        "fhir_base": FHIR_BASE,
        "cache_path": str(CACHE_PATH),
        "v1_tasks_path": str(V1_TASKS_PATH),
        "v2_tasks_path": str(V2_TASKS_PATH),
        "per_task": rows,
        "summary": summary,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {args.out}  ({args.out.stat().st_size/1024:.1f} KB)")

    _print_table(summary)


if __name__ == "__main__":
    main()
