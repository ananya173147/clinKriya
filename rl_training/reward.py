"""
Unified reward computation for MedAgentBench RL tasks.

Supports: task1, task2, task4, task5, task6, task7, task8, task9, task10

Design principle
────────────────
The reward function does NOT know which specific FHIR tools or codes the agent
should use — the agent learns that from the task instruction.  Rewards are
purely outcome-based:

  +terminal        : official task evaluator passes (+1.0)
  +decision        : correct act/no-act decision, gated on having made ≥1 GET
                     (only fires when ground truth is known, e.g. task7 QTc)
  -spurious_action : agent POSTed when ground truth says no action needed
  -redundant_lookup: agent repeated an identical GET URL (-0.05 each, cap -0.20)
  -invalid_fhir    : FHIR call rejected or errored (-0.10)

  Task-7 extras (specific required actions are part of clinical correctness):
  +ecg_order       : placed follow-up ECG ServiceRequest
  +drug_stop       : discontinued QT-prolonging medication
  -coupled_missing : ECG without drug stop or vice versa

The `decision` reward gate ("made ≥1 GET") indirectly incentivizes reading the
chart before deciding, without prescribing which endpoint to call.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

# ── path setup ────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent / "medagentbench_v2" / "src"
for _p in [str(_SRC), str(_HERE.parent / "medagentbench_v2")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from src.medagentbenchevals.utils import send_get_request as _send_get
except ImportError:
    _send_get = None

# Lazy evaluator imports — each wrapped so missing modules don't crash startup
def _try_import_evaluator(task_type: str):
    """Return the official task evaluator function or None if unavailable."""
    try:
        if task_type == "task7":
            from src.medagentbenchevals.new_refsol import task7
            return task7
    except Exception:
        pass
    return None

from .config import RewardConfig


# ─── Data structures ──────────────────────────────────────────────────────────

@dataclass
class HistoryItem:
    """Matches the ChatHistoryItem format expected by extract_posts()."""
    role: str     # "user" | "agent"
    content: str


@dataclass
class TaskResult:
    """Matches the TaskResult expected by official evaluators."""
    result: str
    history: list = field(default_factory=list)


@dataclass
class UnifiedEpisodeRewards:
    """Per-episode reward breakdown — shared across all task types."""
    decision: float = 0.0         # correct act/no-act decision (gated on ≥1 GET)
    terminal: float = 0.0         # official evaluator pass
    spurious_action: float = 0.0  # POSTed when ground truth says no action needed
    redundant_lookup: float = 0.0 # repeated identical GET URLs
    invalid_fhir: float = 0.0     # malformed/rejected FHIR call

    # Generic per-task required-action step rewards
    # action_a: primary required action (all tasks with specific required POST)
    # action_b: secondary required action (coupled tasks: task5, task6, v2_task9)
    action_a: float = 0.0
    action_b: float = 0.0
    coupled_missing: float = 0.0  # one of a required pair missing at episode end

    # Task-7 named aliases (kept for backward compat — map to action_a/action_b)
    @property
    def ecg_order(self) -> float:
        return self.action_b   # ECG = secondary action for task7

    @property
    def drug_stop(self) -> float:
        return self.action_a   # drug stop = primary action for task7

    @property
    def total(self) -> float:
        return (
            self.decision + self.terminal
            + self.spurious_action + self.redundant_lookup + self.invalid_fhir
            + self.action_a + self.action_b + self.coupled_missing
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        d["total"] = self.total
        return d


# ─── Ground-truth helpers ─────────────────────────────────────────────────────

def fetch_ground_truth_qtc(
    mrn: str,
    fhir_api_base: str,
    cutoff_ts: str = "2025-01-01T00:00:00+00:00",
) -> Optional[float]:
    """Fetch the latest pre-cutoff QTc value for a patient from FHIR."""
    if _send_get is None:
        return None
    cutoff_dt = datetime.fromisoformat(cutoff_ts)
    url = (
        f"{fhir_api_base}Observation?"
        f"patient={mrn}&code=QTCINTERVAL&_count=5000&_format=json"
    )
    resp = _send_get(url)
    if "error" in resp:
        return None
    try:
        data = resp["data"]
        bundle = json.loads(data) if isinstance(data, str) else data
        entries = bundle.get("entry", [])
    except Exception:
        return None

    valid = []
    for e in entries:
        res = e.get("resource", {})
        ts = res.get("meta", {}).get("lastUpdated")
        if ts:
            try:
                if datetime.fromisoformat(ts) <= cutoff_dt:
                    valid.append(res)
            except ValueError:
                pass

    if not valid:
        return None
    latest = max(valid, key=lambda r: r.get("meta", {}).get("lastUpdated", ""))
    try:
        return latest["valueQuantity"]["value"]
    except (KeyError, TypeError):
        return None


# ─── FHIR parsing helpers ─────────────────────────────────────────────────────

def _parse_fhir_bundle(resp: dict) -> list[dict]:
    """Extract entry resources from a FHIR bundle response."""
    if not resp or "error" in resp:
        return []
    try:
        data = resp.get("data", resp)
        bundle = json.loads(data) if isinstance(data, str) else data
        return [e.get("resource", {}) for e in bundle.get("entry", [])]
    except Exception:
        return []


def _resource_date(resource: dict) -> Optional[datetime]:
    """Best-effort date extraction from a FHIR resource."""
    for field_name in (
        "effectiveDateTime",
        "performedDateTime",
        "authoredOn",
        "recordedDate",
        "occurrenceDateTime",
        "date",
    ):
        val = resource.get(field_name)
        if val:
            try:
                return datetime.fromisoformat(val)
            except ValueError:
                pass
    # performedPeriod
    period = resource.get("performedPeriod") or resource.get("occurrencePeriod", {})
    for key in ("start", "end"):
        val = period.get(key) if isinstance(period, dict) else None
        if val:
            try:
                return datetime.fromisoformat(val)
            except ValueError:
                pass
    # fallback: meta.lastUpdated
    ts = resource.get("meta", {}).get("lastUpdated")
    if ts:
        try:
            return datetime.fromisoformat(ts)
        except ValueError:
            pass
    return None


def _extract_current_time(task: Optional[dict], default_ts: str) -> datetime:
    """
    Parse 'It's <ISO timestamp> now' from the task context field.
    Falls back to default_ts if not found.
    """
    if task:
        ctx = task.get("context") or ""
        m = re.search(r"It'?s\s+([\d\-T:+Z]+)\s+now", ctx)
        if m:
            try:
                return datetime.fromisoformat(m.group(1))
            except ValueError:
                pass
    try:
        return datetime.fromisoformat(default_ts)
    except ValueError:
        return datetime.now(tz=timezone.utc)


def _fhir_get(fhir_api_base: str, resource_type: str, params: str) -> list[dict]:
    """Issue a FHIR GET and return parsed resources."""
    if _send_get is None:
        return []
    url = f"{fhir_api_base}{resource_type}?{params}&_count=1000&_format=json"
    return _parse_fhir_bundle(_send_get(url))


def _before_cutoff(resource: dict, cutoff: datetime) -> bool:
    d = _resource_date(resource)
    if d is None:
        return False
    if d.tzinfo is None:
        d = d.replace(tzinfo=timezone.utc)
    if cutoff.tzinfo is None:
        cutoff = cutoff.replace(tzinfo=timezone.utc)
    return d <= cutoff


# ─── Per-task ground-truth fetchers ──────────────────────────────────────────

def _gt_task1(mrn: str, fhir_api_base: str, cutoff: datetime) -> dict:
    """CT Abdomen more than 12 months before cutoff → should_act."""
    resources = _fhir_get(fhir_api_base, "Procedure",
                          f"patient={mrn}&code=IMGCT0491,IMGIL0001")
    valid = [r for r in resources if _before_cutoff(r, cutoff)]
    if not valid:
        return {"should_act": True, "last_ct_date": None}
    latest = max(valid, key=lambda r: _resource_date(r) or datetime.min.replace(tzinfo=timezone.utc))
    last_date = _resource_date(latest)
    if last_date and last_date.tzinfo is None:
        last_date = last_date.replace(tzinfo=timezone.utc)
    if cutoff.tzinfo is None:
        cutoff = cutoff.replace(tzinfo=timezone.utc)
    should_act = last_date is None or (cutoff - last_date) > timedelta(days=365)
    return {"should_act": should_act, "last_ct_date": last_date.isoformat() if last_date else None}


def _gt_task2(mrn: str, fhir_api_base: str, cutoff: datetime) -> dict:
    """DVT prophylaxis count ≠ 1 → should_act."""
    _ANTICOAG_KEYWORDS = (
        "heparin", "enoxaparin", "lovenox", "fondaparinux", "rivaroxaban",
        "apixaban", "dabigatran", "warfarin", "tinzaparin", "dalteparin",
    )
    resources = _fhir_get(fhir_api_base, "MedicationRequest",
                          f"patient={mrn}&status=active")
    active_anticoag = []
    for r in resources:
        authored = r.get("authoredOn", "")
        try:
            authored_dt = datetime.fromisoformat(authored)
            if authored_dt.tzinfo is None:
                authored_dt = authored_dt.replace(tzinfo=timezone.utc)
            if cutoff.tzinfo is None:
                cutoff = cutoff.replace(tzinfo=timezone.utc)
            if authored_dt > cutoff:
                continue
        except (ValueError, TypeError):
            pass
        blob = json.dumps(r).lower()
        if any(k in blob for k in _ANTICOAG_KEYWORDS):
            active_anticoag.append(r)
    count = len(active_anticoag)
    return {"should_act": count != 1, "anticoag_count": count}


def _gt_task4(mrn: str, fhir_api_base: str, current_time: datetime) -> dict:
    """Urinary catheter in place > 48 h without removal → should_act."""
    resources = _fhir_get(fhir_api_base, "Procedure",
                          f"patient={mrn}&code=NUR1373")
    if current_time.tzinfo is None:
        current_time = current_time.replace(tzinfo=timezone.utc)
    relevant = []
    for r in resources:
        d = _resource_date(r)
        if d is None:
            continue
        if d.tzinfo is None:
            d = d.replace(tzinfo=timezone.utc)
        if d <= current_time:
            relevant.append((d, r))
    if not relevant:
        return {"should_act": False, "catheter_date": None}
    latest_date, latest_r = max(relevant, key=lambda x: x[0])
    # Check the status — if it's already stopped/completed, no action needed
    status = (latest_r.get("status") or "").lower()
    if status in ("stopped", "completed", "cancelled", "entered-in-error"):
        return {"should_act": False, "catheter_date": latest_date.isoformat()}
    age_hours = (current_time - latest_date).total_seconds() / 3600
    return {"should_act": age_hours > 48, "catheter_hours": round(age_hours, 1),
            "catheter_date": latest_date.isoformat()}


def _gt_task5(mrn: str, fhir_api_base: str, cutoff: datetime) -> dict:
    """Has C64.2 diagnosis AND last CT Abdomen/Pelvis > 3 months old → should_act."""
    conditions = _fhir_get(fhir_api_base, "Condition", f"patient={mrn}&code=C64.2")
    has_c64 = any(_before_cutoff(r, cutoff) for r in conditions)
    if not has_c64:
        return {"should_act": False, "has_c64_2": False, "last_ct_date": None}

    ct_resources = _fhir_get(fhir_api_base, "Procedure",
                             f"patient={mrn}&code=IMGCT0491,IMGIL0001,74177")
    valid_cts = [r for r in ct_resources if _before_cutoff(r, cutoff)]
    if not valid_cts:
        return {"should_act": True, "has_c64_2": True, "last_ct_date": None}
    latest = max(valid_cts, key=lambda r: _resource_date(r) or datetime.min.replace(tzinfo=timezone.utc))
    last_date = _resource_date(latest)
    if last_date and last_date.tzinfo is None:
        last_date = last_date.replace(tzinfo=timezone.utc)
    if cutoff.tzinfo is None:
        cutoff = cutoff.replace(tzinfo=timezone.utc)
    should_act = last_date is None or (cutoff - last_date) > timedelta(days=90)
    return {"should_act": should_act, "has_c64_2": True,
            "last_ct_date": last_date.isoformat() if last_date else None}


def _gt_task6(mrn: str, fhir_api_base: str, cutoff: datetime) -> dict:
    """TSH > 10 on two instances ≥ 1 month apart → should_act."""
    resources = _fhir_get(fhir_api_base, "Observation",
                          f"patient={mrn}&code=TSH")
    elevated = []
    for r in resources:
        if not _before_cutoff(r, cutoff):
            continue
        try:
            val = r.get("valueQuantity", {}).get("value")
            if val is not None and float(val) > 10:
                d = _resource_date(r)
                if d:
                    if d.tzinfo is None:
                        d = d.replace(tzinfo=timezone.utc)
                    elevated.append(d)
        except (TypeError, ValueError):
            pass
    if len(elevated) < 2:
        return {"should_act": False, "tsh_elevated_count": len(elevated)}
    elevated.sort()
    # Check if any two are ≥ 1 month (30 days) apart
    found_pair = any(
        (elevated[j] - elevated[i]).days >= 30
        for i in range(len(elevated))
        for j in range(i + 1, len(elevated))
    )
    return {"should_act": found_pair, "tsh_elevated_count": len(elevated)}


def _gt_task8(mrn: str, fhir_api_base: str, cutoff: datetime) -> dict:
    """Active opioid without a matching naloxone order → should_act."""
    _OPIOIDS = ("hydromorphone", "oxycodone", "fentanyl", "hydrocodone", "morphine")
    resources = _fhir_get(fhir_api_base, "MedicationRequest",
                          f"patient={mrn}&status=active")
    if cutoff.tzinfo is None:
        cutoff = cutoff.replace(tzinfo=timezone.utc)
    active_opioids = []
    has_naloxone = False
    for r in resources:
        authored = r.get("authoredOn", "")
        try:
            authored_dt = datetime.fromisoformat(authored)
            if authored_dt.tzinfo is None:
                authored_dt = authored_dt.replace(tzinfo=timezone.utc)
            if authored_dt > cutoff:
                continue
        except (ValueError, TypeError):
            pass
        blob = json.dumps(r).lower()
        if any(k in blob for k in _OPIOIDS):
            active_opioids.append(r)
        if "naloxone" in blob:
            has_naloxone = True
    should_act = len(active_opioids) > 0 and not has_naloxone
    return {"should_act": should_act, "opioid_count": len(active_opioids),
            "has_naloxone": has_naloxone}


def _gt_task9(mrn: str, fhir_api_base: str, current_time: datetime) -> dict:
    """Flu vaccine > 365 days ago → should_act."""
    resources = _fhir_get(fhir_api_base, "Procedure", f"patient={mrn}&code=90686")
    if current_time.tzinfo is None:
        current_time = current_time.replace(tzinfo=timezone.utc)
    valid = [r for r in resources if _before_cutoff(r, current_time)]
    if not valid:
        return {"should_act": True, "last_flu_date": None}
    latest = max(valid, key=lambda r: _resource_date(r) or datetime.min.replace(tzinfo=timezone.utc))
    last_date = _resource_date(latest)
    if last_date and last_date.tzinfo is None:
        last_date = last_date.replace(tzinfo=timezone.utc)
    should_act = last_date is None or (current_time - last_date) > timedelta(days=365)
    return {"should_act": should_act,
            "last_flu_date": last_date.isoformat() if last_date else None}


def _gt_task10(mrn: str, fhir_api_base: str, current_time: datetime) -> dict:
    """COVID vaccine > 12 months ago → should_act."""
    resources = _fhir_get(fhir_api_base, "Procedure", f"patient={mrn}&code=COVIDVACCINE")
    if current_time.tzinfo is None:
        current_time = current_time.replace(tzinfo=timezone.utc)
    valid = [r for r in resources if _before_cutoff(r, current_time)]
    if not valid:
        return {"should_act": True, "last_covid_vax_date": None}
    latest = max(valid, key=lambda r: _resource_date(r) or datetime.min.replace(tzinfo=timezone.utc))
    last_date = _resource_date(latest)
    if last_date and last_date.tzinfo is None:
        last_date = last_date.replace(tzinfo=timezone.utc)
    should_act = last_date is None or (current_time - last_date) > timedelta(days=365)
    return {"should_act": should_act,
            "last_covid_vax_date": last_date.isoformat() if last_date else None}


# ─── test_data_v2.json task fetchers (v2_ prefix) ────────────────────────────

def _gt_v2_task5(mrn: str, fhir_api_base: str, current_time: datetime) -> dict:
    """
    Mg level in last 24 h. If present and low (< 2.0 mg/dL) → should_act.
    If no reading in last 24 h → should_act = False (task says don't order anything).
    """
    resources = _fhir_get(fhir_api_base, "Observation", f"patient={mrn}&code=MG")
    if current_time.tzinfo is None:
        current_time = current_time.replace(tzinfo=timezone.utc)
    window_start = current_time - timedelta(hours=24)
    recent = []
    for r in resources:
        d = _resource_date(r)
        if d is None:
            continue
        if d.tzinfo is None:
            d = d.replace(tzinfo=timezone.utc)
        if window_start <= d <= current_time:
            try:
                val = float(r.get("valueQuantity", {}).get("value", float("nan")))
                recent.append((d, val))
            except (TypeError, ValueError):
                pass
    if not recent:
        return {"should_act": False, "mg_value": None, "reason": "no_recent_mg"}
    _, latest_mg = max(recent, key=lambda x: x[0])
    return {"should_act": latest_mg < 2.0, "mg_value": latest_mg}


def _gt_v2_task9(mrn: str, fhir_api_base: str, current_time: datetime) -> dict:
    """Most recent K+ < 3.5 mEq/L → should_act."""
    resources = _fhir_get(fhir_api_base, "Observation", f"patient={mrn}&code=K")
    if current_time.tzinfo is None:
        current_time = current_time.replace(tzinfo=timezone.utc)
    valid = []
    for r in resources:
        d = _resource_date(r)
        if d is None:
            continue
        if d.tzinfo is None:
            d = d.replace(tzinfo=timezone.utc)
        if d <= current_time:
            try:
                val = float(r.get("valueQuantity", {}).get("value", float("nan")))
                valid.append((d, val))
            except (TypeError, ValueError):
                pass
    if not valid:
        return {"should_act": None, "k_value": None}  # no data, can't determine
    _, latest_k = max(valid, key=lambda x: x[0])
    return {"should_act": latest_k < 3.5, "k_value": latest_k}


def _gt_v2_task10(mrn: str, fhir_api_base: str, current_time: datetime) -> dict:
    """Last A1C result date > 1 year before current_time → should_act."""
    resources = _fhir_get(fhir_api_base, "Observation", f"patient={mrn}&code=A1C")
    if current_time.tzinfo is None:
        current_time = current_time.replace(tzinfo=timezone.utc)
    valid = []
    for r in resources:
        d = _resource_date(r)
        if d is None:
            continue
        if d.tzinfo is None:
            d = d.replace(tzinfo=timezone.utc)
        if d <= current_time:
            valid.append(d)
    if not valid:
        return {"should_act": True, "last_a1c_date": None}
    last_date = max(valid)
    should_act = (current_time - last_date) > timedelta(days=365)
    return {"should_act": should_act, "last_a1c_date": last_date.isoformat()}


# ─── Dispatcher ───────────────────────────────────────────────────────────────

def fetch_ground_truth_state(
    task_type: str,
    mrn: str,
    fhir_api_base: str,
    cutoff_ts: str = "2025-01-01T00:00:00+00:00",
    qt_threshold: int = 500,
    task: Optional[dict] = None,
) -> dict:
    """
    Fetch ground-truth patient state for reward shaping.

    Returns a dict with at minimum:
        {"should_act": bool | None, ...task_specific_fields}

    `should_act = None` means we couldn't determine it (evaluator is
    the sole signal for this task).
    """
    if _send_get is None:
        return {"should_act": None}

    try:
        cutoff_dt = datetime.fromisoformat(cutoff_ts)
        if cutoff_dt.tzinfo is None:
            cutoff_dt = cutoff_dt.replace(tzinfo=timezone.utc)

        if task_type == "task7":
            qtc = fetch_ground_truth_qtc(mrn, fhir_api_base, cutoff_ts)
            if qtc is None:
                return {"should_act": None, "qtc": None}
            return {"should_act": qtc > qt_threshold, "qtc": qtc}

        elif task_type == "task1":
            return _gt_task1(mrn, fhir_api_base, cutoff_dt)

        elif task_type == "task2":
            return _gt_task2(mrn, fhir_api_base, cutoff_dt)

        elif task_type == "task4":
            current_time = _extract_current_time(task, cutoff_ts)
            return _gt_task4(mrn, fhir_api_base, current_time)

        elif task_type == "task5":
            return _gt_task5(mrn, fhir_api_base, cutoff_dt)

        elif task_type == "task6":
            return _gt_task6(mrn, fhir_api_base, cutoff_dt)

        elif task_type == "task8":
            return _gt_task8(mrn, fhir_api_base, cutoff_dt)

        elif task_type == "task9":
            current_time = _extract_current_time(task, cutoff_ts)
            return _gt_task9(mrn, fhir_api_base, current_time)

        elif task_type == "task10":
            current_time = _extract_current_time(task, cutoff_ts)
            return _gt_task10(mrn, fhir_api_base, current_time)

        # ── test_data_v2.json tasks (Mg / K+ / A1c) ──────────────────
        elif task_type == "v2_task5":
            current_time = _extract_current_time(task, cutoff_ts)
            return _gt_v2_task5(mrn, fhir_api_base, current_time)

        elif task_type == "v2_task9":
            current_time = _extract_current_time(task, cutoff_ts)
            return _gt_v2_task9(mrn, fhir_api_base, current_time)

        elif task_type == "v2_task10":
            current_time = _extract_current_time(task, cutoff_ts)
            return _gt_v2_task10(mrn, fhir_api_base, current_time)

    except Exception as exc:
        print(f"  [WARN] fetch_ground_truth_state({task_type}, {mrn}): {exc}")

    return {"should_act": None}


# ─── Unified reward tracker ───────────────────────────────────────────────────

@dataclass
class TaskRewardTracker:
    """
    Mutable per-episode state that accumulates rewards for any RL task type.

    Usage:
        tracker = TaskRewardTracker(cfg, task_type="task7",
                                    ground_truth={"should_act": True, "qtc": 520})
        for each tool call:
            r = tracker.on_tool_call(tool_name, tool_args, is_post, url, succeeded)
        terminal_r = tracker.on_episode_end(task, history, fhir_api_base)
    """
    cfg: RewardConfig
    task_type: str
    ground_truth: dict = field(default_factory=dict)

    # ── Universal tracking ──────────────────────────────────────────────
    _made_any_get: bool = field(default=False, init=False)
    _any_post_made: bool = field(default=False, init=False)
    _seen_get_urls: dict = field(default_factory=dict, init=False)

    # ── Per-task required-action tracking (all tasks) ───────────────
    _found_action_a: bool = field(default=False, init=False)
    _found_action_b: bool = field(default=False, init=False)

    rewards: UnifiedEpisodeRewards = field(
        default_factory=UnifiedEpisodeRewards, init=False
    )

    @property
    def should_act(self) -> Optional[bool]:
        return self.ground_truth.get("should_act")

    def on_tool_call(
        self,
        tool_name: str,
        tool_args: dict,
        is_post: bool,
        url: str,
        tool_succeeded: bool,
    ) -> float:
        """
        Called after each tool execution.
        Returns the incremental reward from this step.
        """
        step_reward = 0.0

        if not tool_succeeded:
            self.rewards.invalid_fhir += self.cfg.invalid_fhir
            return self.cfg.invalid_fhir

        # ── Track any GET (task-agnostic — agent learns which tool to use) ─
        if not is_post:
            self._made_any_get = True

        # ── Redundancy penalty for repeated GET calls ──────────────────
        if not is_post and url:
            count = self._seen_get_urls.get(url, 0)
            self._seen_get_urls[url] = count + 1
            if count >= 1:  # second+ call to same URL
                # Only penalise up to the cap
                if self.rewards.redundant_lookup > self.cfg.redundant_lookup_cap:
                    pass
                else:
                    self.rewards.redundant_lookup += self.cfg.redundant_lookup
                    step_reward += self.cfg.redundant_lookup

        # ── POST tracking ──────────────────────────────────────────────
        if is_post:
            self._any_post_made = True

            # Spurious action: fire immediately when ground truth says no-act
            if self.should_act is False:
                self.rewards.spurious_action += self.cfg.spurious_action
                step_reward += self.cfg.spurious_action

        # ── Per-task required-action step rewards ──────────────────────
        # Only award when should_act is known True — spurious_action already
        # penalises incorrect POSTs when should_act is False.
        if is_post and self.should_act:
            step_reward += self._task_step_reward(tool_name, tool_args)

        return step_reward

    # ── Tasks that require exactly two coupled actions ─────────────────
    _COUPLED_TASKS = frozenset({"task5", "task6", "task7", "v2_task9"})

    def _task_step_reward(self, tool_name: str, tool_args: dict) -> float:
        """
        Detect task-specific required actions and award action_a / action_b.
        Each fires at most once per episode (idempotent).
        """
        blob = json.dumps(tool_args).lower()
        r = 0.0

        def award_a() -> float:
            nonlocal r
            if not self._found_action_a:
                self._found_action_a = True
                self.rewards.action_a = self.cfg.action_a
                r += self.cfg.action_a
            return r

        def award_b() -> float:
            nonlocal r
            if not self._found_action_b:
                self._found_action_b = True
                self.rewards.action_b = self.cfg.action_b
                r += self.cfg.action_b
            return r

        tt = self.task_type

        if tt == "task1":
            # Order new CT Abdomen (CPT 74177)
            if tool_name == "fhir_service_request_create" and "74177" in blob:
                award_a()

        elif tt == "task2":
            # Create heparin order (0 orders case) or discontinue duplicate (2+ case)
            if tool_name == "fhir_medication_request_create":
                _ANTICOAG = ("heparin", "enoxaparin", "lovenox", "fondaparinux",
                             "rivaroxaban", "apixaban", "dabigatran", "warfarin")
                if any(k in blob for k in _ANTICOAG):
                    award_a()

        elif tt == "task4":
            # ServiceRequest to remove catheter
            if tool_name == "fhir_service_request_create":
                if "catheter" in blob or "nur1373" in blob or "removal" in blob:
                    award_a()

        elif tt == "task5":
            # action_a: CT Abdomen/Pelvis order (74177)
            # action_b: IR referral (CON417)
            if tool_name == "fhir_service_request_create":
                if "74177" in blob:
                    award_a()
                if "con417" in blob or "interventional radiology" in blob:
                    award_b()

        elif tt == "task6":
            # action_a: levothyroxine MedicationRequest
            # action_b: repeat TSH/FT4 order
            if tool_name == "fhir_medication_request_create" and "levothyroxine" in blob:
                award_a()
            if tool_name == "fhir_service_request_create":
                if "tsh" in blob or "ft4" in blob or "thyroid" in blob:
                    award_b()

        elif tt == "task7":
            # action_a: drug stop (QT-prolonging med discontinued)
            # action_b: ECG ServiceRequest
            if tool_name == "fhir_medication_request_create":
                status = (tool_args.get("status") or "").lower()
                if status in set(self.cfg.non_active_statuses):
                    if any(w in blob for w in self.cfg.qt_prolonging_meds):
                        award_a()
            if tool_name == "fhir_service_request_create":
                if self.cfg.ecg_snomed_code in blob or "ecg" in blob:
                    award_b()

        elif tt == "task8":
            # Naloxone MedicationRequest
            if tool_name == "fhir_medication_request_create" and "naloxone" in blob:
                award_a()

        elif tt == "task9":
            # Influenza vaccine ServiceRequest (CPT 90686)
            if tool_name == "fhir_service_request_create":
                if "90686" in blob or "influenza" in blob:
                    award_a()

        elif tt == "task10":
            # COVID booster ServiceRequest (CPT 91320)
            if tool_name == "fhir_service_request_create":
                if "91320" in blob or "covid" in blob:
                    award_a()

        elif tt == "v2_task5":
            # IV magnesium replacement MedicationRequest
            if tool_name == "fhir_medication_request_create":
                if "0338-1715-40" in blob or "magnesium" in blob:
                    award_a()

        elif tt == "v2_task9":
            # action_a: potassium replacement MedicationRequest
            # action_b: morning serum potassium lab order
            if tool_name == "fhir_medication_request_create":
                if "40032-917-01" in blob or "potassium" in blob:
                    award_a()
            if tool_name == "fhir_service_request_create":
                if "potassium" in blob or "2823-3" in blob:  # LOINC serum K+
                    award_b()

        elif tt == "v2_task10":
            # A1C lab order (LOINC 4548-4)
            if "4548-4" in blob or "a1c" in blob or "hba1c" in blob:
                award_a()

        return r

    def on_episode_end(
        self,
        task: dict,
        history: list[HistoryItem],
        fhir_api_base: str,
    ) -> float:
        """
        Called once at episode end.
        Computes decision, coupled-missing (task7), and terminal rewards.
        Returns total terminal-phase reward delta.
        """
        terminal_reward = 0.0

        # ── Official terminal evaluator ────────────────────────────────
        evaluator = _try_import_evaluator(self.task_type)
        terminal_passed = False
        if evaluator is not None:
            task_result = TaskResult(result="[]", history=history)
            try:
                terminal_passed = evaluator(task, task_result, fhir_api_base)
            except Exception as exc:
                print(f"  [WARN] {self.task_type}() evaluator raised: {exc}")

        if terminal_passed:
            self.rewards.terminal = self.cfg.terminal_pass
            terminal_reward += self.cfg.terminal_pass

        # ── Decision reward (gated on having read the chart) ──────────
        # Only fires when ground truth is known (should_act is not None).
        # Gated on _made_any_get — agent must look at FHIR data before getting
        # credit for a correct no-action.  The gate does NOT specify which tool
        # or code to use; the agent learns that from the task instruction.
        if self.should_act is not None and self._made_any_get:
            correct_decision = (
                (self.should_act and self._any_post_made)
                or (not self.should_act and not self._any_post_made)
            )
            if correct_decision:
                self.rewards.decision = self.cfg.decision
                terminal_reward += self.cfg.decision

        # ── Coupled-action penalty (tasks requiring two actions) ───────
        # Fires when should_act=True but agent placed only one of the pair.
        if self.task_type in self._COUPLED_TASKS and self.should_act:
            if self._found_action_a != self._found_action_b:  # XOR
                self.rewards.coupled_missing = self.cfg.coupled_missing
                terminal_reward += self.cfg.coupled_missing

        return terminal_reward

    def get_total_reward(self) -> float:
        return self.rewards.total


# ─── Backward-compat alias ────────────────────────────────────────────────────
# env.py currently imports EpisodeRewardTracker — keep it working.

@dataclass
class EpisodeRewardTracker(TaskRewardTracker):
    """
    Backward-compatible shim.  Accepts the old (cfg, is_prolonged, true_qtc)
    signature and maps it to TaskRewardTracker(cfg, "task7", ground_truth).
    """
    is_prolonged: bool = False
    true_qtc: Optional[float] = None

    def __post_init__(self):
        # Synthesise ground_truth from the old fields
        self.task_type = "task7"
        self.ground_truth = {
            "should_act": self.is_prolonged if self.true_qtc is not None else None,
            "qtc": self.true_qtc,
        }

    # Keep old EpisodeRewards shape accessible for callers that read .rewards
    # directly — UnifiedEpisodeRewards is a superset, so no adaptation needed.
