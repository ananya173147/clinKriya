"""
Mock FHIR server backed by a cached response database.

Eliminates the need for a running FHIR Docker container during training.
Cache is built once against the real server, then used for all subsequent
training runs.

Usage:
    # Build cache (requires real FHIR server running):
    python -m medagentbench_env.server.fhir_cache --build \
        --fhir-url http://localhost:8080/fhir/ \
        --output cache.json

    # In the environment, use MockFHIR instead of real requests:
    mock = MockFHIR.from_cache("cache.json")
    result = mock.get("http://localhost:8080/fhir/Observation?patient=S123&code=A1C")
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

import requests


# ---------------------------------------------------------------------------
# Cache builder
# ---------------------------------------------------------------------------

def _get_all_mrns(tasks: List[Dict]) -> set:
    """Extract all unique patient MRNs from the task dataset."""
    return {t["eval_MRN"] for t in tasks if t.get("eval_MRN")}


def _build_cache_entries(fhir_api_base: str, tasks: List[Dict]) -> Dict[str, Any]:
    """Query the real FHIR server and cache all responses needed for
    evaluation and typical agent interactions.

    Returns a dict mapping normalized URL → response data.
    """
    cache: Dict[str, Any] = {}
    mrns = _get_all_mrns(tasks)
    fhir_base = fhir_api_base.rstrip("/")

    # ---- Patterns needed by evaluators and agents ----
    # _count=5000 ensures we retrieve all records, not just the default page size.

    resource_queries = [
        # ── Unfiltered (agent exploration + fallback) ────────────────
        ("Patient",            {"_count": "5000", "_format": "json"}),
        ("Condition",          {"_count": "5000", "_format": "json"}),
        ("MedicationRequest",  {"_count": "5000", "_format": "json"}),
        ("Procedure",          {"_count": "5000", "_format": "json"}),
        ("Observation",        {"_count": "5000", "_format": "json"}),
        # Server has no Immunization resources (vaccinations stored as Procedure),
        # but cached so agents can query it without a cache miss.
        ("Immunization",       {"_count": "5000", "_format": "json"}),

        # ── Active medications (task2, task8) ────────────────────────
        ("MedicationRequest",  {"status": "active", "_count": "5000", "_format": "json"}),

        # ── Observation codes (server uses local codes, not LOINC) ───
        ("Observation", {"code": "A1C",         "_count": "5000", "_format": "json"}),  # task10, v2_task10
        ("Observation", {"code": "QTCINTERVAL", "_count": "5000", "_format": "json"}),  # task7
        ("Observation", {"code": "QTINTERVAL",  "_count": "5000", "_format": "json"}),  # task7 raw QT
        ("Observation", {"code": "TSH",         "_count": "5000", "_format": "json"}),  # task6
        ("Observation", {"code": "FT4",         "_count": "5000", "_format": "json"}),  # task6
        ("Observation", {"code": "MG",          "_count": "5000", "_format": "json"}),  # v2_task5
        ("Observation", {"code": "K",           "_count": "5000", "_format": "json"}),  # v2_task9
        ("Observation", {"code": "HEARTRATE",   "_count": "5000", "_format": "json"}),  # task3
        ("Observation", {"code": "BP",          "_count": "5000", "_format": "json"}),  # vitals
        ("Observation", {"category": "vital-signs", "_count": "5000", "_format": "json"}),

        # ── Procedure codes ──────────────────────────────────────────
        ("Procedure", {"code": "IMGCT0491",   "_count": "5000", "_format": "json"}),  # task1, task5
        ("Procedure", {"code": "IMGIL0001",   "_count": "5000", "_format": "json"}),  # task1, task5
        ("Procedure", {"code": "NUR1373",     "_count": "5000", "_format": "json"}),  # task4
        ("Procedure", {"code": "90686",       "_count": "5000", "_format": "json"}),  # task9 flu vax
        ("Procedure", {"code": "COVIDVACCINE","_count": "5000", "_format": "json"}),  # task10 covid vax

        # ── Condition codes ──────────────────────────────────────────
        ("Condition", {"code": "C64.2",       "_count": "5000", "_format": "json"}),  # task5
        ("Condition", {"category": "problem-list-item", "_count": "5000", "_format": "json"}),
    ]

    total = len(mrns) * len(resource_queries)
    done = 0

    for mrn in sorted(mrns):
        # Also cache patient lookup by identifier
        patient_url = f"{fhir_base}/Patient?identifier={mrn}&_format=json"
        _fetch_and_cache(patient_url, cache)

        for resource, params in resource_queries:
            query_params = {**params, "patient": mrn}
            param_str = "&".join(f"{k}={v}" for k, v in sorted(query_params.items()))
            url = f"{fhir_base}/{resource}?{param_str}"
            _fetch_and_cache(url, cache)
            done += 1
            if done % 50 == 0:
                print(f"  Cached {done}/{total} queries...")

    # Cache the metadata endpoint (used for health checks)
    _fetch_and_cache(f"{fhir_base}/metadata", cache)
    _fetch_and_cache(f"{fhir_base}/metadata?_format=json", cache)

    print(f"Cache built: {len(cache)} entries")
    return cache


def _fetch_and_cache(url: str, cache: Dict[str, Any]) -> None:
    """Fetch a URL and store the response in the cache."""
    key = _normalize_url(url)
    if key in cache:
        return
    try:
        resp = requests.get(url, timeout=30)
        content_type = resp.headers.get("Content-Type", "")
        if "json" in content_type:
            data = resp.json()
        else:
            data = resp.text
        cache[key] = {
            "status_code": resp.status_code,
            "data": data,
        }
    except Exception as e:
        cache[key] = {"error": str(e)}


def _normalize_url(url: str) -> str:
    """Normalize a URL for consistent cache lookups.

    Sorts query parameters so the same logical query always maps to
    the same cache key regardless of parameter order.
    """
    parsed = urlparse(url)
    params = parse_qs(parsed.query, keep_blank_values=True)
    # Flatten single-value lists and sort
    flat = {k: v[0] if len(v) == 1 else v for k, v in sorted(params.items())}
    sorted_query = "&".join(f"{k}={v}" for k, v in sorted(flat.items()))
    return f"{parsed.scheme}://{parsed.netloc}{parsed.path}?{sorted_query}" if sorted_query else f"{parsed.scheme}://{parsed.netloc}{parsed.path}"


# ---------------------------------------------------------------------------
# Mock FHIR client
# ---------------------------------------------------------------------------

class MockFHIR:
    """Mock FHIR client that returns cached responses.

    Falls back to a generic empty Bundle for uncached GET queries
    (so the agent can still explore without crashing).
    """

    def __init__(self, cache: Dict[str, Any], fhir_api_base: str = ""):
        self._cache = cache
        self._fhir_api_base = fhir_api_base.rstrip("/")
        # Lazy index: (family.lower, given.lower, birthDate) → identifier-cache value.
        # Built on first name+DOB Patient lookup to support v1_task1
        # (cache only contains Patient?identifier= queries; no name+DOB queries
        # were captured when the cache was built against the live HAPI server).
        self._patient_name_index: Optional[Dict[tuple, Dict[str, Any]]] = None

    @classmethod
    def from_cache(cls, cache_path: str, fhir_api_base: str = "") -> "MockFHIR":
        with open(cache_path) as f:
            cache = json.load(f)
        return cls(cache, fhir_api_base)

    # Resource types that require a patient parameter to be a valid query
    _PATIENT_REQUIRED = frozenset({
        "Observation", "MedicationRequest", "Procedure", "Condition",
        "ServiceRequest", "AllergyIntolerance", "Immunization", "DiagnosticReport",
    })

    def get(self, url: str) -> Dict[str, Any]:
        """Look up a cached response for the given URL.

        Returns dict with 'status_code' and 'data', or an error dict if the
        query is structurally invalid (missing required patient param), or an
        empty FHIR Bundle for valid but uncached queries.
        """
        key = _normalize_url(url)

        # Exact match
        if key in self._cache:
            return self._cache[key]

        # Try without _format parameter (often appended dynamically)
        stripped = re.sub(r'[&?]_format=json', '', key).rstrip('?').rstrip('&')
        if stripped in self._cache:
            return self._cache[stripped]

        # Try matching just the path + essential params (patient, code)
        fuzzy_match = self._fuzzy_lookup(key)
        if fuzzy_match is not None:
            return fuzzy_match

        # Patient name+DOB search — synthesize from the identifier-cache.
        parsed = urlparse(key)
        path = parsed.path.rstrip("/").split("/")[-1]
        params = parse_qs(parsed.query)
        if path == "Patient" and "identifier" not in params and (
            "family" in params or "given" in params or "birthdate" in params
        ):
            synth = self._patient_name_dob_lookup(params)
            if synth is not None:
                return synth

        # Multi-code query (`code=A,B`): merge per-code lookups.
        # The HAPI FHIR server interprets a comma-separated list as OR-of-codes,
        # but the cache only stores per-code keys. Without this, the v2-new
        # graders for task1/9/10 (which query e.g. code=IMGCT0491,IMGIL0001)
        # always return empty Bundles, breaking the no-action branch grader.
        if "code" in params and "," in (params["code"][0] or ""):
            merged = self._merge_multi_code_query(key, params)
            if merged is not None:
                return merged

        # Return an error for clinical resource queries missing the patient param —
        # this activates the invalid_fhir penalty so malformed queries get penalised.
        if path in self._PATIENT_REQUIRED and "patient" not in params:
            return {"error": f"Missing required 'patient' parameter for {path} query"}

        # Fallback: return an empty FHIR Bundle (valid response, no data)
        return {
            "status_code": 200,
            "data": {
                "resourceType": "Bundle",
                "type": "searchset",
                "total": 0,
                "entry": [],
            },
        }

    def _build_patient_name_index(self) -> Dict[tuple, Dict[str, Any]]:
        """Index identifier-cache Patient entries by (family, given, birthDate)."""
        index: Dict[tuple, Dict[str, Any]] = {}
        for cached_key, cached_val in self._cache.items():
            if "/Patient?" not in cached_key or "identifier=" not in cached_key:
                continue
            data = cached_val.get("data") if isinstance(cached_val, dict) else None
            if isinstance(data, str):
                try: data = json.loads(data)
                except Exception: continue
            if not isinstance(data, dict): continue
            for entry in data.get("entry", []):
                res = entry.get("resource", {})
                dob = res.get("birthDate")
                for name in res.get("name", []):
                    family = (name.get("family") or "").lower()
                    for given in name.get("given", []):
                        index[(family, given.lower(), dob)] = cached_val
                        # Also key on family alone or given alone for partial-name lookups
        return index

    def _merge_multi_code_query(self, key: str, params: Dict[str, list]) -> Optional[Dict[str, Any]]:
        """For a query with `code=A,B,C`, look up each code individually and
        union the resulting Bundle entries. Returns a merged Bundle, or None
        if no per-code keys hit."""
        codes = [c.strip() for c in params["code"][0].split(",") if c.strip()]
        merged_entries = []
        seen_ids = set()
        any_hit = False
        for code in codes:
            new_params = dict(params)
            new_params["code"] = [code]
            qs = "&".join(
                f"{k}={v[0]}" for k, v in sorted(new_params.items())
            )
            sub_key = key.split("?")[0] + "?" + qs
            sub_norm = _normalize_url(sub_key)
            sub_resp = self._cache.get(sub_norm) or self._cache.get(
                re.sub(r'[&?]_format=json', '', sub_norm).rstrip('?').rstrip('&')
            )
            if not sub_resp:
                # Try fuzzy with single code
                sub_resp = self._fuzzy_lookup(sub_norm)
            if not sub_resp:
                continue
            any_hit = True
            data = sub_resp.get("data", {})
            if isinstance(data, str):
                try: data = json.loads(data)
                except Exception: continue
            for entry in data.get("entry", []) or []:
                rid = entry.get("resource", {}).get("id")
                if rid in seen_ids: continue
                seen_ids.add(rid)
                merged_entries.append(entry)
        if not any_hit:
            return None
        return {
            "status_code": 200,
            "data": {
                "resourceType": "Bundle", "type": "searchset",
                "total": len(merged_entries), "entry": merged_entries,
            },
        }

    def _patient_name_dob_lookup(self, params: Dict[str, list]) -> Optional[Dict[str, Any]]:
        """Synthesize a name+DOB Patient search response from the identifier-cache.

        Returns a search Bundle wrapping the matched Patient, or an empty Bundle
        if no match exists in the cache. None if the cache index is unusable.
        """
        if self._patient_name_index is None:
            self._patient_name_index = self._build_patient_name_index()
        family = (params.get("family", [""])[0] or "").lower()
        given = (params.get("given", [""])[0] or "").lower()
        birthdate = params.get("birthdate", [""])[0] or ""
        # Require at least family+birthdate or given+birthdate to avoid spurious matches
        if not birthdate or not (family or given):
            return None
        cached_val = self._patient_name_index.get((family, given, birthdate))
        if cached_val is None:
            return {
                "status_code": 200,
                "data": {
                    "resourceType": "Bundle", "type": "searchset",
                    "total": 0, "entry": [],
                },
            }
        # The cached identifier-lookup IS already a search Bundle of total=1; reuse it.
        return cached_val

    def _fuzzy_lookup(self, key: str) -> Optional[Dict[str, Any]]:
        """Match on resource path + patient + code (symmetric) + category (symmetric).

        Both code and category are matched symmetrically: if the query specifies
        a value, the cached entry must have the same value; if the query omits it,
        the cached entry must also omit it. This prevents broad no-code queries
        from returning code-specific cached entries, which would hide incorrect
        query patterns from the agent.
        """
        parsed = urlparse(key)
        params = parse_qs(parsed.query)
        patient = params.get("patient", [None])[0]
        code = params.get("code", [None])[0]
        category = params.get("category", [None])[0]
        path = parsed.path.rstrip("/").split("/")[-1]  # e.g. "Observation"

        if not patient:
            return None

        for cached_key, cached_val in self._cache.items():
            cached_parsed = urlparse(cached_key)
            cached_params = parse_qs(cached_parsed.query)
            cached_path = cached_parsed.path.rstrip("/").split("/")[-1]

            if cached_path != path:
                continue
            if cached_params.get("patient", [None])[0] != patient:
                continue
            if code != cached_params.get("code", [None])[0]:
                continue
            if category != cached_params.get("category", [None])[0]:
                continue
            return cached_val

        return None


# ---------------------------------------------------------------------------
# Replacement for _send_get_request that uses the mock
# ---------------------------------------------------------------------------

def mock_send_get_request(mock: MockFHIR, url: str) -> Dict[str, Any]:
    """Drop-in replacement for _send_get_request using cached data."""
    return mock.get(url)


# ---------------------------------------------------------------------------
# CLI for building cache
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build FHIR response cache")
    parser.add_argument(
        "--build", action="store_true",
        help="Build the cache from a running FHIR server",
    )
    parser.add_argument(
        "--fhir-url", type=str, default="http://localhost:8080/fhir/",
        help="FHIR server base URL",
    )
    parser.add_argument(
        "--data-file", type=str, default=None,
        help="Path to new_patient_tasks.json (default: auto-detect)",
    )
    parser.add_argument(
        "--output", type=str, default="data/fhir_cache.json",
        help="Output cache file path",
    )
    args = parser.parse_args()

    if not args.build:
        parser.print_help()
        return

    # Load task data — new_patient_tasks.json + v2 tasks for full MRN coverage
    _HERE = Path(__file__).resolve()
    _ROOT = _HERE.parents[2]

    if args.data_file:
        data_path = Path(args.data_file)
    else:
        data_path = _HERE.parents[1] / "data" / "new_patient_tasks.json"

    print(f"Loading tasks from {data_path}")
    with open(data_path) as f:
        tasks = json.load(f)

    # Also include v2 task patients (Mg/K+/A1c — different MRN set)
    v2_path = (
        _ROOT / "medagentbenchv2" / "medagentbench_v2" / "src"
        / "MedAgentBench" / "data" / "medagentbench" / "test_data_v2.json"
    )
    if v2_path.exists():
        with open(v2_path) as f:
            v2_tasks = json.load(f)
        tasks = tasks + v2_tasks
        print(f"Also loaded {len(v2_tasks)} v2 tasks")

    print(f"Loaded {len(tasks)} total tasks with {len(_get_all_mrns(tasks))} unique MRNs")

    print(f"Building cache from {args.fhir_url}...")
    cache = _build_cache_entries(args.fhir_url, tasks)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(cache, f)
    print(f"Cache saved to {output_path} ({output_path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
