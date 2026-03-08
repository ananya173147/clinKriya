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

    # All FHIR resource types the agent might query
    resource_queries = [
        # Task 10: A1C observations (required by evaluator)
        ("Observation", {"code": "A1C", "_count": "5000", "_format": "json"}),
        # Common agent queries for context
        ("Observation", {"category": "vital-signs", "_format": "json"}),
        ("Observation", {"code": "BP", "_format": "json"}),
        ("Observation", {"code": "BP", "_count": "5000", "_format": "json"}),
        ("MedicationRequest", {"_format": "json"}),
        ("Condition", {"category": "problem-list-item", "_format": "json"}),
        ("Condition", {"_format": "json"}),
        ("Patient", {"_format": "json"}),
        ("Procedure", {"_format": "json"}),
        # Task 8: agent might look up imaging/radiology
        ("Observation", {"code": "IMAGINGCODE", "_format": "json"}),
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

    @classmethod
    def from_cache(cls, cache_path: str, fhir_api_base: str = "") -> "MockFHIR":
        with open(cache_path) as f:
            cache = json.load(f)
        return cls(cache, fhir_api_base)

    def get(self, url: str) -> Dict[str, Any]:
        """Look up a cached response for the given URL.

        Returns dict with 'status_code' and 'data', or a fallback
        empty FHIR Bundle if the URL isn't cached.
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

    def _fuzzy_lookup(self, key: str) -> Optional[Dict[str, Any]]:
        """Try to match by resource type + patient MRN + code."""
        parsed = urlparse(key)
        params = parse_qs(parsed.query)
        patient = params.get("patient", [None])[0]
        code = params.get("code", [None])[0]
        path = parsed.path.rstrip("/").split("/")[-1]  # e.g. "Observation"

        if not patient:
            return None

        for cached_key, cached_val in self._cache.items():
            cached_parsed = urlparse(cached_key)
            cached_params = parse_qs(cached_parsed.query)
            cached_path = cached_parsed.path.rstrip("/").split("/")[-1]

            if (cached_path == path
                and cached_params.get("patient", [None])[0] == patient
                and (code is None or cached_params.get("code", [None])[0] == code)):
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
        help="Path to stratified_benchmark.json",
    )
    parser.add_argument(
        "--output", type=str, default="data/fhir_cache.json",
        help="Output cache file path",
    )
    args = parser.parse_args()

    if not args.build:
        parser.print_help()
        return

    # Load task data
    if args.data_file:
        data_path = Path(args.data_file)
    else:
        data_path = (
            Path(__file__).resolve().parents[2]
            / "medagentbenchv2"
            / "medagentbench_v2"
            / "src"
            / "MedAgentBench"
            / "data"
            / "medagentbench"
            / "stratified_benchmark.json"
        )

    print(f"Loading tasks from {data_path}")
    with open(data_path) as f:
        tasks = json.load(f)
    print(f"Loaded {len(tasks)} tasks with {len(_get_all_mrns(tasks))} unique MRNs")

    print(f"Building cache from {args.fhir_url}...")
    cache = _build_cache_entries(args.fhir_url, tasks)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(cache, f)
    print(f"Cache saved to {output_path} ({output_path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
