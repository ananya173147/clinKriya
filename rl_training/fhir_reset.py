"""
FHIR server state management for RL training.

Each RL episode's POST actions modify the FHIR database (creating
MedicationRequests, ServiceRequests, etc.).  Before starting a new episode
we must clean up those changes so the environment is reproducible.

Strategy:
  - Delete all FHIR resources whose meta.lastUpdated > CUTOFF
    (2025-01-01 00:00:00 UTC).  These are resources created by the agent
    during training — the baseline data all predates the cutoff.
  - This is fast (~seconds) and does not require Docker restarts.
"""

from __future__ import annotations

import json
import requests
from datetime import datetime
from typing import Optional

from .config import FHIRConfig


# Resource types that the agent can create via POST tools
MUTABLE_RESOURCE_TYPES = [
    "MedicationRequest",
    "ServiceRequest",
    "Observation",   # vitals_create
]


def verify_fhir_server(fhir_api_base: str) -> bool:
    """Return True if the FHIR server is reachable."""
    try:
        resp = requests.get(
            f"{fhir_api_base}metadata",
            timeout=10,
        )
        return resp.status_code == 200
    except Exception:
        return False


def cleanup_post_cutoff_resources(
    fhir_api_base: str,
    cutoff_ts: str = "2025-01-01T00:00:00+00:00",
    resource_types: Optional[list[str]] = None,
    verbose: bool = False,
) -> int:
    """
    Delete all FHIR resources created after the cutoff timestamp.

    These are resources that were created by agent POST actions during
    training or evaluation episodes.

    Args:
        fhir_api_base: FHIR server base URL (e.g. "http://localhost:8080/fhir/")
        cutoff_ts: ISO timestamp; resources with lastUpdated > this are deleted
        resource_types: Which resource types to clean (default: all mutable types)
        verbose: Print progress

    Returns:
        Total number of resources deleted
    """
    if resource_types is None:
        resource_types = MUTABLE_RESOURCE_TYPES

    cutoff_dt = datetime.fromisoformat(cutoff_ts)
    total_deleted = 0

    for rtype in resource_types:
        url = f"{fhir_api_base}{rtype}?_count=500&_format=json"
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            bundle = resp.json()
        except Exception as e:
            if verbose:
                print(f"  [cleanup] Failed to fetch {rtype}: {e}")
            continue

        entries = bundle.get("entry", [])
        to_delete = []

        for entry in entries:
            resource = entry.get("resource", {})
            last_updated = resource.get("meta", {}).get("lastUpdated")
            if last_updated:
                try:
                    if datetime.fromisoformat(last_updated) > cutoff_dt:
                        resource_id = resource.get("id")
                        if resource_id:
                            to_delete.append(resource_id)
                except ValueError:
                    pass

        # Delete each identified resource
        for rid in to_delete:
            delete_url = f"{fhir_api_base}{rtype}/{rid}"
            try:
                del_resp = requests.delete(delete_url, timeout=10)
                if del_resp.status_code in (200, 204, 410):
                    total_deleted += 1
                    if verbose:
                        print(f"  [cleanup] Deleted {rtype}/{rid}")
                else:
                    if verbose:
                        print(
                            f"  [cleanup] Failed to delete {rtype}/{rid}: "
                            f"HTTP {del_resp.status_code}"
                        )
            except Exception as e:
                if verbose:
                    print(f"  [cleanup] Error deleting {rtype}/{rid}: {e}")

    if verbose:
        print(f"  [cleanup] Total resources deleted: {total_deleted}")

    return total_deleted


def reset_fhir_for_episode(
    fhir_cfg: FHIRConfig,
    verbose: bool = False,
) -> bool:
    """
    Reset the FHIR server to the baseline state before an episode.

    Returns True if cleanup succeeded.
    """
    if not verify_fhir_server(fhir_cfg.api_base):
        print("[ERROR] FHIR server not reachable!")
        return False

    cleanup_post_cutoff_resources(
        fhir_api_base=fhir_cfg.api_base,
        cutoff_ts=fhir_cfg.cutoff_ts,
        verbose=verbose,
    )
    return True


def get_fhir_resource_count(
    fhir_api_base: str,
    resource_type: str,
    cutoff_ts: str = "2025-01-01T00:00:00+00:00",
) -> dict[str, int]:
    """
    Count resources before and after the cutoff for diagnostic purposes.

    Returns:
        {"before_cutoff": N, "after_cutoff": M, "total": N+M}
    """
    cutoff_dt = datetime.fromisoformat(cutoff_ts)
    url = f"{fhir_api_base}{resource_type}?_count=5000&_format=json"

    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        bundle = resp.json()
    except Exception:
        return {"before_cutoff": 0, "after_cutoff": 0, "total": 0}

    entries = bundle.get("entry", [])
    before = 0
    after = 0

    for entry in entries:
        resource = entry.get("resource", {})
        last_updated = resource.get("meta", {}).get("lastUpdated")
        if last_updated:
            try:
                if datetime.fromisoformat(last_updated) <= cutoff_dt:
                    before += 1
                else:
                    after += 1
            except ValueError:
                before += 1
        else:
            before += 1

    return {"before_cutoff": before, "after_cutoff": after, "total": before + after}
