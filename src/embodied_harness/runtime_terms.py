"""Local, secret-free authorization receipt for license-gated runtimes.

The interactive command lives in ``scripts/confirm_runtime_terms.py``.  This
module contains the shared plan and verifier so native workers and the command
cannot silently disagree about what was accepted.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import tempfile
from typing import Any, Iterable, Mapping

from .paths import EXTERNAL_ROOT_ENV, resolve_project_root


REPOSITORY_ROOT = resolve_project_root()
DEFAULT_RECEIPT = (
    REPOSITORY_ROOT
    / ".arena"
    / "local-authorization"
    / "runtime-terms-v1.json"
)
SCHEMA_VERSION = "agentic-embodied-arena/runtime-terms-acceptance/v1"
SUPPORTED_SCOPES = ("behavior1k", "robodojo")

NVIDIA_EULA_URL = (
    "https://www.nvidia.com/en-us/agreements/enterprise-software/"
    "nvidia-software-license-agreement/"
)
BEHAVIOR_INSTALL_URL = "https://behavior.stanford.edu/getting_started/installation.html"
BEHAVIOR_SETUP_URL = (
    "https://github.com/StanfordVL/BEHAVIOR-1K/blob/"
    "88454bd04f75dc57c00ab1f1a00bcde1ff505950/setup.sh"
)
ROBODOJO_README_URL = (
    "https://github.com/RoboDojo-Benchmark/RoboDojo/blob/"
    "e41d848837ec5bf15e7b045672794f33d78433ce/README.md"
)
ROBODOJO_LICENSE_URL = (
    "https://github.com/RoboDojo-Benchmark/RoboDojo/blob/"
    "e41d848837ec5bf15e7b045672794f33d78433ce/LICENSE"
)

AUTHORITY_PHRASE = "I AM AUTHORIZED TO ACCEPT THESE TERMS"
NVIDIA_PHRASE = "I HAVE READ AND ACCEPT THE NVIDIA ISAAC SIM EULA"
BEHAVIOR_LICENSE_PHRASE = "I HAVE READ AND ACCEPT THE BEHAVIOR DATA BUNDLE AGREEMENT"
BEHAVIOR_USE_PHRASE = "I CONFIRM BEHAVIOR DATA USE IS NON-COMMERCIAL ACADEMIC RESEARCH"
ROBODOJO_USE_PHRASE = "I CONFIRM ROBODOJO USE IS NON-COMMERCIAL RESEARCH OR EDUCATION"
FINAL_PHRASE = "WRITE LOCAL RECEIPT"


def _external_root(explicit: Path | None = None) -> Path:
    if explicit is not None:
        return explicit.expanduser().resolve()
    configured = os.environ.get(EXTERNAL_ROOT_ENV)
    if not configured:
        return REPOSITORY_ROOT / "external"
    path = Path(configured).expanduser()
    if not path.is_absolute():
        path = REPOSITORY_ROOT / path
    return path.resolve()


def _behavior_setup_path(external_root: Path | None = None) -> Path:
    return _external_root(external_root) / "upstreams" / "behavior1k" / "setup.sh"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _source_fingerprint(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        return {"path": str(path), "present": False, "sha256": None}
    return {
        "path": str(path),
        "present": True,
        "sha256": _sha256_bytes(path.read_bytes()),
    }


def _normalized_scopes(scopes: Iterable[str]) -> tuple[str, ...]:
    requested = tuple(dict.fromkeys(scopes))
    if not requested:
        requested = SUPPORTED_SCOPES
    unknown = sorted(set(requested) - set(SUPPORTED_SCOPES))
    if unknown:
        raise ValueError(f"unsupported scope(s): {', '.join(unknown)}")
    return tuple(scope for scope in SUPPORTED_SCOPES if scope in requested)


def acceptance_plan(
    scopes: Iterable[str], *, external_root: Path | None = None
) -> dict[str, Any]:
    selected = _normalized_scopes(scopes)
    agreements: list[dict[str, Any]] = [
        {
            "id": "nvidia_isaac_sim_eula",
            "url": NVIDIA_EULA_URL,
            "required_by": list(selected),
            "confirmation_phrase": NVIDIA_PHRASE,
        }
    ]
    if "behavior1k" in selected:
        agreements.append(
            {
                "id": "behavior_data_bundle_agreement",
                "url": BEHAVIOR_INSTALL_URL,
                "pinned_source_url": BEHAVIOR_SETUP_URL,
                "required_by": ["behavior1k"],
                "restrictions_relied_on": [
                    "non-commercial academic research only",
                    "use the data only within OmniGibson",
                    "do not redistribute the key or the data",
                    "do not reverse engineer the encrypted data",
                ],
                "confirmation_phrases": [
                    BEHAVIOR_LICENSE_PHRASE,
                    BEHAVIOR_USE_PHRASE,
                ],
                "local_source": _source_fingerprint(
                    _behavior_setup_path(external_root)
                ),
            }
        )
    if "robodojo" in selected:
        agreements.append(
            {
                "id": "robodojo_noncommercial_usage_guard",
                "readme_url": ROBODOJO_README_URL,
                "license_url": ROBODOJO_LICENSE_URL,
                "required_by": ["robodojo"],
                "note": (
                    "The pinned README describes non-commercial research use, while "
                    "the linked LICENSE file contains MIT text. This repository uses "
                    "the more conservative non-commercial gate unless the maintainers "
                    "clarify commercial use in writing."
                ),
                "confirmation_phrase": ROBODOJO_USE_PHRASE,
            }
        )
    return {"scopes": list(selected), "agreements": agreements}


def _build_receipt(plan: Mapping[str, Any]) -> dict[str, Any]:
    accepted_at = datetime.now(timezone.utc).isoformat()
    agreement_snapshot = json.dumps(
        plan["agreements"], sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return {
        "schema_version": SCHEMA_VERSION,
        "accepted_at": accepted_at,
        "scope": list(plan["scopes"]),
        "human_interactive_confirmation": True,
        "authorized_representative_confirmed": True,
        "installation_authorized": True,
        "host": socket.gethostname(),
        "agreements": plan["agreements"],
        "agreement_snapshot_sha256": _sha256_bytes(agreement_snapshot),
        "contains_secrets": False,
    }


def _absolute_without_following_leaf(path: Path) -> Path:
    expanded = path.expanduser()
    if not expanded.is_absolute():
        expanded = Path.cwd() / expanded
    return expanded.absolute()


def _write_receipt(path: Path, receipt: Mapping[str, Any], *, replace: bool) -> str:
    path = _absolute_without_following_leaf(path)
    if path.is_symlink():
        raise RuntimeError(f"refusing symlink receipt target: {path}")
    if path.exists() and not replace:
        raise RuntimeError(f"receipt already exists: {path}; pass --replace to reconfirm")
    if path.exists() and not path.is_file():
        raise RuntimeError(f"refusing to replace non-regular receipt target: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode("utf-8")
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        os.fchmod(file_descriptor, 0o600)
        with os.fdopen(file_descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
        path.chmod(0o600)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return _sha256_bytes(payload)


def verify_receipt(
    path: Path,
    required_scopes: Iterable[str] = (),
    *,
    external_root: Path | None = None,
) -> dict[str, Any]:
    path = _absolute_without_following_leaf(path)
    if path.is_symlink():
        return {"ok": False, "path": str(path), "errors": ["receipt_symlink"]}
    errors: list[str] = []
    try:
        payload = path.read_bytes()
        data = json.loads(payload)
    except FileNotFoundError:
        return {"ok": False, "path": str(path), "errors": ["receipt_missing"]}
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "ok": False,
            "path": str(path),
            "errors": [f"receipt_unreadable:{type(exc).__name__}"],
        }

    if not isinstance(data, dict):
        return {"ok": False, "path": str(path), "errors": ["receipt_not_object"]}
    if data.get("schema_version") != SCHEMA_VERSION:
        errors.append("schema_version_mismatch")
    for field in (
        "human_interactive_confirmation",
        "authorized_representative_confirmed",
        "installation_authorized",
    ):
        if data.get(field) is not True:
            errors.append(f"{field}_missing")
    if data.get("contains_secrets") is not False:
        errors.append("contains_secrets_marker_invalid")

    scopes = data.get("scope")
    if not isinstance(scopes, list) or any(
        scope not in SUPPORTED_SCOPES for scope in scopes
    ):
        errors.append("scope_invalid")
        scopes = []
    requested = tuple(required_scopes)
    required = _normalized_scopes(requested) if requested else ()
    for scope in required:
        if scope not in scopes:
            errors.append(f"scope_not_accepted:{scope}")

    agreements = data.get("agreements")
    if not isinstance(agreements, list):
        errors.append("agreements_missing")
        agreements = []
    expected_snapshot = _sha256_bytes(
        json.dumps(
            agreements, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
    )
    if data.get("agreement_snapshot_sha256") != expected_snapshot:
        errors.append("agreement_snapshot_digest_mismatch")

    if scopes:
        current_plan = acceptance_plan(scopes, external_root=external_root)
        # A restored receipt retains the original acceptance and source digest.
        # Relocating the same source file does not change the accepted terms.
        # Keep every other field (including presence and digest) in the check.
        recorded_sources = {
            item.get("id"): item.get("local_source")
            for item in agreements if isinstance(item, dict)
        }
        for item in current_plan["agreements"]:
            current_source = item.get("local_source")
            recorded_source = recorded_sources.get(item.get("id"))
            if isinstance(current_source, dict) and isinstance(recorded_source, dict):
                current_source["path"] = recorded_source.get("path")
        current_snapshot = _sha256_bytes(
            json.dumps(
                current_plan["agreements"],
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("utf-8")
        )
        if expected_snapshot != current_snapshot:
            errors.append("agreement_snapshot_not_current")

    agreement_ids = {
        item.get("id") for item in agreements if isinstance(item, dict)
    }
    if scopes and "nvidia_isaac_sim_eula" not in agreement_ids:
        errors.append("nvidia_agreement_missing")
    if "behavior1k" in scopes and "behavior_data_bundle_agreement" not in agreement_ids:
        errors.append("behavior_agreement_missing")
    if "robodojo" in scopes and "robodojo_noncommercial_usage_guard" not in agreement_ids:
        errors.append("robodojo_usage_guard_missing")

    return {
        "ok": not errors,
        "path": str(path),
        "scope": scopes,
        "accepted_at": data.get("accepted_at"),
        "receipt_sha256": _sha256_bytes(payload),
        "errors": errors,
    }
