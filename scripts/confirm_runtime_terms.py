#!/usr/bin/env python3
"""Record human confirmation for license-gated local benchmark runtimes.

This command deliberately cannot accept terms non-interactively.  It displays
the official sources and the restrictions relied on by this repository, asks
the person running it to type explicit confirmations, and writes a local
receipt containing no license key or other secret.

The receipt is an installation safety gate, not legal advice or a substitute
for the underlying agreements.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

try:
    from _repo_bootstrap import bootstrap_repo_src
except ModuleNotFoundError:  # pragma: no cover - importlib-based tests
    from scripts._repo_bootstrap import bootstrap_repo_src

bootstrap_repo_src()

from embodied_harness.runtime_terms import (
    AUTHORITY_PHRASE,
    BEHAVIOR_INSTALL_URL,
    BEHAVIOR_LICENSE_PHRASE,
    BEHAVIOR_SETUP_URL,
    BEHAVIOR_USE_PHRASE,
    DEFAULT_RECEIPT,
    FINAL_PHRASE,
    NVIDIA_EULA_URL,
    NVIDIA_PHRASE,
    ROBODOJO_LICENSE_URL,
    ROBODOJO_README_URL,
    ROBODOJO_USE_PHRASE,
    SUPPORTED_SCOPES,
    _build_receipt,
    _write_receipt,
    acceptance_plan,
    verify_receipt,
)


def _print_plan(plan: Mapping[str, Any]) -> None:
    print("\nLicense-gated runtime confirmation")
    print("===================================")
    print("This program does not download or install anything.")
    print("It does not create, print, or store an OmniGibson key.")
    print("The resulting receipt is only a local safety gate; the official terms control.\n")

    print("NVIDIA Isaac Sim EULA (required for every selected scope):")
    print(f"  {NVIDIA_EULA_URL}\n")

    if "behavior1k" in plan["scopes"]:
        print("BEHAVIOR Data Bundle agreement:")
        print(f"  Installation guide: {BEHAVIOR_INSTALL_URL}")
        print(f"  Pinned agreement source: {BEHAVIOR_SETUP_URL}")
        print("  Repository safety assumptions:")
        print("    - non-commercial academic research only")
        print("    - data is used only within OmniGibson")
        print("    - no key or data redistribution")
        print("    - no reverse engineering of encrypted data\n")

    if "robodojo" in plan["scopes"]:
        print("RoboDojo usage notice:")
        print(f"  Pinned README: {ROBODOJO_README_URL}")
        print(f"  Pinned LICENSE file: {ROBODOJO_LICENSE_URL}")
        print("  The README says non-commercial research/education/evaluation, while")
        print("  the LICENSE file contains MIT text. We therefore require a conservative")
        print("  non-commercial-use confirmation for this installation.\n")


def _require_phrase(prompt: str, phrase: str) -> None:
    print(prompt)
    print(f"Type exactly: {phrase}")
    entered = input("> ").strip()
    if entered != phrase:
        raise RuntimeError("confirmation text did not match; no receipt was written")
    print()


def _interactive_accept(plan: Mapping[str, Any]) -> None:
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        raise RuntimeError(
            "acceptance requires an interactive terminal; piping answers or CI acceptance is disabled"
        )
    _require_phrase(
        "Confirm that you are authorized to accept terms for this user or organization.",
        AUTHORITY_PHRASE,
    )
    _require_phrase("Confirm the NVIDIA agreement.", NVIDIA_PHRASE)
    if "behavior1k" in plan["scopes"]:
        _require_phrase("Confirm the BEHAVIOR Data Bundle agreement.", BEHAVIOR_LICENSE_PHRASE)
        _require_phrase("Confirm the permitted BEHAVIOR data use.", BEHAVIOR_USE_PHRASE)
    if "robodojo" in plan["scopes"]:
        _require_phrase("Confirm the conservative RoboDojo usage condition.", ROBODOJO_USE_PHRASE)
    _require_phrase("Confirm creation of the local, secret-free receipt.", FINAL_PHRASE)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scope",
        action="append",
        choices=SUPPORTED_SCOPES,
        default=[],
        help="runtime to authorize; repeat as needed (default: both)",
    )
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument(
        "--show-only",
        action="store_true",
        help="show sources and restrictions without prompting or writing",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="verify an existing receipt instead of prompting",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="replace an existing receipt after completing every confirmation again",
    )
    args = parser.parse_args(argv)

    try:
        plan = acceptance_plan(args.scope)
        if args.verify:
            result = verify_receipt(args.receipt, required_scopes=plan["scopes"])
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0 if result["ok"] else 1

        _print_plan(plan)
        if args.show_only:
            return 0
        _interactive_accept(plan)
        receipt = _build_receipt(plan)
        digest = _write_receipt(args.receipt, receipt, replace=args.replace)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print("Confirmation recorded locally.")
    print(f"receipt: {args.receipt.expanduser().resolve()}")
    print(f"receipt_sha256: {digest}")
    print(f"scope: {', '.join(plan['scopes'])}")
    print("Do not send an OmniGibson key in chat; this receipt contains no key.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
