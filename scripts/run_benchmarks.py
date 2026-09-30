#!/usr/bin/env python3
"""Unified CLI for fixed suites, arbitrary model IDs and HTTP recovery."""
from _repo_bootstrap import bootstrap_repo_src

ROOT = bootstrap_repo_src().parent

from embodied_harness.campaign import main

if __name__ == "__main__":
    raise SystemExit(main(ROOT))
