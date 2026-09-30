#!/usr/bin/env python3
"""Dispatch a mixed-suite task to its existing native or Pukun runner."""
from pathlib import Path
import os
import sys


def _option(arguments, name):
    for i, arg in enumerate(arguments):
        if arg == name and i + 1 < len(arguments):
            return arguments[i + 1]
        if arg.startswith(name + "="):
            return arg.split("=", 1)[1]
    return None


def runner_name(arguments):
    # W1/W5 share --sample-id with W2, so route by benchmark first.
    benchmark = _option(arguments, "--benchmark")
    if benchmark in {"multispa", "influx", "mapfree", "tracespatial_2d", "tracespatial_3d"} or (benchmark or "").startswith("vlm_"):
        return "run_w1_native_case.py"
    if _option(arguments, "--benchmark") in {"reasonaff", "ragnet", "umd"}:
        return "run_w5_native_case.py"
    if any(a == "--sample-id" or a.startswith("--sample-id=") for a in arguments):
        return "run_w2_native_case.py"
    pukun = any(a == "--entry-id" or a.startswith("--entry-id=") for a in arguments)
    return "run_pukun_native_case.py" if pukun else "run_native_universal_case.py"


def bounded_arguments(arguments, environment):
    """Apply the campaign API request ceiling without widening case limits.

    Pukun already takes min(request ceiling, phase timeout) in its author.
    Native cases have explicit request flags, which otherwise override dotenv.
    """
    result = list(arguments)
    def value(option):
        found = None
        for i, arg in enumerate(result):
            if arg == option and i + 1 < len(result):
                found = result[i + 1]
            elif arg.startswith(option + "="):
                found = arg.split("=", 1)[1]
        return found
    if runner_name(result) != "run_native_universal_case.py" or value("--model-provider") != "openai-compatible":
        return result
    ceiling = environment.get("LLM_REQUEST_TIMEOUT_SECONDS")
    if ceiling:
        option = "--request-timeout-seconds"
        case_limit = value(option)
        limit = min(float(ceiling), float(case_limit)) if case_limit else float(ceiling)
        # argparse uses the final occurrence, so the original frozen flags stay
        # visible in the parent campaign's command/provenance.
        result += [option, str(limit)]
    return result


if __name__ == "__main__":
    arguments = bounded_arguments(sys.argv[1:], os.environ)
    runner = Path(__file__).resolve().parent / runner_name(arguments)
    if os.environ.get("ARENA_CASE_STUDY_HZ") != "1":
        os.execv(sys.executable, [sys.executable, str(runner), *arguments])
    import runpy
    sys.path.insert(0, str(runner.parents[1] / "src"))
    from embodied_harness.case_study_recording import start_for_case
    recorder = start_for_case(arguments)
    sys.argv = [str(runner), *arguments]
    try:
        runpy.run_path(str(runner), run_name="__main__")
    finally:
        if recorder is not None:
            recorder.close()
