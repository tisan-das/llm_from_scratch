"""
run_all.py — Parallel test runner for the project's test suite.

Discovers all test_*.py files in the tests/ directory, runs each in a
separate subprocess (ThreadPoolExecutor), collects results, and prints
a unified summary.

Each suite is isolated — failures in one don't block others from running.
Exit code is 0 only when every suite passes.

Usage:
    python tests/run_all.py              (from project root)
    python -m tests.run_all              (from parent directory)

Output:
    test_attention     ...  102 passed, 0 failed  (3.2s)
    test_embeddings    ...   17 passed, 0 failed  (3.1s)
    test_layers        ...  225 passed, 0 failed  (3.5s)
    test_transformer   ...  394 passed, 0 failed  (3.8s)
    ----------------------------------------------------------
    TOTAL                       738 passed, 0 failed  (4.1s wall)
"""

from __future__ import annotations

import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover_suites() -> list[Path]:
    """Find all test_*.py files in the tests/ directory, sorted by name."""
    tests_dir = Path(__file__).parent
    suites = sorted(tests_dir.glob("test_*.py"))
    if not suites:
        print("No test_*.py files found in", tests_dir)
        sys.exit(1)
    return suites


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_suite(path: Path) -> tuple[str, int, str, float]:
    """
    Run a single test suite in a subprocess.

    Returns:
        (suite_name, exit_code, output_summary, elapsed_seconds)
    """
    t0 = time.perf_counter()
    result = subprocess.run(
        [sys.executable, str(path)],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent,  # project root
    )
    elapsed = time.perf_counter() - t0

    # Extract the summary line ("N passed, M failed") from stdout
    summary = "0 passed, 0 failed"
    for line in result.stdout.splitlines():
        if "passed" in line and "failed" in line:
            summary = line.strip()
            break

    # If the subprocess crashed without printing a summary, surface stderr
    if result.returncode != 0 and summary == "0 passed, 0 failed":
        if result.stderr.strip():
            summary = f"CRASH: {result.stderr.strip().splitlines()[-1]}"

    return (path.stem, result.returncode, summary, elapsed)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    suites = discover_suites()

    print(f"Running {len(suites)} test suites in parallel ...\n")
    t_start = time.perf_counter()

    all_passed = True
    total_passed = 0
    total_failed = 0

    # Run all suites in parallel (one subprocess per suite).
    # ThreadPoolExecutor is correct here: the actual work happens in
    # subprocess.Popen, releasing the GIL, so threads ≠ bottleneck.
    with ThreadPoolExecutor(max_workers=len(suites)) as pool:
        futures = {pool.submit(run_suite, p): p for p in suites}
        for future in futures:
            name, exit_code, summary, elapsed = future.result()
            status = "PASS" if exit_code == 0 else "FAIL"

            # Parse "N passed, M failed" from the summary
            parts = summary.replace(",", "").split()
            try:
                p_idx = parts.index("passed")
                f_idx = parts.index("failed")
                n_pass = int(parts[p_idx - 1])
                n_fail = int(parts[f_idx - 1])
            except (ValueError, IndexError):
                n_pass, n_fail = 0, 0

            total_passed += n_pass
            total_failed += n_fail
            if exit_code != 0:
                all_passed = False

            # Print per-suite line
            print(f"  {name:<22} {n_pass:>4} passed, {n_fail:>3} failed"
                  f"  ({elapsed:.1f}s)  [{status}]")

    t_total = time.perf_counter() - t_start
    print("-" * 58)
    print(f"  {'TOTAL':<22} {total_passed:>4} passed, {total_failed:>3} failed"
          f"  ({t_total:.1f}s wall)")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
