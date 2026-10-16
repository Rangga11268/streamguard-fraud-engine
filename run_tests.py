"""StreamGuard Master Multi-Tier Test Suite Runner.

Single-command multi-tier test execution harness supporting:
  - Standard library unittest (zero external dependency / zero-install on Windows)
  - Pytest (when present or explicitly requested)
  - Tier selection: --tier 1, 2, 3, 4, 5, all
  - Adversarial challenge suites: --adversarial
  - Machine-readable JSON output telemetry: --json-output <path>
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
import unittest
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

# Avoid joblib core discovery issues on Windows
os.environ["LOKY_MAX_CPU_COUNT"] = "4"


TIER_MODULE_MAP: Dict[str, str] = {
    "1": "tests.test_tier1_units",
    "2": "tests.test_tier2_boundaries",
    "3": "tests.test_tier3_statistical",
    "4": "tests.test_tier4_benchmarks",
    "5": "tests.test_tier5_api_endpoints",
}

ADVERSARIAL_MODULE_MAP: Dict[str, str] = {
    "Adv-M1": "tests.test_adversarial_m1",
    "Adv-M6": "tests.test_adversarial_m6",
}

TIER_NAME_MAP: Dict[str, str] = {
    "1": "Tier 1: Functional Unit Tests",
    "2": "Tier 2: Boundary & Edge Cases",
    "3": "Tier 3: Statistical Verification",
    "4": "Tier 4: Latency & Benchmarks",
    "5": "Tier 5: API Serving Endpoints",
    "Adv-M1": "Adversarial: Feature Store & Windows",
    "Adv-M6": "Adversarial: FastAPI Serving Layer",
}


class TierResultTracker:
    """Tracks test outcomes per tier for telemetry and reporting."""

    def __init__(self, tier_id: str, module_name: str) -> None:
        self.tier_id = tier_id
        self.module_name = module_name
        self.total: int = 0
        self.passed: int = 0
        self.failed: int = 0
        self.errors: int = 0
        self.skipped: int = 0
        self.duration_sec: float = 0.0
        self.test_details: List[Dict[str, Any]] = []

    @property
    def is_success(self) -> bool:
        return self.failed == 0 and self.errors == 0


def run_tier_with_unittest(tier_id: str, module_name: str, verbose: bool = False) -> TierResultTracker:
    """Run an individual test tier using standard library unittest."""
    tracker = TierResultTracker(tier_id, module_name)
    loader = unittest.TestLoader()

    try:
        mod = importlib.import_module(module_name)
    except ImportError as e:
        tracker.errors += 1
        tracker.test_details.append({
            "test_name": f"{module_name}.__import__",
            "status": "ERROR",
            "message": str(e),
        })
        return tracker

    suite = loader.loadTestsFromModule(mod)
    tracker.total = suite.countTestCases()

    # Custom result recorder
    class CustomResult(unittest.TestResult):
        def __init__(self) -> None:
            super().__init__()
            self._start_time: float = 0.0

        def startTest(self, test: unittest.TestCase) -> None:
            super().startTest(test)
            self._start_time = time.perf_counter()

        def addSuccess(self, test: unittest.TestCase) -> None:
            super().addSuccess(test)
            dt = time.perf_counter() - self._start_time
            tracker.passed += 1
            tracker.test_details.append({
                "test_name": test.id(),
                "status": "PASS",
                "duration_sec": round(dt, 4),
            })

        def addFailure(self, test: unittest.TestCase, err: Any) -> None:
            super().addFailure(test, err)
            dt = time.perf_counter() - self._start_time
            tracker.failed += 1
            msg = self._exc_info_to_string(err, test)
            tracker.test_details.append({
                "test_name": test.id(),
                "status": "FAIL",
                "duration_sec": round(dt, 4),
                "error": msg,
            })

        def addError(self, test: unittest.TestCase, err: Any) -> None:
            super().addError(test, err)
            dt = time.perf_counter() - self._start_time
            tracker.errors += 1
            msg = self._exc_info_to_string(err, test)
            tracker.test_details.append({
                "test_name": test.id(),
                "status": "ERROR",
                "duration_sec": round(dt, 4),
                "error": msg,
            })

        def addSkip(self, test: unittest.TestCase, reason: str) -> None:
            super().addSkip(test, reason)
            tracker.skipped += 1
            tracker.test_details.append({
                "test_name": test.id(),
                "status": "SKIP",
                "reason": reason,
            })

    result = CustomResult()
    t0 = time.perf_counter()
    suite.run(result)
    t1 = time.perf_counter()
    tracker.duration_sec = round(t1 - t0, 3)

    return tracker


def run_with_pytest(tier_selection: str, json_path: Optional[str] = None, verbose: bool = False, adversarial: bool = False) -> int:
    """Run tests using pytest if requested."""
    try:
        import pytest
    except ImportError:
        print("[!] Pytest is not installed in the environment. Falling back to unittest runner.")
        return -1

    args = ["-q"]
    if verbose:
        args = ["-v"]

    if adversarial or tier_selection == "adversarial":
        args.extend(["tests/test_adversarial_m1.py", "tests/test_adversarial_m6.py"])
    elif tier_selection == "all":
        args.append("tests")
    else:
        mod_name = TIER_MODULE_MAP.get(tier_selection)
        if mod_name:
            args.append(f"{mod_name.replace('.', '/')}.py")
        else:
            args.append("tests")

    return pytest.main(args)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="StreamGuard Multi-Tier Test Suite Runner (Windows Python 3.12 Native)"
    )
    parser.add_argument(
        "--tier",
        type=str,
        default="all",
        choices=["1", "2", "3", "4", "5", "all", "adversarial"],
        help="Test tier to execute (1, 2, 3, 4, 5, all, or adversarial)",
    )
    parser.add_argument(
        "--adversarial",
        action="store_true",
        help="Execute adversarial stress challenge test suites (tests.test_adversarial_m1 and tests.test_adversarial_m6)",
    )
    parser.add_argument(
        "--runner",
        type=str,
        default="unittest",
        choices=["unittest", "pytest", "auto"],
        help="Test runner backend (unittest, pytest, auto)",
    )
    parser.add_argument(
        "--json-output",
        type=str,
        default=None,
        help="Path to export structured JSON test results",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose test output",
    )
    args = parser.parse_args()

    is_adversarial = args.adversarial or args.tier == "adversarial"

    # Determine runner
    if args.runner == "pytest":
        ret = run_with_pytest(args.tier, args.json_output, args.verbose, adversarial=is_adversarial)
        if ret != -1:
            return ret

    # Standard library unittest runner
    if is_adversarial:
        banner_title = "STREAMGUARD ADVERSARIAL STRESS TEST SUITE"
        run_modules: List[Tuple[str, str]] = [
            ("Adv-M1", "tests.test_adversarial_m1"),
            ("Adv-M6", "tests.test_adversarial_m6"),
        ]
    else:
        banner_title = "STREAMGUARD AUTOMATED MULTI-TIER TEST SUITE"
        selected_tiers = list(TIER_MODULE_MAP.keys()) if args.tier == "all" else [args.tier]
        run_modules = [(f"Tier {tid}", TIER_MODULE_MAP[tid]) for tid in selected_tiers]

    print("=" * 82)
    print(f"{banner_title:^82}")
    print(f"       Native Windows Python {sys.version.split()[0]} | Zero External Dependency")
    print("=" * 82)
    print(f"{'Tier':<8} | {'Module':<32} | {'Tests':<6} | {'Pass':<5} | {'Fail':<5} | {'Status'}")
    print("-" * 82)

    tier_trackers: List[TierResultTracker] = []
    total_tests = 0
    total_passed = 0
    total_failed = 0
    total_errors = 0
    total_skipped = 0
    suite_start = time.perf_counter()

    for tid, mod_name in run_modules:
        tracker = run_tier_with_unittest(tid, mod_name, verbose=args.verbose)
        tier_trackers.append(tracker)

        total_tests += tracker.total
        total_passed += tracker.passed
        total_failed += tracker.failed
        total_errors += tracker.errors
        total_skipped += tracker.skipped

        status_str = "[PASS]" if tracker.is_success else "[FAIL]"
        print(f"{tid:<8} | {mod_name:<32} | {tracker.total:<6} | {tracker.passed:<5} | {tracker.failed:<5} | {status_str}")

        if args.verbose and not tracker.is_success:
            for detail in tracker.test_details:
                if detail["status"] in ("FAIL", "ERROR"):
                    print(f"  [!] {detail['test_name']}: {detail.get('error', detail.get('message', ''))}")

    suite_duration = round(time.perf_counter() - suite_start, 3)
    all_passed = (total_failed == 0 and total_errors == 0 and total_tests > 0)
    overall_status = "PASS" if all_passed else "FAIL"

    print("=" * 82)
    print(f"Total Tests: {total_tests} | Passed: {total_passed} | Failed: {total_failed} | Errors: {total_errors} | Skipped: {total_skipped}")
    print(f"Duration: {suite_duration}s | Overall Status: [{overall_status}]")
    print("=" * 82)

    # Export JSON telemetry if requested
    if args.json_output:
        telemetry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "python_version": sys.version.split()[0],
            "platform": sys.platform,
            "runner": "unittest",
            "tier_selection": "adversarial" if is_adversarial else args.tier,
            "total_tests": total_tests,
            "passed": total_passed,
            "failed": total_failed,
            "errors": total_errors,
            "skipped": total_skipped,
            "duration_sec": suite_duration,
            "overall_status": overall_status,
            "tiers": [
                {
                    "tier_id": t.tier_id,
                    "module": t.module_name,
                    "total": t.total,
                    "passed": t.passed,
                    "failed": t.failed,
                    "errors": t.errors,
                    "duration_sec": t.duration_sec,
                    "status": "PASS" if t.is_success else "FAIL",
                    "details": t.test_details,
                }
                for t in tier_trackers
            ],
        }
        try:
            with open(args.json_output, "w", encoding="utf-8") as f:
                json.dump(telemetry, f, indent=2)
            print(f"[+] Structured test telemetry exported to: {args.json_output}")
        except Exception as e:
            print(f"[!] Warning: Failed to write JSON output: {e}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
