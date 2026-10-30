"""
run_had.py
==========
HAD Framework — end-to-end test runner and demonstration.

Test suite
----------
The patterns are organised into five categories:

  SAFE         — should produce NO vulnerability (false-positive check)
  L0_VULN      — Layer-0 structural IDA (O(n²))
  L1_VULN      — Layer-1 backreference ReDoS (O(n²))
  L2_VULN      — Layer-2 compound CIDA (O(n³))
  AMBIGUOUS    — edge-cases / regression tests

Run
---
  python run_had.py              → summary table
  python run_had.py --verbose    → full report per pattern + repair suggestions
"""

from __future__ import annotations

import sys
import textwrap
from dataclasses import dataclass
from typing import List, Optional

from detection.had_detector import HADDetector, HADReport
from repair.repair_engine import RepairEngine, RepairResult


# ─────────────────────────────────────────────────────────────────────────────
# Test cases
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TestCase:
    pattern: str
    label: str
    expected_vulnerable: bool
    expected_layer: Optional[int] = None          # None means "don't check"
    expected_complexity: Optional[str] = None     # None means "don't check"
    description: str = ""


TEST_CASES: List[TestCase] = [
    # ── SAFE (no vulnerability expected) ─────────────────────────────────────
    TestCase(
        pattern=r"a+b",
        label="SAFE",
        expected_vulnerable=False,
        description="Simple concat — a-loop and b-literal share no backtracking path",
    ),
    TestCase(
        pattern=r"abc",
        label="SAFE",
        expected_vulnerable=False,
        description="No quantifiers at all",
    ),
    TestCase(
        pattern=r"[a-z]+\d+",
        label="SAFE",
        expected_vulnerable=False,
        description="Two non-overlapping loops — char classes disjoint",
    ),
    TestCase(
        pattern=r"(?:abc)+def",
        label="SAFE",
        expected_vulnerable=False,
        description="Non-capturing group + literal suffix anchors the match",
    ),
    TestCase(
        pattern=r"a{3}b{2}",
        label="SAFE",
        expected_vulnerable=False,
        description="Fixed-count repetitions — no ambiguity",
    ),

    # ── LAYER-0 vulnerabilities ───────────────────────────────────────────────
    TestCase(
        pattern=r"(a+)+",
        label="L0",
        expected_vulnerable=True,
        expected_layer=0,
        expected_complexity="O(n²)",
        description="Classic nested quantification — textbook IDA",
    ),
    TestCase(
        pattern=r"(a*)*",
        label="L0",
        expected_vulnerable=True,
        expected_layer=0,
        expected_complexity="O(n²)",
        description="Star-of-star — inner * ε-matches interact with outer *",
    ),
    TestCase(
        pattern=r"(a|aa)+",
        label="L0",
        expected_vulnerable=True,
        expected_layer=0,
        expected_complexity="O(n²)",
        description="Overlapping alternation inside + — a vs aa share prefix",
    ),
    TestCase(
        pattern=r"(\w+\s+)+",
        label="L0",
        expected_vulnerable=True,
        expected_layer=0,
        expected_complexity="O(n²)",
        description="Alternating word/space loops — Weideman canonical example",
    ),
    TestCase(
        pattern=r"(a+b+)+c",
        label="L0",
        expected_vulnerable=True,
        expected_layer=0,
        expected_complexity="O(n²)",
        description="Nested quantifiers with suffix — catastrophic on 'aaaa...'",
    ),

    # ── LAYER-1 vulnerabilities ───────────────────────────────────────────────
    TestCase(
        pattern=r"(a+)\1",
        label="L1",
        expected_vulnerable=True,
        expected_layer=1,
        expected_complexity="O(n²)",
        description="Variable capture + backref — length ambiguity on \\1",
    ),
    TestCase(
        pattern=r"(a+)\1b",
        label="L1",
        expected_vulnerable=True,
        expected_layer=1,
        expected_complexity="O(n²)",
        description="Same as above with literal suffix — string 'aaaa...' triggers",
    ),
    TestCase(
        pattern=r"(\w+)\s+\1",
        label="L1",
        expected_vulnerable=True,
        expected_layer=1,
        expected_complexity="O(n²)",
        description="Word-boundary repeated-word pattern — common in NLP regexes",
    ),
    TestCase(
        pattern=r"(a+b+)\1",
        label="L1",
        expected_vulnerable=True,
        expected_layer=1,
        expected_complexity="O(n²)",
        description="Multi-char group with backref",
    ),

    # ── LAYER-2 compound CIDA ─────────────────────────────────────────────────
    TestCase(
        pattern=r"((a+)\2)+",
        label="L2",
        expected_vulnerable=True,
        expected_layer=2,
        expected_complexity="O(n³)",
        description=(
            "Outer + (L0 IDA) wraps a group containing a+ (L1 P1) and \\2 (L1 P2)."
            " Path-connected → CIDA O(n³)."
        ),
    ),
    TestCase(
        pattern=r"((\w+)\2\s*)+",
        label="L2",
        expected_vulnerable=True,
        expected_layer=2,
        expected_complexity="O(n³)",
        description="Repeated word-duplicate pattern — real-world tokeniser regex",
    ),

    # ── REGRESSION / EDGE CASES ───────────────────────────────────────────────
    TestCase(
        pattern=r"(a*)",
        label="EDGE",
        expected_vulnerable=False,
        description="Single group with star — no backref, no outer loop → safe",
    ),
    TestCase(
        pattern=r"(a*)\1",
        label="EDGE",
        expected_vulnerable=True,
        expected_layer=1,
        description="Star (not plus) in group — C1 still fires (zero-length loop)",
    ),
    TestCase(
        pattern=r"a+|b+",
        label="EDGE",
        expected_vulnerable=False,
        description="Top-level alternation — loops are in separate branches, not path-connected",
    ),
    TestCase(
        pattern=r"^(https?://)([\w\-]+\.)+[\w\-]+(/[\w\-._~:/?#[\]@!$&'()*+,;=%]*)?$",
        label="REAL",
        expected_vulnerable=True,
        description="URL validation regex — known ReDoS pattern in the wild",
    ),
    TestCase(
        pattern=r"^(\s*\w+\s*)+$",
        label="REAL",
        expected_vulnerable=True,
        expected_layer=0,
        description="Input validation — vulnerable on strings of many spaces",
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
WARN = "\033[93m?\033[0m"


def _check(report: HADReport, tc: TestCase) -> Tuple[bool, str]:
    """Return (passed, reason)."""
    if report.is_vulnerable != tc.expected_vulnerable:
        return (
            False,
            f"Expected {'VULN' if tc.expected_vulnerable else 'SAFE'}, "
            f"got {'VULN' if report.is_vulnerable else 'SAFE'}"
        )

    if tc.expected_layer is not None and report.is_vulnerable:
        found_layers = {r.layer for r in report.all_results}
        if tc.expected_layer not in found_layers:
            return (
                False,
                f"Expected layer {tc.expected_layer}, found layers {sorted(found_layers)}"
            )

    if tc.expected_complexity is not None and report.is_vulnerable:
        if report.worst_complexity != tc.expected_complexity:
            return (
                False,
                f"Expected {tc.expected_complexity}, got {report.worst_complexity}"
            )

    return (True, "")


def run_tests(verbose: bool = False) -> None:
    detector = HADDetector()
    engine = RepairEngine()

    total = len(TEST_CASES)
    passed = 0
    failed = 0

    print("\n" + "═" * 72)
    print("  HAD Framework — Test Suite")
    print("═" * 72)
    print(f"  {'#':<4} {'Label':<8} {'Pattern':<40} {'Result':<8} {'Check'}")
    print("─" * 72)

    for i, tc in enumerate(TEST_CASES, 1):
        report = detector.analyze(tc.pattern)
        ok, reason = _check(report, tc)

        complexity = report.worst_complexity if report.is_vulnerable else "—"
        result_str = f"{complexity}" if report.is_vulnerable else "safe"
        check_icon = PASS if ok else FAIL

        truncated = tc.pattern[:38] + ".." if len(tc.pattern) > 40 else tc.pattern
        print(f"  {i:<4} {tc.label:<8} {truncated:<40} {result_str:<8} {check_icon}",
              end="")
        if not ok:
            print(f"  ← {reason}", end="")
        print()

        if ok:
            passed += 1
        else:
            failed += 1

        if verbose:
            print()
            print(textwrap.indent(report.summary(), "      "))
            repairs = engine.repair(report)
            if repairs:
                print("\n      ── Repair suggestions ──")
                for r in repairs:
                    print(textwrap.indent(str(r), "        "))
            print()

    print("─" * 72)
    print(f"  Passed: {passed}/{total}   Failed: {failed}/{total}")
    print("═" * 72 + "\n")

    if failed > 0:
        sys.exit(1)


def demo_single(pattern: str, verbose: bool = True) -> None:
    """Analyse a single user-supplied regex."""
    detector = HADDetector()
    engine = RepairEngine()

    report = detector.analyze(pattern)
    print("\n" + report.summary())

    repairs = engine.repair(report)
    if repairs:
        print("\n── Repair suggestions " + "─" * 50)
        for r in repairs:
            print(f"\n{r}")
    else:
        print("\nNo repair needed — pattern appears safe.")
    print()


# Support  _check  reference from inside run_tests without forward-declaration issues
from typing import Tuple   # noqa: E402 (already imported via __future__ at top)


if __name__ == "__main__":
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    single  = next((a for a in sys.argv[1:] if not a.startswith("-")), None)

    if single:
        demo_single(single, verbose=True)
    else:
        run_tests(verbose=verbose)
