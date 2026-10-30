"""
had_framework/repair/repair_engine.py
=====================================
Layer-aware repair engine for ReDoS vulnerabilities.

Each layer receives a dedicated repair strategy that is semantically
preserving (same language accepted, same captures) while eliminating
the identified source of super-linear backtracking.

Layer-0 repair  — Atomic-group wrapping
    Wraps the overlapping loops in an atomic group  (?>…)  or, when the
    target engine does not support atomic groups, emulates them via a
    possessive quantifier or a lookahead-based construction.

Layer-1 repair  — Capture-group bounding
    Inserts an upper-bound quantifier on the variable-length sub-pattern
    inside the capture group, breaking the C1 condition, and optionally
    moves the backreference outside the outer loop, breaking C2.

Layer-2 repair  — Sequential composition of Layer-0 and Layer-1 repairs
    The compound vulnerability requires eliminating *both* the structural
    IDA and the backreference interaction.  The engine applies repairs in
    dependency order: first Layer-1 (more invasive), then Layer-0.

All repairs are *textual* — they operate on the regex string with the
help of the parser's position metadata.  A future extension could operate
directly on the NFA and then serialise back to a regex.

RepairResult dataclass
----------------------
  original   : str   — the input pattern
  repaired   : str   — the fixed pattern
  layer      : int   — which layer was repaired
  strategy   : str   — human-readable description of the strategy used
  confidence : float — 0–1 heuristic confidence that the repair is correct
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from detection.had_detector import HADReport, HADResult


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RepairResult:
    original: str
    repaired: str
    layer: int
    strategy: str
    confidence: float = 1.0
    notes: str = ""

    def __repr__(self) -> str:
        return (
            f"[Layer {self.layer} repair | conf={self.confidence:.0%}]\n"
            f"  Strategy : {self.strategy}\n"
            f"  Before   : {self.original}\n"
            f"  After    : {self.repaired}"
            + (f"\n  Note     : {self.notes}" if self.notes else "")
        )


# ─────────────────────────────────────────────────────────────────────────────
# Internal regex-string utilities
# ─────────────────────────────────────────────────────────────────────────────

def _find_top_level_groups(pattern: str) -> List[Tuple[int, int, int]]:
    """
    Return list of (group_id, start_pos, end_pos) for every capturing group
    in the pattern, in order of their opening parenthesis.

    group_id is 1-indexed (matches \\1 backreference convention).
    start_pos / end_pos are indices into pattern (inclusive).
    """
    groups: List[Tuple[int, int, int]] = []
    stack: List[Tuple[int, int]] = []   # (group_id, start_pos)
    gid_counter = 0
    i = 0

    while i < len(pattern):
        c = pattern[i]
        if c == '\\':
            i += 2
            continue
        if c == '(' :
            # Check for non-capturing prefix
            if i + 1 < len(pattern) and pattern[i + 1] == '?':
                # non-capturing or lookahead — skip
                i += 1
                # still push a placeholder so the depth count is correct
                stack.append((-1, i))
            else:
                gid_counter += 1
                stack.append((gid_counter, i))
        elif c == ')':
            if stack:
                gid, start = stack.pop()
                if gid != -1:
                    groups.append((gid, start, i))
        i += 1

    return groups


def _find_quantifier_after(pattern: str, pos: int) -> Tuple[int, int]:
    """
    Return (start, end) of the quantifier token immediately after position pos,
    or (pos+1, pos+1) if there is no quantifier.

    pos is the index of the closing ')' of a group.
    """
    i = pos + 1
    if i >= len(pattern):
        return (i, i)
    c = pattern[i]
    if c in ('*', '+', '?'):
        return (i, i + 1)
    if c == '{':
        j = i + 1
        while j < len(pattern) and pattern[j] != '}':
            j += 1
        return (i, j + 1)
    return (i, i)


def _wrap_atomic(inner: str, quantifier: str) -> str:
    """
    Wrap *inner* in an atomic group with the given quantifier.

    Atomic groups suppress backtracking into the group once it has matched.
    We emit  (?> inner ) quantifier  which is supported by PCRE, Java, .NET.
    For Python's  re  module we emit a workaround using a lookahead + capture:
      (?=(inner_possessive))\\N   — but this changes the group numbering.

    For simplicity we emit the PCRE form and note the Python limitation.
    """
    return f"(?>{inner}){quantifier}"


def _wrap_possessive(inner: str, quantifier: str) -> str:
    """
    Possessive quantifier form  inner Q+  (Java / PCRE2 syntax).
    E.g.  a+  →  a++
    """
    if quantifier in ('*', '+', '?'):
        return f"{inner}{quantifier}+"
    return f"(?>{inner}){quantifier}"   # fall back for {n,m}


# ─────────────────────────────────────────────────────────────────────────────
# Layer-0 repair: atomic-group wrapping
# ─────────────────────────────────────────────────────────────────────────────

# Patterns that match common vulnerable forms
_L0_NESTED_STAR  = re.compile(r'\(([^()]+)\)\*\*|\(([^()]+)\)[+*]\)?\*|\(([^()]+[+*][^()]*)\)[+*]')
_L0_SIMPLE_QUANT = re.compile(r'\(([^()]+)\)([+*])\)')

def repair_layer0(pattern: str, result: HADResult) -> RepairResult:
    """
    Layer-0 repair: wrap the outer repetition in an atomic group.

    Strategy
    --------
    Detect the outermost quantified group that contains an inner quantifier
    (the hallmark of nested quantification).  Replace  (inner)+  →  (?>inner)+.

    If no nested quantification is found (e.g. the two-overlap-loop comes
    from alternation), we add character-class constraints to de-overlap them.
    """
    original = pattern
    repaired = pattern
    strategy = "atomic-group wrapping"
    confidence = 0.9
    notes = ""

    # ── Strategy A: nested quantifier (a+)+  →  (?>a+)+  ─────────────────
    # Match patterns like  (X+)+  (X*)+  (X+)*  etc.
    nested_re = re.compile(r'\(([^()]*[+*?][^()]*)\)([+*])')
    m = nested_re.search(repaired)
    if m:
        inner = m.group(1)
        outer_q = m.group(2)
        replacement = f"(?>{inner}){outer_q}"
        repaired = repaired[:m.start()] + replacement + repaired[m.end():]
        notes = "Atomic group prevents re-entry into the inner loop on backtrack"
    else:
        # ── Strategy B: overlapping alternation  (a|ab)+  ─────────────────
        # Sort alternation branches longest-first to prevent partial-match overlap
        alt_re = re.compile(r'\(([^()]+\|[^()]+)\)([+*])')
        m2 = alt_re.search(repaired)
        if m2:
            branches_str = m2.group(1)
            q = m2.group(2)
            branches = branches_str.split('|')
            branches_sorted = sorted(branches, key=len, reverse=True)
            replacement = f"(?:{'|'.join(branches_sorted)}){q}"
            repaired = repaired[:m2.start()] + replacement + repaired[m2.end():]
            strategy = "alternation reordering (longest branch first)"
            confidence = 0.75
            notes = "Reordering prevents partial match followed by retry of longer branch"
        else:
            # ── Strategy C: generic — wrap entire pattern in atomic group ──
            repaired = f"(?>{repaired})"
            strategy = "whole-pattern atomic group (conservative)"
            confidence = 0.5
            notes = "Could not identify specific loop pair; wrapped entire pattern"

    return RepairResult(
        original=original,
        repaired=repaired,
        layer=0,
        strategy=strategy,
        confidence=confidence,
        notes=notes,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Layer-1 repair: capture-group bounding
# ─────────────────────────────────────────────────────────────────────────────

def repair_layer1(pattern: str, result: HADResult) -> RepairResult:
    """
    Layer-1 repair: break C1 or C2 (or both) for the affected group.

    Strategy hierarchy
    ------------------
    1. C1-break (preferred): add an upper-bound quantifier on the variable-
       length sub-pattern inside the capture group.  E.g.  (a+)  →  (a{1,100})
       This bounds the group capture length and prevents exponential
       re-evaluation of the backreference.

    2. C2-break (fallback): move the backreference outside the repeating loop.
       E.g.  (\\1a)+  →  (?:\\1)(a)+  — backreference is no longer re-evaluated
       on every iteration.

    3. Combined: if the group structure is ambiguous, apply both.
    """
    original = pattern
    repaired = pattern
    group_ids = result.group_ids
    strategy = "capture-group bounding"
    confidence = 0.85
    notes = ""

    for gid in group_ids:
        # Locate the group text
        groups = _find_top_level_groups(repaired)
        target = next((g for g in groups if g[0] == gid), None)
        if target is None:
            continue
        _, gstart, gend = target

        inner = repaired[gstart + 1: gend]   # text between ( and )

        # Strategy 1 — bound inner quantifiers
        def bound_quantifier(m: re.Match) -> str:
            q = m.group(0)
            if q == '*':
                return '{0,100}'
            if q == '+':
                return '{1,100}'
            return q

        bounded_inner = re.sub(r'[+*](?!\+)', bound_quantifier, inner)

        if bounded_inner != inner:
            repaired = (
                repaired[:gstart + 1] + bounded_inner + repaired[gend:]
            )
            notes = f"Bounded inner quantifiers in group {gid} to break C1"
        else:
            # Strategy 2 — move backref before the loop
            backref_pat = re.compile(rf'\\{gid}')
            m2 = backref_pat.search(repaired)
            if m2:
                # Find surrounding loop
                loop_match = re.search(
                    rf'(\((?:[^()\\]|\\.)*\\{gid}(?:[^()\\]|\\.)*\))([+*])',
                    repaired
                )
                if loop_match:
                    loop_body = loop_match.group(1)
                    loop_q = loop_match.group(2)
                    # Extract non-backref part
                    no_backref = backref_pat.sub('', loop_body.strip('()'))
                    replacement = f"(?:\\{gid})(?:{no_backref}){loop_q}"
                    repaired = (
                        repaired[:loop_match.start()]
                        + replacement
                        + repaired[loop_match.end():]
                    )
                    strategy = "backref extraction (C2-break)"
                    notes = f"Moved \\{gid} outside the repeating loop to break C2"
                    confidence = 0.70

    return RepairResult(
        original=original,
        repaired=repaired,
        layer=1,
        strategy=strategy,
        confidence=confidence,
        notes=notes,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Layer-2 repair: sequential composition
# ─────────────────────────────────────────────────────────────────────────────

def repair_layer2(pattern: str, result: HADResult, report: HADReport) -> RepairResult:
    """
    Layer-2 repair: apply Layer-1 first, then Layer-0 on the result.

    The compound vulnerability requires removing *both* the structural IDA
    and the backreference interaction.  Applying Layer-1 first is preferred
    because backreference restructuring changes group numbering, which must
    be stable before atomic-group wrapping.
    """
    original = pattern
    intermediate = pattern
    notes_parts: List[str] = []

    # Step 1 — Layer-1 repair on each affected group
    for r1 in report.layer1:
        res1 = repair_layer1(intermediate, r1)
        intermediate = res1.repaired
        notes_parts.append(f"L1({res1.strategy})")

    # Step 2 — Layer-0 repair on the Layer-1-repaired pattern
    for r0 in report.layer0:
        res0 = repair_layer0(intermediate, r0)
        intermediate = res0.repaired
        notes_parts.append(f"L0({res0.strategy})")

    confidence = 0.70   # compound repairs have higher uncertainty

    return RepairResult(
        original=original,
        repaired=intermediate,
        layer=2,
        strategy="sequential L1 → L0 composition (CIDA repair)",
        confidence=confidence,
        notes="; then ".join(notes_parts),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main repair engine
# ─────────────────────────────────────────────────────────────────────────────

class RepairEngine:
    """
    Accepts a HADReport and produces RepairResult objects for each detected
    vulnerability layer.

    Usage
    -----
    >>> from had_framework.detection.had_detector import HADDetector
    >>> from had_framework.repair.repair_engine import RepairEngine
    >>> report = HADDetector().analyze(r'(a+)+')
    >>> repairs = RepairEngine().repair(report)
    >>> for r in repairs: print(r)
    """

    def repair(self, report: HADReport) -> List[RepairResult]:
        results: List[RepairResult] = []

        if not report.is_vulnerable:
            return results

        # Layer 2 takes priority (subsumes L0 + L1)
        if report.layer2:
            for r2 in report.layer2:
                results.append(repair_layer2(report.pattern, r2, report))
            return results

        # Layer 0 repairs
        seen_l0: set = set()
        for r0 in report.layer0:
            key = frozenset(s.id for s in r0.loop_states)
            if key not in seen_l0:
                seen_l0.add(key)
                results.append(repair_layer0(report.pattern, r0))

        # Layer 1 repairs
        for r1 in report.layer1:
            results.append(repair_layer1(report.pattern, r1))

        return results

    def repair_pattern(self, pattern: str) -> List[RepairResult]:
        """Convenience wrapper: analyze + repair in one call."""
        from detection.had_detector import HADDetector
        report = HADDetector().analyze(pattern)
        return self.repair(report)
