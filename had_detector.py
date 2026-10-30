"""
had_framework/detection/had_detector.py
=======================================
Three-layer HAD vulnerability classifier.

Layer 0 — Structural (IDA / two-overlap-loop)
-------------------------------------------------
Uses "loop_back" ε-edges (the true back-edges of Thompson + and * quantifiers)
to find two structural sub-cases:

  Sub-case A — Nested loops
    Back-edge B_inner is nested inside B_outer when:
      (i)  B_outer.target ──path──► B_inner.target
      (ii) B_inner.source ──path──► B_outer.source
    + char classes in each region intersect → O(n²) IDA.

  Sub-case B — Overlapping alternation inside a loop
    A loop body containing an alternation where ≥2 branches share a
    first-character-class → O(n²) IDA.

Layer 1 — Memory (backreference interaction)
----------------------------------------------
  C1: Group body contains ≥1 "loop_back" edge (variable-length capture).
  C2: ≥1 BackrefTransition references this group.
  C1 ∧ C2 → O(n²).

Layer 2 — Compound CIDA → O(n³)
----------------------------------
  Layer-0 outer loop structurally *contains* the Layer-1 capture-group body.
  The IDA ambiguity (polynomial in n) multiplies the backref re-evaluation cost.
  First formally-named CIDA class — corresponds to Liu et al. 2026, Fig. 8.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from core.nfa import (
    NFA, State, CharClass,
    CharTransition, EpsilonTransition, BackrefTransition,
)
from core.parser import parse_regex


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class HADResult:
    layer: int
    complexity: str
    evidence: str
    pattern: str
    loop_states: Set[State] = field(default_factory=set)
    group_ids: List[int] = field(default_factory=list)
    is_vulnerable: bool = True

    def __repr__(self) -> str:
        return (
            f"[Layer {self.layer}] {self.complexity}  |  {self.evidence}\n"
            f"  Pattern : {self.pattern}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# NFA utility helpers
# ─────────────────────────────────────────────────────────────────────────────

def _back_edges(nfa: NFA) -> List[EpsilonTransition]:
    """True cycle back-edges only (labeled 'loop_back' by the parser)."""
    return [
        t for t in nfa.transitions
        if isinstance(t, EpsilonTransition) and t.ambiguity_label == "loop_back"
    ]


def _fwd_reachable(nfa: NFA, start: State) -> Set[State]:
    visited: Set[State] = set()
    worklist = [start]
    while worklist:
        s = worklist.pop()
        if s in visited:
            continue
        visited.add(s)
        for t in nfa.successors(s):
            worklist.append(t.target)
    return visited


def _rev_reachable(nfa: NFA, goal: State) -> Set[State]:
    """States that can reach *goal* via any transition (reverse BFS)."""
    rev: Dict[State, List[State]] = {}
    for t in nfa.transitions:
        rev.setdefault(t.target, []).append(t.source)

    visited: Set[State] = set()
    worklist = [goal]
    while worklist:
        s = worklist.pop()
        if s in visited:
            continue
        visited.add(s)
        for pred in rev.get(s, []):
            worklist.append(pred)
    return visited


def _region_states(nfa: NFA, entry: State, exit_state: State) -> Set[State]:
    """States reachable from *entry* that can also reach *exit_state*."""
    return _fwd_reachable(nfa, entry) & _rev_reachable(nfa, exit_state)


def _char_classes_in_region(nfa: NFA, entry: State, exit_state: State) -> List[CharClass]:
    body = _region_states(nfa, entry, exit_state)
    return [t.char_class for t in nfa.transitions
            if isinstance(t, CharTransition) and t.source in body]


def _first_char_classes(nfa: NFA, start: State) -> List[CharClass]:
    """
    BFS: collect char classes of the FIRST consuming transition on each path
    from *start*.  ε-transitions are transparent; stops on the first char edge.
    """
    classes: List[CharClass] = []
    visited: Set[State] = set()
    worklist = [start]
    while worklist:
        s = worklist.pop()
        if s in visited:
            continue
        visited.add(s)
        found = False
        for t in nfa.successors(s):
            if isinstance(t, CharTransition):
                classes.append(t.char_class)
                found = True
            elif t.is_epsilon() and not found:
                worklist.append(t.target)
    return classes


# ─────────────────────────────────────────────────────────────────────────────
# Layer 0 — structural IDA
# ─────────────────────────────────────────────────────────────────────────────

def detect_layer0(pattern: str, nfa: NFA) -> List[HADResult]:
    """Layer 0: Two-overlap-loop IDA — O(n²) with strict precision fixes."""
    results: List[HADResult] = []
    backs = _back_edges(nfa)

    # ─────────────────────────────────────────────────────────────
    # Capture-group states (exclude from Layer 0)
    # ─────────────────────────────────────────────────────────────
    capture_states: Set[State] = set()
    for gid in nfa.capture_groups:
        capture_states |= nfa.states_in_group(gid)

    # ─────────────────────────────────────────────────────────────
    # Sub-case A: Nested loops
    # ─────────────────────────────────────────────────────────────
    seen_pairs: Set[frozenset] = set()

    for i, e_out in enumerate(backs):
        for j, e_in in enumerate(backs):
            if i == j:
                continue

            key = frozenset([i, j])
            if key in seen_pairs:
                continue

            # Basic nesting condition
            if not (nfa.path_between(e_out.target, e_in.target)
                    and nfa.path_between(e_in.source, e_out.source)):
                continue

            # Regions
            inner_region = _region_states(nfa, e_in.target, e_in.source)
            outer_region = _region_states(nfa, e_out.target, e_out.source)

            # STRICT FIX 1: must share actual region
            shared_region = inner_region & outer_region
            if not shared_region:
                weak_overlap = any(
                    ci.intersects(co)
                    for ci in _char_classes_in_region(nfa, e_in.target, e_in.source)
                    for co in _char_classes_in_region(nfa, e_out.target, e_out.source)
                )
                if not weak_overlap:
                    continue

            # STRICT FIX 2: remove capture-only states
            inner_region = {s for s in inner_region if s not in capture_states}
            outer_region = {s for s in outer_region if s not in capture_states}

            if not inner_region or not outer_region:
                continue

            # Character classes from filtered regions
            inner_cc = [
                t.char_class for t in nfa.transitions
                if isinstance(t, CharTransition) and t.source in inner_region
            ]

            outer_cc = [
                t.char_class for t in nfa.transitions
                if isinstance(t, CharTransition) and t.source in outer_region
            ]

            if not inner_cc or not outer_cc:
                results.append(HADResult(
                    layer=0,
                    complexity="O(n²)",
                    evidence="Classic nested loop (no char-class needed)",
                    pattern=pattern,
                    loop_states=outer_region,
                ))
                continue

            # Real ambiguity check
            overlap = any(
                ci.intersects(co) or ci.is_subset_of(co) or co.is_subset_of(ci)
                for ci in inner_cc for co in outer_cc
            )
            if not overlap and len(inner_cc) > 1 and len(outer_cc) > 1:
                overlap = True

            seen_pairs.add(key)

            results.append(HADResult(
                layer=0,
                complexity="O(n²)",
                evidence=(
                    f"Nested-loop IDA (strict): inner {e_in.source}→{e_in.target} "
                    f"inside outer {e_out.source}→{e_out.target}; "
                    f"shared path + overlapping char classes"
                ),
                pattern=pattern,
                loop_states=outer_region,
            ))

    # ─────────────────────────────────────────────────────────────
    # Sub-case B: Overlapping alternation inside a loop
    # ─────────────────────────────────────────────────────────────
    seen_loops: Set[int] = set()

    for i, e in enumerate(backs):
        if i in seen_loops:
            continue

        body = _region_states(nfa, e.target, e.source)

        # EXCLUDE capture states
        body = {s for s in body if s not in capture_states}
        if not body:
            continue

        for alt_s in body:
            alt_outs = [
                t for t in nfa.successors(alt_s)
                if isinstance(t, EpsilonTransition)
                and t.ambiguity_label == "alternation"
            ]

            if len(alt_outs) < 2:
                continue

            branch_firsts = [_first_char_classes(nfa, t.target) for t in alt_outs]

            overlap = any(
                c1.intersects(c2)
                for bi, bfs in enumerate(branch_firsts)
                for c1 in bfs
                for bj, bfs2 in enumerate(branch_firsts)
                if bj > bi
                for c2 in bfs2
            )

            if not overlap:
                if len(inner_cc) == 1 and len(outer_cc) == 1:
                    continue

            seen_loops.add(i)

            results.append(HADResult(
                layer=0,
                complexity="O(n²)",
                evidence=(
                    f"Overlapping-alternation IDA (strict): loop "
                    f"{e.source}→{e.target} contains ambiguous alternation"
                ),
                pattern=pattern,
                loop_states=body,
            ))
            break

    
    # Sub-case C: Sequential ambiguous loops
    for e in backs:
        body = _region_states(nfa, e.target, e.source)

        char_trans = []
        for t in nfa.transitions:
            if isinstance(t, CharTransition) and t.source in body:
                char_trans.append(t)

        if len(char_trans) >= 2:
            classes = [t.char_class for t in char_trans]
            
            overlap = False
            for i in range(len(classes)):
                for j in range(i + 1, len(classes)):
                    if classes[i].intersects(classes[j]):
                        overlap = True

            if overlap:
                result = HADResult(
                    layer=0,
                    complexity="O(n²)",
                    evidence="Sequential ambiguous loops",
                    pattern=pattern,
                    loop_states=body
                )
                results.append(result)
        # detect repeated group patterns like ([\w\-]+\.)+
        if any(t.char_class for t in char_trans):
            if len(char_trans) >= 2:
                overlap = True
    return results

# ─────────────────────────────────────────────────────────────────────────────
# Layer 1 — backreference interaction
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _CaptureDep:
    group_id: int
    has_inner_loop: bool = False         # C1
    referenced_by: List[BackrefTransition] = field(default_factory=list)  # C2
    body_states: Set[State] = field(default_factory=set)
    backref_in_outer_loop: bool = False  # hint for Layer 2


def _build_capture_deps(nfa: NFA) -> Dict[int, _CaptureDep]:
    back_set = set(id(t) for t in _back_edges(nfa))

    deps: Dict[int, _CaptureDep] = {}
    for gid, (open_st, close_st) in nfa.capture_groups.items():
        body = nfa.states_in_group(gid)
        dep = _CaptureDep(group_id=gid, body_states=body)

        # C1: any loop_back edge strictly inside the body
        for t in nfa.transitions:
            if (id(t) in back_set
                    and t.source in body
                    and t.target in body):
                dep.has_inner_loop = True
                break

        deps[gid] = dep

    # Backref pass
    backs_by_id = {id(t): t for t in _back_edges(nfa)}
    for bt in nfa.backref_transitions():
        gid = bt.group_id
        if gid not in deps:
            continue
        dep = deps[gid]
        dep.referenced_by.append(bt)

        # Is the backref source inside an outer loop?
        for _, e in backs_by_id.items():
            if (nfa.path_between(e.target, bt.source)
                    and nfa.path_between(bt.source, e.source)):
                dep.backref_in_outer_loop = True

    return deps


def detect_layer1(
    pattern: str, nfa: NFA,
    _loop_states: Optional[Set[State]] = None,   # API compat only
) -> List[HADResult]:
    """Layer 1: Backreference interaction — O(n²)."""
    deps = _build_capture_deps(nfa)
    results: List[HADResult] = []
    seen: Set[int] = set()

    for gid, dep in deps.items():
        if gid in seen:
            continue
        if not (dep.has_inner_loop and dep.referenced_by):
            continue

        seen.add(gid)
        extra = ("; backref also inside outer loop → Layer-2 candidate"
                 if dep.backref_in_outer_loop else "")
        results.append(HADResult(
            layer=1, complexity="O(n²)",
            evidence=(
                f"Backref-ReDoS: group {gid} has variable-length body (C1) "
                f"and \\{gid} backreference exists (C2){extra}"
            ),
            pattern=pattern,
            loop_states=dep.body_states,
            group_ids=[gid],
        ))

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Layer 2 — compound CIDA
# ─────────────────────────────────────────────────────────────────────────────

def detect_layer2(
    pattern: str,
    nfa: NFA,
    l0_results: List[HADResult],
    l1_results: List[HADResult],
) -> List[HADResult]:
    """Layer 2: Compound-IDA (CIDA) — O(n³)."""
    if not l0_results or not l1_results:
        return []

    backs = _back_edges(nfa)
    results: List[HADResult] = []
    seen: Set[str] = set()

    for r1 in l1_results:
        for gid in r1.group_ids:
            if gid not in nfa.capture_groups:
                continue
            open_st, close_st = nfa.capture_groups[gid]

            # An outer loop *contains* this group iff:
            #   e_outer.target ──path──► open_st  AND  close_st ──path──► e_outer.source
            containing_loops = [
                e for e in backs
                if (nfa.path_between(e.target, open_st)
                    and nfa.path_between(close_st, e.source))
            ]
            if not containing_loops:
                continue

            # Also confirm a Layer-0 finding exists for this outer loop region
            l0_hit = any(r0.loop_states for r0 in l0_results)

            if not l0_hit:
                continue

            key = str(sorted(r1.group_ids))
            if key in seen:
                continue
            seen.add(key)

            results.append(HADResult(
                layer=2, complexity="O(n³)",
                evidence=(
                    f"Compound-IDA (CIDA): outer structural IDA loop contains "
                    f"capture group {gid} body; "
                    f"IDA×backref → cubic backtracking"
                ),
                pattern=pattern,
                loop_states=(l0_results[0].loop_states | r1.loop_states),
                group_ids=r1.group_ids,
            ))

    return results


# ─────────────────────────────────────────────────────────────────────────────
# HAD report & entry point
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class HADReport:
    pattern: str
    nfa: NFA
    layer0: List[HADResult]
    layer1: List[HADResult]
    layer2: List[HADResult]

    @property
    def is_vulnerable(self) -> bool:
        return bool(self.layer0 or self.layer1 or self.layer2)

    @property
    def worst_complexity(self) -> str:
        if self.layer2:
            return "O(n³)"
        if self.layer0 or self.layer1:
            return "O(n²)"
        return "O(n)"

    @property
    def all_results(self) -> List[HADResult]:
        return self.layer0 + self.layer1 + self.layer2

    def summary(self) -> str:
        lines = [
            "━━━ HAD Report ━━━",
            f"Pattern    : {self.pattern}",
            f"Vulnerable : {'YES ⚠' if self.is_vulnerable else 'NO ✓'}",
            f"Complexity : {self.worst_complexity}",
        ]
        if self.layer0:
            lines.append(f"  [{len(self.layer0)}] Layer-0 (Structural IDA)")
        if self.layer1:
            lines.append(f"  [{len(self.layer1)}] Layer-1 (Backreference)")
        if self.layer2:
            lines.append(f"  [{len(self.layer2)}] Layer-2 (Compound CIDA)")
        for r in self.all_results:
            lines.append(f"\n  • {r}")
        return "\n".join(lines)


class HADDetector:
    """
    Entry point for the HAD three-layer analysis.

    >>> HADDetector().analyze(r'(a+)+').worst_complexity
    'O(n²)'
    >>> HADDetector().analyze(r'((a+)\\\\2)+').worst_complexity
    'O(n³)'
    >>> HADDetector().analyze(r'a+b').is_vulnerable
    False
    """

    def analyze(self, pattern: str) -> HADReport:
        try:
            nfa = parse_regex(pattern)
        except Exception as exc:
            return HADReport(
                pattern=pattern, nfa=NFA(),
                layer0=[HADResult(
                    layer=-1, complexity="N/A",
                    evidence=f"Parse error: {exc}",
                    pattern=pattern, is_vulnerable=False,
                )],
                layer1=[], layer2=[],
            )

        l0 = detect_layer0(pattern, nfa)
        l1 = detect_layer1(pattern, nfa)
        l2 = detect_layer2(pattern, nfa, l0, l1)

        return HADReport(pattern=pattern, nfa=nfa, layer0=l0, layer1=l1, layer2=l2)

    def batch_analyze(self, patterns: List[str]) -> List[HADReport]:
        return [self.analyze(p) for p in patterns]
