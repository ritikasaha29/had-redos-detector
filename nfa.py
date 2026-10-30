"""
had_framework/core/nfa.py
=========================
Augmented NFA with labeled ambiguity edges and backreference transitions.

Architecture
------------
- CharClass         : immutable character-set abstraction with intersection test
- State             : NFA node carrying capture-group membership metadata
- Transition family : CharTransition | EpsilonTransition | CaptureTransition | BackrefTransition
- NFAFragment       : Thompson-construction building block (start + end state)
- NFA               : full automaton with HAD-specific graph queries

HAD layers use the NFA as follows
  Layer 0 (Structural) — reads EpsilonTransition.ambiguity_label edges
  Layer 1 (Memory)     — reads BackrefTransition + CaptureTransition metadata
  Layer 2 (Compound)   — combines both views
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Character classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CharClass:
    """Immutable representation of a set of characters."""

    chars: FrozenSet[str]
    negated: bool = False   # True → complement set
    any_char: bool = False  # True → dot  (matches everything)

    # ── factories ────────────────────────────────────────────────────────────

    @classmethod
    def literal(cls, c: str) -> "CharClass":
        return cls(chars=frozenset([c]))

    @classmethod
    def from_set(cls, chars: Set[str], negated: bool = False) -> "CharClass":
        return cls(chars=frozenset(chars), negated=negated)

    @classmethod
    def any(cls) -> "CharClass":
        return cls(chars=frozenset(), any_char=True)

    @classmethod
    def word(cls) -> "CharClass":
        r"""\\w  =  [a-zA-Z0-9_]"""
        s = set(
            "abcdefghijklmnopqrstuvwxyz"
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "0123456789_"
        )
        return cls(chars=frozenset(s))

    @classmethod
    def digit(cls) -> "CharClass":
        return cls(chars=frozenset("0123456789"))

    @classmethod
    def whitespace(cls) -> "CharClass":
        return cls(chars=frozenset(" \t\n\r\f\v"))

    # ── membership ───────────────────────────────────────────────────────────

    def accepts(self, c: str) -> bool:
        if self.any_char:
            return True
        return (c not in self.chars) if self.negated else (c in self.chars)

    def intersects(self, other: "CharClass") -> bool:
        """True iff some character is accepted by both classes."""
        if self.any_char or other.any_char:
            return True
        if not self.negated and not other.negated:
            return bool(self.chars & other.chars)
        if self.negated and not other.negated:
            return bool(other.chars - self.chars)      # other has chars not in self's exclusion
        if not self.negated and other.negated:
            return bool(self.chars - other.chars)
        # both negated → complement-of-A ∩ complement-of-B = complement-of-(A∪B), non-empty
        return True

    def __repr__(self) -> str:
        if self.any_char:
            return "."
        prefix = "^" if self.negated else ""
        return f"[{prefix}{''.join(sorted(self.chars))}]"


# ─────────────────────────────────────────────────────────────────────────────
# NFA States
# ─────────────────────────────────────────────────────────────────────────────

_state_counter = itertools.count()


@dataclass
class State:
    """A single NFA node."""

    id: int = field(default_factory=lambda: next(_state_counter))
    accepting: bool = False
    # Capture-group membership (set during parsing)
    group_starts: Set[int] = field(default_factory=set)  # groups that open HERE
    group_ends: Set[int] = field(default_factory=set)    # groups that close HERE

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return isinstance(other, State) and self.id == other.id

    def __repr__(self):
        tag = "*" if self.accepting else ""
        return f"S{self.id}{tag}"


# ─────────────────────────────────────────────────────────────────────────────
# Transitions
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Transition:
    """Abstract base transition."""
    source: State
    target: State

    def is_epsilon(self) -> bool:
        return False


@dataclass
class CharTransition(Transition):
    """Consumes exactly one character from char_class."""
    char_class: CharClass

    def __repr__(self):
        return f"{self.source} --{self.char_class}--> {self.target}"


@dataclass
class EpsilonTransition(Transition):
    """
    Silent ε-transition.

    ambiguity_label encodes the structural role for Layer 0:
      'loop'        — back-edge of a *, + or {n,} repetition
      'alternation' — branch of |
      'overlap'     — synthesized: two paths from same state that both loop
    """
    ambiguity_label: Optional[str] = None

    def is_epsilon(self) -> bool:
        return True

    def __repr__(self):
        lbl = f"[{self.ambiguity_label}]" if self.ambiguity_label else ""
        return f"{self.source} --ε{lbl}--> {self.target}"


@dataclass
class CaptureTransition(Transition):
    """
    Marks the open/close boundary of capture group gid.

    Treated as ε in the NFA simulation; used by the capture-dependency
    graph to map which states lie inside which groups (Layer 1).
    """
    group_id: int
    is_open: bool   # True → '(' boundary;  False → ')' boundary

    def is_epsilon(self) -> bool:
        return True

    def __repr__(self):
        kind = "(" if self.is_open else ")"
        return f"{self.source} --G{self.group_id}{kind}--> {self.target}"


@dataclass
class BackrefTransition(Transition):
    r"""
    Backreference \N — consumes whatever group N most recently captured.

    Length is unknown at compile time; the transition carries only the
    group id for static analysis (Layer 1).
    """
    group_id: int

    def is_epsilon(self) -> bool:
        return False

    def __repr__(self):
        return f"{self.source} --\\{self.group_id}--> {self.target}"


# ─────────────────────────────────────────────────────────────────────────────
# NFA Fragment  (Thompson-construction brick)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class NFAFragment:
    """
    A sub-NFA with exactly one start state and one (non-accepting) end state.

    The end state is wired to accepting=True only after the full NFA is
    assembled by the parser.
    """
    start: State
    end: State
    states: Set[State] = field(default_factory=set)
    transitions: List[Transition] = field(default_factory=list)

    def add_transition(self, t: Transition) -> None:
        self.transitions.append(t)
        self.states.add(t.source)
        self.states.add(t.target)


# ─────────────────────────────────────────────────────────────────────────────
# Full NFA
# ─────────────────────────────────────────────────────────────────────────────

class NFA:
    """
    Augmented NFA for the HAD framework.

    Supports:
      • Standard character transitions
      • Labeled ε-transitions (ambiguity edges for Layer 0)
      • Capture open/close transitions (Layer 1 metadata)
      • Backreference transitions (Layer 1 core)

    Key invariants
    --------------
    - capture_groups[gid] = (open_state, close_state) for every group gid
    - All states are reachable from self.start
    """

    def __init__(self):
        self.states: Set[State] = set()
        self.transitions: List[Transition] = []
        self.start: Optional[State] = None
        self.accept: Optional[State] = None
        self.capture_groups: Dict[int, Tuple[State, State]] = {}

    # ── construction helpers ─────────────────────────────────────────────────

    def add_state(self, accepting: bool = False) -> State:
        s = State(accepting=accepting)
        self.states.add(s)
        return s

    def add_transition(self, t: Transition) -> None:
        self.transitions.append(t)
        self.states.add(t.source)
        self.states.add(t.target)

    def add_char(self, src: State, tgt: State, cc: CharClass) -> CharTransition:
        t = CharTransition(source=src, target=tgt, char_class=cc)
        self.add_transition(t)
        return t

    def add_epsilon(
        self, src: State, tgt: State, label: Optional[str] = None
    ) -> EpsilonTransition:
        t = EpsilonTransition(source=src, target=tgt, ambiguity_label=label)
        self.add_transition(t)
        return t

    def add_capture_open(self, src: State, tgt: State, gid: int) -> CaptureTransition:
        t = CaptureTransition(source=src, target=tgt, group_id=gid, is_open=True)
        self.add_transition(t)
        return t

    def add_capture_close(self, src: State, tgt: State, gid: int) -> CaptureTransition:
        t = CaptureTransition(source=src, target=tgt, group_id=gid, is_open=False)
        self.add_transition(t)
        return t

    def add_backref(self, src: State, tgt: State, gid: int) -> BackrefTransition:
        t = BackrefTransition(source=src, target=tgt, group_id=gid)
        self.add_transition(t)
        return t

    # ── adjacency ────────────────────────────────────────────────────────────

    def successors(self, state: State) -> List[Transition]:
        return [t for t in self.transitions if t.source == state]

    # ── closure / reachability ───────────────────────────────────────────────

    def epsilon_closure(self, states: Set[State]) -> Set[State]:
        """
        Standard ε-closure.

        Includes CaptureTransition edges (they are semantically ε)
        but NOT BackrefTransition edges (consume input).
        """
        closure = set(states)
        worklist = list(states)
        while worklist:
            s = worklist.pop()
            for t in self.successors(s):
                if t.is_epsilon() and t.target not in closure:
                    closure.add(t.target)
                    worklist.append(t.target)
        return closure

    def reachable_states(self) -> Set[State]:
        """BFS from start state across all transition types."""
        if self.start is None:
            return set()
        visited: Set[State] = set()
        worklist = [self.start]
        while worklist:
            s = worklist.pop()
            if s in visited:
                continue
            visited.add(s)
            for t in self.successors(s):
                worklist.append(t.target)
        return visited

    def path_between(self, src: State, tgt: State) -> bool:
        """True iff there is a directed path from src to tgt (BFS)."""
        if src == tgt:
            return True
        visited: Set[State] = set()
        worklist = [src]
        while worklist:
            s = worklist.pop()
            if s in visited:
                continue
            visited.add(s)
            for t in self.successors(s):
                if t.target == tgt:
                    return True
                worklist.append(t.target)
        return False

    def states_in_group(self, group_id: int) -> Set[State]:
        """
        Return all states strictly inside the body of capture group gid.

        Uses BFS from the open-state; stops at (but excludes) the close-state.
        """
        if group_id not in self.capture_groups:
            return set()
        open_st, close_st = self.capture_groups[group_id]
        visited: Set[State] = set()
        worklist = [open_st]
        while worklist:
            s = worklist.pop()
            if s in visited or s is close_st:
                continue
            visited.add(s)
            for t in self.successors(s):
                worklist.append(t.target)
        return visited

    # ── loop / cycle detection ───────────────────────────────────────────────

    def loop_states(self) -> Set[State]:
        """
        Return states that participate in a cycle.

        We compute SCCs via iterative DFS (Tarjan's algorithm), then mark
        every state in an SCC of size > 1 *or* with a self-loop.
        """
        index_counter = itertools.count()
        index: Dict[State, int] = {}
        lowlink: Dict[State, int] = {}
        on_stack: Set[State] = set()
        stack: List[State] = []
        sccs: List[Set[State]] = []

        def strongconnect(v: State):
            idx = next(index_counter)
            index[v] = lowlink[v] = idx
            stack.append(v)
            on_stack.add(v)

            for t in self.successors(v):
                w = t.target
                if w not in index:
                    strongconnect(w)
                    lowlink[v] = min(lowlink[v], lowlink[w])
                elif w in on_stack:
                    lowlink[v] = min(lowlink[v], index[w])

            if lowlink[v] == index[v]:
                scc: Set[State] = set()
                while True:
                    w = stack.pop()
                    on_stack.discard(w)
                    scc.add(w)
                    if w is v:
                        break
                sccs.append(scc)

        for s in self.states:
            if s not in index:
                strongconnect(s)

        in_loop: Set[State] = set()
        for scc in sccs:
            if len(scc) > 1:
                in_loop |= scc
            else:
                # Single-node SCC — check for self-loop
                (v,) = scc
                for t in self.successors(v):
                    if t.target is v:
                        in_loop.add(v)
                        break
        return in_loop

    # ── typed transition accessors ───────────────────────────────────────────

    def char_transitions(self) -> List[CharTransition]:
        return [t for t in self.transitions if isinstance(t, CharTransition)]

    def backref_transitions(self) -> List[BackrefTransition]:
        return [t for t in self.transitions if isinstance(t, BackrefTransition)]

    def epsilon_transitions_labeled(self, label: str) -> List[EpsilonTransition]:
        return [
            t for t in self.transitions
            if isinstance(t, EpsilonTransition) and t.ambiguity_label == label
        ]

    # ── debug ─────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        lines = [f"NFA(start={self.start}, accept={self.accept}, "
                 f"states={len(self.states)}, transitions={len(self.transitions)})"]
        for t in sorted(self.transitions, key=lambda x: x.source.id):
            lines.append(f"  {t}")
        return "\n".join(lines)
