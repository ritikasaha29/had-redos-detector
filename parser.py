"""
had_framework/core/parser.py
============================
Recursive-descent regex parser → Augmented NFA (Thompson construction).

Supported syntax
----------------
  Literals        a  b  c  …
  Dot             .
  Escape          \\d  \\w  \\s  \\D  \\W  \\S  and \\N (backreference)
  Character class [abc]  [^abc]  [a-z]
  Quantifiers     *  +  ?  {n}  {n,}  {n,m}   (greedy only)
  Grouping        (…)   (?:…)   (?=…)  (?!…)   (lookaheads stripped)
  Alternation     |
  Anchors         ^  $  (silently consumed — not modelled in NFA)

Ambiguity labeling
------------------
Ε-edges that introduce non-determinism are tagged so Layer 0 can find them:
  'loop'        — the back-edge of a  *  or  +  repetition
  'alternation' — branches of  |
"""

from __future__ import annotations

from typing import List, Optional, Set, Tuple

from core.nfa import (
    NFA, NFAFragment, State,
    CharClass, CharTransition, EpsilonTransition,
    CaptureTransition, BackrefTransition,
)


# ─────────────────────────────────────────────────────────────────────────────
# Tokeniser / scanner
# ─────────────────────────────────────────────────────────────────────────────

class RegexScanner:
    """
    Thin character-level scanner with one-character lookahead.
    """

    def __init__(self, pattern: str):
        self.pattern = pattern
        self.pos = 0

    def peek(self) -> Optional[str]:
        return self.pattern[self.pos] if self.pos < len(self.pattern) else None

    def peek_ahead(self, offset: int = 1) -> Optional[str]:
        p = self.pos + offset
        return self.pattern[p] if p < len(self.pattern) else None

    def consume(self) -> str:
        c = self.pattern[self.pos]
        self.pos += 1
        return c

    def expect(self, c: str) -> None:
        got = self.consume()
        if got != c:
            raise SyntaxError(
                f"Expected '{c}' at pos {self.pos - 1}, got '{got}'"
                f"\n  pattern: {self.pattern}"
            )

    def at_end(self) -> bool:
        return self.pos >= len(self.pattern)

    def rest(self) -> str:
        return self.pattern[self.pos:]


# ─────────────────────────────────────────────────────────────────────────────
# Parser
# ─────────────────────────────────────────────────────────────────────────────

class RegexParser:
    """
    Converts a regex string to an augmented NFA via Thompson construction.

    Grammar (informal, left-to-right precedence)
    ---------------------------------------------
      regex      ::= alternation
      alternation::= concat ( '|' concat )*
      concat     ::= quantified *
      quantified ::= atom quantifier?
      quantifier ::= '*' | '+' | '?' | '{' digits (',' digits?)? '}'
      atom       ::= literal | '.' | escape | charclass | group
      group      ::= '(' group_prefix? regex ')'
      group_prefix::= '?:' | '?=' | '?!'
    """

    def __init__(self, pattern: str):
        self.scan = RegexScanner(pattern)
        self.nfa = NFA()
        self._group_counter = 0          # capture group id allocator

    # ── public entry ─────────────────────────────────────────────────────────

    def parse(self) -> NFA:
        frag = self._parse_alternation()
        self.nfa.start = frag.start
        frag.end.accepting = True
        self.nfa.accept = frag.end
        self.nfa.states |= frag.states
        self.nfa.transitions.extend(
            [t for t in frag.transitions if t not in self.nfa.transitions]
        )
        return self.nfa

    # ── grammar rules ────────────────────────────────────────────────────────

    def _parse_alternation(self) -> NFAFragment:
        """alternation ::= concat ( '|' concat )*"""
        branches: List[NFAFragment] = [self._parse_concat()]
        while self.scan.peek() == '|':
            self.scan.consume()
            branches.append(self._parse_concat())

        if len(branches) == 1:
            return branches[0]

        # Thompson alternation: new start + end, ε-edges to each branch
        start = self.nfa.add_state()
        end = self.nfa.add_state()
        frag = NFAFragment(start=start, end=end)

        for b in branches:
            self.nfa.add_epsilon(start, b.start, label="alternation")
            self.nfa.add_epsilon(b.end, end, label="alternation")
            frag.states |= b.states
            frag.transitions.extend(b.transitions)

        frag.states |= {start, end}
        return frag

    def _parse_concat(self) -> NFAFragment:
        """concat ::= quantified *"""
        parts: List[NFAFragment] = []
        while self.scan.peek() not in (None, '|', ')'):
            parts.append(self._parse_quantified())

        if not parts:
            # Empty alternative — ε fragment
            s = self.nfa.add_state()
            e = self.nfa.add_state()
            self.nfa.add_epsilon(s, e)
            return NFAFragment(start=s, end=e, states={s, e})

        # Chain: end of parts[i] → start of parts[i+1]
        result = parts[0]
        for i in range(1, len(parts)):
            nxt = parts[i]
            self.nfa.add_epsilon(result.end, nxt.start)
            result = NFAFragment(
                start=result.start,
                end=nxt.end,
                states=result.states | nxt.states,
                transitions=result.transitions + nxt.transitions,
            )
        return result

    def _parse_quantified(self) -> NFAFragment:
        """quantified ::= atom quantifier?"""
        atom = self._parse_atom()
        q = self.scan.peek()

        if q == '*':
            self.scan.consume()
            return self._make_star(atom)
        if q == '+':
            self.scan.consume()
            return self._make_plus(atom)
        if q == '?':
            self.scan.consume()
            return self._make_optional(atom)
        if q == '{':
            return self._parse_counted_quantifier(atom)

        return atom

    def _parse_counted_quantifier(self, atom: NFAFragment) -> NFAFragment:
        """
        {n}  {n,}  {n,m}  — expand to concatenated / starred copies.

        We keep it simple: {n,} → n copies followed by *.
                           {n,m} → n copies followed by (n-m) optional copies.
        """
        self.scan.expect('{')
        n = self._read_digits()
        m: Optional[int] = n

        if self.scan.peek() == ',':
            self.scan.consume()
            if self.scan.peek() == '}':
                m = None   # unbounded
            else:
                m = self._read_digits()

        self.scan.expect('}')

        # We'll just model this as star for detection purposes
        # (exact counts don't affect ambiguity class)
        if m is None or (m is not None and m > n):
            return self._make_star(atom)
        return atom   # {n,n} — exact, no ambiguity

    def _read_digits(self) -> int:
        buf = []
        while self.scan.peek() and self.scan.peek().isdigit():
            buf.append(self.scan.consume())
        if not buf:
            raise SyntaxError("Expected digit in counted quantifier")
        return int(''.join(buf))

    # ── atom ─────────────────────────────────────────────────────────────────

    def _parse_atom(self) -> NFAFragment:
        c = self.scan.peek()

        if c == '(':
            return self._parse_group()
        if c == '[':
            return self._parse_char_class()
        if c == '\\':
            return self._parse_escape()
        if c in ('^', '$'):
            # Anchors: consume silently, produce ε fragment
            self.scan.consume()
            s = self.nfa.add_state()
            e = self.nfa.add_state()
            self.nfa.add_epsilon(s, e)
            return NFAFragment(start=s, end=e, states={s, e})
        if c == '.':
            self.scan.consume()
            return self._make_char_frag(CharClass.any())
        if c is not None and c not in (')', '|', '*', '+', '?', '{', '}'):
            self.scan.consume()
            return self._make_char_frag(CharClass.literal(c))

        # Fallback: ε (shouldn't normally be hit)
        s = self.nfa.add_state()
        e = self.nfa.add_state()
        self.nfa.add_epsilon(s, e)
        return NFAFragment(start=s, end=e, states={s, e})

    # ── groups ────────────────────────────────────────────────────────────────

    def _parse_group(self) -> NFAFragment:
        self.scan.expect('(')

        # Detect special prefixes: (?:  (?=  (?!
        non_capturing = False
        lookahead = False

        if self.scan.peek() == '?':
            self.scan.consume()
            nxt = self.scan.peek()
            if nxt == ':':
                self.scan.consume()
                non_capturing = True
            elif nxt in ('=', '!'):
                self.scan.consume()
                lookahead = True
            else:
                # Unknown extension — treat as non-capturing
                non_capturing = True

        if non_capturing or lookahead:
            inner = self._parse_alternation()
            self.scan.expect(')')
            if lookahead:
                # Lookaheads: model as ε (not actually evaluated in NFA)
                s = self.nfa.add_state()
                e = self.nfa.add_state()
                self.nfa.add_epsilon(s, e)
                return NFAFragment(start=s, end=e, states={s, e})
            return inner

        # Capturing group
        self._group_counter += 1
        gid = self._group_counter

        # Create boundary states
        open_st = self.nfa.add_state()
        open_st.group_starts.add(gid)
        close_st = self.nfa.add_state()
        close_st.group_ends.add(gid)

        inner = self._parse_alternation()
        self.scan.expect(')')

        # Wire: open_st → inner.start → inner.end → close_st
        self.nfa.add_capture_open(open_st, inner.start, gid)
        self.nfa.add_capture_close(inner.end, close_st, gid)

        # Register in NFA
        self.nfa.capture_groups[gid] = (open_st, close_st)

        frag = NFAFragment(
            start=open_st,
            end=close_st,
            states=inner.states | {open_st, close_st},
            transitions=inner.transitions,
        )
        return frag

    # ── escapes ───────────────────────────────────────────────────────────────

    def _parse_escape(self) -> NFAFragment:
        self.scan.expect('\\')
        c = self.scan.consume()

        if c.isdigit():
            gid = int(c)
            return self._make_backref_frag(gid)

        cc_map = {
            'd': CharClass.digit(),
            'D': CharClass.from_set(set("0123456789"), negated=True),
            'w': CharClass.word(),
            'W': CharClass.from_set(
                set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"),
                negated=True,
            ),
            's': CharClass.whitespace(),
            'S': CharClass.from_set(set(" \t\n\r\f\v"), negated=True),
            'n': CharClass.literal('\n'),
            't': CharClass.literal('\t'),
            'r': CharClass.literal('\r'),
        }
        if c in cc_map:
            return self._make_char_frag(cc_map[c])

        # Literal escape (e.g. \. \* \( )
        return self._make_char_frag(CharClass.literal(c))

    # ── character classes  [abc]  [^a-z] ─────────────────────────────────────

    def _parse_char_class(self) -> NFAFragment:
        self.scan.expect('[')
        negated = False
        if self.scan.peek() == '^':
            self.scan.consume()
            negated = True

        chars: Set[str] = set()
        while self.scan.peek() and self.scan.peek() != ']':
            c = self.scan.consume()
            if self.scan.peek() == '-' and self.scan.peek_ahead() not in (None, ']'):
                self.scan.consume()   # consume '-'
                end_c = self.scan.consume()
                for code in range(ord(c), ord(end_c) + 1):
                    chars.add(chr(code))
            elif c == '\\':
                esc = self.scan.consume()
                esc_map = {
                    'd': set("0123456789"),
                    'w': set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"),
                    's': set(" \t\n\r\f\v"),
                    'n': {'\n'}, 't': {'\t'}, 'r': {'\r'},
                }
                chars |= esc_map.get(esc, {esc})
            else:
                chars.add(c)

        self.scan.expect(']')
        cc = CharClass.from_set(chars, negated=negated)
        return self._make_char_frag(cc)

    # ── Thompson builders ────────────────────────────────────────────────────

    def _make_char_frag(self, cc: CharClass) -> NFAFragment:
        s = self.nfa.add_state()
        e = self.nfa.add_state()
        t = self.nfa.add_char(s, e, cc)
        return NFAFragment(start=s, end=e, states={s, e}, transitions=[t])

    def _make_backref_frag(self, gid: int) -> NFAFragment:
        s = self.nfa.add_state()
        e = self.nfa.add_state()
        t = self.nfa.add_backref(s, e, gid)
        return NFAFragment(start=s, end=e, states={s, e}, transitions=[t])

    def _make_star(self, inner: NFAFragment) -> NFAFragment:
        """
        Thompson star.
        Edges are labeled distinctly so the detector can find true back-edges:
          'loop_back'  — the edge that creates the cycle (used by Layer-0 detection)
          'loop_exit'  — the forward exit edge out of the loop body
          'loop_skip'  — the zero-iteration skip edge
        """
        start = self.nfa.add_state()
        end = self.nfa.add_state()

        self.nfa.add_epsilon(start, inner.start)                          # enter body
        self.nfa.add_epsilon(inner.end, end, label="loop_exit")           # exit body
        self.nfa.add_epsilon(inner.end, inner.start, label="loop_back")   # back-edge ←
        self.nfa.add_epsilon(start, end, label="loop_skip")               # skip (zero times)

        return NFAFragment(
            start=start, end=end,
            states=inner.states | {start, end},
            transitions=inner.transitions,
        )

    def _make_plus(self, inner: NFAFragment) -> NFAFragment:
        """
        Thompson plus: body once, then like star.
        Uses same labeling convention as _make_star.
        """
        start = self.nfa.add_state()
        end = self.nfa.add_state()

        self.nfa.add_epsilon(start, inner.start)                          # enter body (no label)
        self.nfa.add_epsilon(inner.end, end, label="loop_exit")           # exit body
        self.nfa.add_epsilon(inner.end, inner.start, label="loop_back")   # back-edge ←

        return NFAFragment(
            start=start, end=end,
            states=inner.states | {start, end},
            transitions=inner.transitions,
        )

    def _make_optional(self, inner: NFAFragment) -> NFAFragment:
        """Thompson optional: enter body OR skip."""
        start = self.nfa.add_state()
        end = self.nfa.add_state()

        self.nfa.add_epsilon(start, inner.start, label="alternation")
        self.nfa.add_epsilon(inner.end, end)
        self.nfa.add_epsilon(start, end, label="alternation")   # skip

        return NFAFragment(
            start=start, end=end,
            states=inner.states | {start, end},
            transitions=inner.transitions,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Public helper
# ─────────────────────────────────────────────────────────────────────────────

def parse_regex(pattern: str) -> NFA:
    """
    Parse *pattern* into an augmented NFA ready for HAD analysis.

    >>> nfa = parse_regex(r'(a+)+')
    >>> len(nfa.states) > 0
    True
    """
    parser = RegexParser(pattern)
    nfa = parser.parse()
    # Propagate all NFA fragments into the top-level state/transition sets
    for t in list(nfa.transitions):
        nfa.states.add(t.source)
        nfa.states.add(t.target)
    return nfa
