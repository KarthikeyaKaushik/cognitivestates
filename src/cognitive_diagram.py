from collections import defaultdict
from typing import Dict, Iterable, List, Tuple, Set
import copy
import math
import numpy as np

# ---------- sequence helpers (operate on tuples of tokens) ----------

def is_prefix(pref: Tuple[str, ...], seq: Tuple[str, ...]) -> bool:
    return len(pref) <= len(seq) and seq[:len(pref)] == pref

def strip_prefix(pref: Tuple[str, ...], seq: Tuple[str, ...]) -> Tuple[str, ...]:
    assert is_prefix(pref, seq)
    return seq[len(pref):]

def all_prefixes(pair: Tuple[Tuple[str, ...], Tuple[str, ...]]) -> Set[Tuple[Tuple[str, ...], Tuple[str, ...]]]:
    u, v = pair
    assert len(u) == len(v), "For Mealy-style samples, |input tokens| must equal |output tokens|"
    return { (u[:i], v[:i]) for i in range(len(u)+1) }

# ---------- Cognitive Diagram (token-based) ----------

class CognitiveDiagram:
    def __init__(self, X: Set[str], Y: Set[str]):
        """
        <Q, X, Y, delta, lambda, q0>
        X, Y are sets of *tokens* (each token is a string; tokens may be multi-character).
        Transitions consume exactly one input token and emit exactly one output token.
        """
        self.start_state = None
        self.X = X  # input tokens
        self.Y = Y  # output tokens (per step)
        self.statetrans: Dict[Tuple[int, str], int] = {}     # (state, x_token) -> state
        self.outputtrans: Dict[Tuple[int, str], str] = {}    # (state, x_token) -> y_token
        self.states: Set[int] = set()

    def _prepare_for_lattice(self):
        """
        Prepares the diagram's data in the format expected by the Lattice class.
        - Creates a .labels dictionary.
        - Creates a .transitions numpy matrix.
        """
        # The Lattice expects a simple mapping of state numbers
        self.labels = {state: state for state in self.states}

        # Create a mapping from string-based input symbols to integer indices
        sorted_inputs = sorted(list(self.X))
        input_map = {symbol: i for i, symbol in enumerate(sorted_inputs)}
        
        num_states = len(self.states)
        num_inputs = len(sorted_inputs)
        
        # Create and populate the transitions matrix
        self.transitions = np.zeros((num_states, num_inputs), dtype=int)
        for (state, symbol), next_state in self.statetrans.items():
            if symbol in input_map:
                input_idx = input_map[symbol]
                self.transitions[state, input_idx] = next_state
    

    # ---------- mechanics ----------

    def add_state(self):
        if not self.states:
            self.start_state = 0
            self.states.add(0)
        else:
            self.states.add(max(self.states) + 1)

    def get_undefined_state_input(self) -> List[Tuple[int, str]]:
        """All (q, x) where delta is undefined."""
        ret = []
        for q in self.states:
            for x in self.X:
                if (q, x) not in self.statetrans:
                    ret.append((q, x))
        return ret

    def apply_transitions(self, state: int, input_tokens: Tuple[str, ...]) -> Tuple[int, Tuple[str, ...]]:
        """
        Run from `state` over a *sequence of tokens* (tuple of str).
        Returns (final_state, output_tokens) where output_tokens has the same length as input_tokens.
        """
        q = state
        outs: List[str] = []
        for x in input_tokens:
            if (q, x) not in self.statetrans:
                return (None, None)  # undefined
            q_next = self.statetrans[(q, x)]
            y = self.outputtrans[(q, x)]
            outs.append(y)
            q = q_next
        return q, tuple(outs)
    
    def trace(self, input_tokens: Tuple[str, ...]) -> List[int]:
        """
        Run from the start state over a *sequence of tokens* (tuple of str).
        Returns the list of states visited, including the start state.
        """
        q = self.start_state
        states_visited = []
        for x in input_tokens:
            q = self.statetrans[(q, x)]
            states_visited.append(q)
        return states_visited


    # ---------- language-theoretic helpers (token-based) ----------

    def get_initial_set(self, e: Set[Tuple[Tuple[str, ...], Tuple[str, ...]]]
                        ) -> Set[Tuple[str, str]]:
        """
        From a set of (u,v) pairs (tuples of tokens), return all first-token pairs (u[0], v[0]).
        """
        init = set()
        for (u, v) in e:
            if len(u) > 0 and len(v) > 0:
                init.add((u[0], v[0]))
        return init

    def get_left_quotient(self,
                          e: Set[Tuple[Tuple[str, ...], Tuple[str, ...]]],
                          f: Set[Tuple[Tuple[str, ...], Tuple[str, ...]]]
                          ) -> Set[Tuple[Tuple[str, ...], Tuple[str, ...]]]:
        """
        E^{-1}F = { (s,v) | ∃(r,u)∈E s.t. (r◦s, u◦v) ∈ F }, all in token-space.
        """
        lq: Set[Tuple[Tuple[str, ...], Tuple[str, ...]]] = set()
        for (rs, uv) in f:
            for (r, u) in e:
                if is_prefix(r, rs) and is_prefix(u, uv):
                    s = strip_prefix(r, rs)
                    v = strip_prefix(u, uv)
                    lq.add((s, v))
        return lq

    def get_prefixes(self, e: Tuple[Tuple[str, ...], Tuple[str, ...]]
                     ) -> Set[Tuple[Tuple[str, ...], Tuple[str, ...]]]:
        return all_prefixes(e)

    def t_g(self, q: int, sf: Set[Tuple[Tuple[str, ...], Tuple[str, ...]]]
            ) -> Set[Tuple[Tuple[str, ...], Tuple[str, ...]]]:
        """
        t_g(q0,q) = { (r,u) | δ(q0,r)=q and λ(q0,r)=u and ∃(s,v) with (r◦s, u◦v)∈sf }.
        Note: We only require δ(q0,r)=q to hold (your original code commented out u matching).
        """
        tg: Set[Tuple[Tuple[str, ...], Tuple[str, ...]]] = set()
        for (rs, uv) in sf:
            for (r, u) in self.get_prefixes((rs, uv)):
                q_prime, u_prime = self.apply_transitions(self.start_state, r)
                if q_prime == q:
                    tg.add((r, u))
        return tg

    def is_compatible(self,
                      x: Tuple[Tuple[str, ...], Tuple[str, ...]],
                      y: Tuple[Tuple[str, ...], Tuple[str, ...]]) -> bool:
        """
        Pairwise consistency: whenever input prefixes match at some length i, the
        corresponding output prefixes must match at i as well. (Token-aligned.)
        """
        (rs, uv) = x
        (rsp, uvp) = y
        L = min(len(rs), len(rsp))
        for i in range(1, L+1):
            if rs[:i] == rsp[:i] and uv[:i] != uvp[:i]:
                return False
        return True

    def is_consistent(self, Xset: Set[Tuple[Tuple[str, ...], Tuple[str, ...]]]) -> bool:
        Xlist = list(Xset)
        n = len(Xlist)
        for i in range(n):
            for j in range(i+1, n):
                if not self.is_compatible(Xlist[i], Xlist[j]):
                    return False
        return True

    def get_condition(self, F: Set[Tuple[Tuple[str, ...], Tuple[str, ...]]]) -> bool:
        undefined = self.get_undefined_state_input()
        if not undefined:
            return False
        for (q, x) in undefined:
            lq = self.get_initial_set(self.get_left_quotient(self.t_g(q, F), F))
            if len(lq) > 0:
                return True
        return False


    def to_tikz(self, radius=2, color1="orange", color2="gray", opacity1=0.7, opacity2=0.7):
        states = sorted(self.states)
        n = len(states)
        tikz = [
            "\\begin{tikzpicture}[->,>=stealth,shorten >=1pt,auto,node distance=2cm,thick,main node/.style={circle,draw,font=\\sffamily\\Large\\bfseries}]"
        ]
        positions = {}
        for idx, state in enumerate(states):
            angle = math.pi - 2 * math.pi * idx / n
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            positions[state] = (x, y)
            tikz.append(f"\\node[main node] (s{state}) at ({x:.2f},{y:.2f}) {{$s_{{{state}}}$}};")

        edge_labels = defaultdict(list)
        for (state, inp), next_state in self.statetrans.items():
            edge_labels[(state, next_state)].append((inp, self.outputtrans.get((state, inp), "")))

        for (state, next_state), transitions in edge_labels.items():
            inps  = [inp for inp, _ in transitions]
            outs  = [out for _, out in transitions]
            inp_label = ", ".join(inps)
            out_label = ", ".join(outs)
            if state < next_state:
                arrow_color, arrow_opacity, label_color = color2, opacity2, color2
            else:
                arrow_color, arrow_opacity, label_color = color1, opacity1, color1

            label = f"\\tiny{{\\textcolor{{{label_color}}}{{$ {inp_label} \\;|\\; {out_label} $}}}}"
            x1, y1 = positions[state]; x2, y2 = positions[next_state]
            if state == next_state:
                tikz.append(
                    f"\\path (s{state}) edge [loop above, {arrow_color}, opacity={arrow_opacity}] "
                    f"node[midway, sloped] {{{label}}} (s{next_state});"
                )
            else:
                pos = 'above' if x2 > x1 else 'below'
                tikz.append(
                    f"\\path (s{state}) edge [bend left=20, {arrow_color}, opacity={arrow_opacity}] "
                    f"node[midway, sloped, {pos}] {{{label}}} (s{next_state});"
                )
        tikz.append("\\end{tikzpicture}")
        return "\n".join(tikz)


def veelenturf(F: List[Tuple[Tuple[str, ...], Tuple[str, ...]]], X: Set[str], Y: Set[str]) -> CognitiveDiagram:
    """
    F: list of (input_tokens, output_tokens) where each is a Tuple[str,...] and len(input_tokens) == len(output_tokens).
    X: set of input tokens; Y: set of output tokens.
    """
    # normalize to set of pairs
    Fset: Set[Tuple[Tuple[str, ...], Tuple[str, ...]]] = set(F)

    cd = CognitiveDiagram(X=X, Y=Y)
    cd.add_state()  # q0

    condition = True
    while condition:
        matched_found, idm = False, 0
        while not matched_found:
            undefined = cd.get_undefined_state_input()
            try:
                (q, x) = sorted(undefined)[idm]  # pick a deterministic undefined (q,x)
            except:
                return cd
            new_pairs = sorted(cd.get_initial_set(cd.get_left_quotient(cd.t_g(q, Fset), Fset)))
            matched_pair = [n_p for n_p in new_pairs if n_p[0] == x]
            if len(matched_pair) == 0:
                idm += 1
            else:
                matched_found = True

        assert len(matched_pair) == 1
        # set output for (q,x)
        cdnew = copy.deepcopy(cd)
        cdnew.outputtrans[(q, x)] = matched_pair[0][1]

        # try linking to an existing state
        case_a = False
        for q_prime in cdnew.states:
            cdnew.statetrans[(q, x)] = q_prime
            consistent = True
            for state in cdnew.states:
                lq = cdnew.get_left_quotient(cdnew.t_g(state, Fset), Fset)
                if not cdnew.is_consistent(lq):
                    consistent = False
                    break
            if consistent:
                case_a = True
                break
            else:
                cdnew.statetrans.pop((q, x))

        # otherwise add new state
        if not case_a:
            cdnew.add_state()
            q_new = max(cdnew.states)
            cdnew.statetrans[(q, x)] = q_new

        cd = copy.deepcopy(cdnew)
        condition = cd.get_condition(Fset)

    return cd
