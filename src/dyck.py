# from typing import List, Tuple

# BRACKET_PAIRS = [("(", ")"), ("<", ">"), ("A", "a"), ("[", "]"), ("<", ">"), ("A", "a"), ("B", "b"), ("C", "c"), ("D", "d"), ("E", "e"), ("F", "f")]

# def get_pairs(k: int) -> List[Tuple[str, str]]:
# 	return BRACKET_PAIRS[:k]

# def enumerate_dyck_with_depth(k: int, m: int, max_pairs: int) -> List[Tuple[str, List[int]]]:
# 	"""
# 	Enumerate all Dyck(k, m) strings with up to `max_pairs` pairs,
# 	returning (string, depth_profile) for each.
# 	Depth at index t is the stack height AFTER consuming symbol t.
# 	"""
# 	pairs = get_pairs(k)
# 	results: List[Tuple[str, List[int]]] = []

# 	def backtrack(current: List[str], depths: List[int], stack: List[int], opened: int, closed: int):
# 		# Record any non-empty balanced string
# 		if opened == closed and opened > 0:
# 			results.append(("".join(current), depths.copy()))
# 		# Stop if we already opened max_pairs pairs
# 		if opened >= max_pairs:
# 			return

# 		# Option 1: open a new bracket (if depth allows)
# 		if len(stack) < m:
# 			for t in range(k):
# 				# open
# 				current.append(pairs[t][0])
# 				stack.append(t)
# 				depths.append(len(stack))  # depth after consuming this opener
# 				backtrack(current, depths, stack, opened + 1, closed)
# 				# undo
# 				depths.pop()
# 				stack.pop()
# 				current.pop()

# 		# Option 2: close a bracket (if possible)
# 		if stack:
# 			t = stack.pop()
# 			current.append(pairs[t][1])
# 			depths.append(len(stack))  # depth after consuming this closer
# 			backtrack(current, depths, stack, opened, closed + 1)
# 			# undo
# 			depths.pop()
# 			current.pop()
# 			stack.append(t)

# 	backtrack([], [], [], 0, 0)
# 	results = [(r[0], [str(k) for k in r[1]]) for r in results]
# 	return results

# def enumerate_dyck(k: int, m: int, max_pairs: int) -> List[str]:
# 	"""
# 	Backwards-compatible: just the strings.
# 	"""
# 	return [s for s, _depth in enumerate_dyck_with_depth(k, m, max_pairs)]

# def filter_by_lengths_strings(strings: List[str], lengths: List[int]) -> List[str]:
# 	return [s for s in strings if len(s) in lengths]

# def filter_by_lengths_with_depth(items: List[Tuple[str, List[int]]], lengths: List[int]) -> List[Tuple[str, List[int]]]:
# 	return [(s, d) for (s, d) in items if len(s) in lengths]

# if __name__ == "__main__":
# 	# Example usage
# 	k = 2   # bracket types
# 	m = 3   # max depth
# 	l = 2   # minimum length
# 	max_len = 6

# 	# (1) Strings only (as before)
# 	all_strings = enumerate_dyck(k, m, max_pairs=max_len // 2)
# 	target_lengths = list(range(l, max_len + 1, 2))  # l, l+2, ...
# 	selected_strings = filter_by_lengths_strings(all_strings, target_lengths)
# 	print("Strings only:")
# 	for s in selected_strings:
# 		print(s)

# 	# (2) Strings with depth profiles
# 	all_with_depth = enumerate_dyck_with_depth(k, m, max_pairs=max_len // 2)
# 	selected_with_depth = filter_by_lengths_with_depth(all_with_depth, target_lengths)
# 	print("\nStrings with depth:")
# 	for s, depth in selected_with_depth:
# 		print(f"{s}\t{depth}")
from typing import List, Tuple, Set, Dict

# Clean bracket pairs (removed duplicates)
BRACKET_PAIRS = [
    ("(", ")"), ("[", "]"), ("{", "}"), ("<", ">"),
    ("A", "a"), ("B", "b"), ("C", "c"), ("D", "d"),
    ("E", "e"), ("F", "f"),
]

def get_pairs(k: int) -> List[Tuple[str, str]]:
    assert 1 <= k <= len(BRACKET_PAIRS)
    return BRACKET_PAIRS[:k]

def _build_symbol_maps(pairs: List[Tuple[str, str]]) -> Tuple[Dict[str, Tuple[int, str]], Dict[str, Tuple[int, str]]]:
    """
    Returns:
      open_map[c]  = (type_id, 'open')
      close_map[c] = (type_id, 'close')
    """
    open_map, close_map = {}, {}
    for i, (op, cl) in enumerate(pairs):
        open_map[op]  = (i, 'open')
        close_map[cl] = (i, 'close')
    return open_map, close_map

def enumerate_dyck_with_depth(k: int, m: int, max_pairs: int) -> List[Tuple[str, List[int]]]:
    """
    Your original function (shortened here)—kept identical in interface.
    Returns (string, depth_profile), where depth_profile[t] = stack height AFTER consuming symbol t.
    """
    pairs = get_pairs(k)
    results: List[Tuple[str, List[int]]] = []

    def backtrack(current: List[str], depths: List[int], stack: List[int], opened: int, closed: int):
        if opened == closed and opened > 0:
            results.append(("".join(current), depths.copy()))
        if opened >= max_pairs:
            return
        # open
        if len(stack) < m:
            for t in range(k):
                current.append(pairs[t][0])
                stack.append(t)
                depths.append(len(stack))
                backtrack(current, depths, stack, opened + 1, closed)
                depths.pop(); stack.pop(); current.pop()
        # close
        if stack:
            t = stack.pop()
            current.append(pairs[t][1])
            depths.append(len(stack))
            backtrack(current, depths, stack, opened, closed + 1)
            depths.pop(); current.pop(); stack.append(t)

    backtrack([], [], [], 0, 0)
    return results

def filter_by_lengths_strings(strings: List[str], lengths: List[int]) -> List[str]:
    return [s for s in strings if len(s) in lengths]

def filter_by_lengths_with_depth(items: List[Tuple[str, List[int]]], lengths: List[int]) -> List[Tuple[str, List[int]]]:
    return [(s, d) for (s, d) in items if len(s) in lengths]

# ---------- NEW: make (inputs, outputs) token pairs for the CognitiveDiagram ----------

def dyck_samples_for_cognitive_diagram(
    k: int,
    m: int,
    max_pairs: int,
    lengths: List[int] = None,
    *,
    output_mode: str = "depth",     # "depth" | "action" | "top" | "ok"
    token_style: str = "raw",       # "raw" | "typed"
) -> Tuple[List[Tuple[Tuple[str, ...], Tuple[str, ...]]], Set[str], Set[str]]:
    """
    Returns:
      F: list of ((x1,...,xT), (y1,...,yT)) pairs
      X: set of input tokens (strings)
      Y: set of output tokens (strings)
    Each pair has len(inputs) == len(outputs).
    """
    assert output_mode in {"depth", "action", "top", "ok"}
    assert token_style in {"raw", "typed"}

    pairs = get_pairs(k)
    open_map, close_map = _build_symbol_maps(pairs)

    # 1) Get all strings with depth profiles
    all_items = enumerate_dyck_with_depth(k, m, max_pairs)
    if lengths is not None:
        all_items = filter_by_lengths_with_depth(all_items, lengths)

    F: List[Tuple[Tuple[str, ...], Tuple[str, ...]]] = []
    X: Set[str] = set()
    Y: Set[str] = set()

    def token_of(ch: str) -> str:
        if token_style == "raw":
            return ch
        # typed tokens
        if ch in open_map:
            i, _ = open_map[ch]
            return f"OPEN_{i}"
        elif ch in close_map:
            i, _ = close_map[ch]
            return f"CLOSE_{i}"
        else:
            # fallback
            return ch

    for s, depths in all_items:
        # inputs: tuple of tokens
        inp_tokens = tuple(token_of(ch) for ch in s)

        # outputs: choose per-step tokenization
        if output_mode == "depth":
            # Use depth AFTER consuming symbol t; prefix with 'H'
            out_tokens = tuple(f"H{h}" for h in depths)

        elif output_mode == "action":
            # Emit P<i> for opener type i, Q<i> for closer type i
            outs = []
            for ch in s:
                if ch in open_map:
                    i, _ = open_map[ch]
                    outs.append(f"P{i}")
                else:
                    i, _ = close_map[ch]
                    outs.append(f"Q{i}")
            out_tokens = tuple(outs)

        elif output_mode == "top":
            # Emit top-of-stack type after each step: T<i>, or T-1 if empty
            # We reconstruct the stack by reading s with the inferred types
            stack = []
            outs = []
            for ch in s:
                if ch in open_map:
                    i, _ = open_map[ch]
                    stack.append(i)
                else:
                    # closer
                    if stack:
                        stack.pop()
                    # if malformed, your enumerator never generates this
                if stack:
                    outs.append(f"T{stack[-1]}")
                else:
                    outs.append("T-1")
            out_tokens = tuple(outs)

            # Note: T-1 corresponds to empty stack; you may want to map it to "H0" instead
            # if you prefer depths. Keeping 'top' semantics distinct here.

        else:  # "ok"
            # All enumerated strings are legal prefixes of Dyck(k,m), so we mark OK each step
            out_tokens = tuple("OK" for _ in s)

        # Collect vocab
        X.update(inp_tokens)
        Y.update(out_tokens)

        # Append to dataset
        F.append((inp_tokens, out_tokens))

    return F, X, Y
