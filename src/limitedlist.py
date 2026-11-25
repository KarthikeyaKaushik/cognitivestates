from itertools import product

class LimitedList:
    def __init__(self):
        # Three slots, each can be None, "x", or "y"
        self.contents = [None, None, None]

    def write(self, elem, idx):
        # Write "x" or "y" to slot idx (0, 1, or 2)
        if elem not in ("x", "y") or idx not in (0, 1):
            return "n"
        self.contents[idx] = elem
        return "n"

    def read(self, idx):
        # Read from slot idx (0, 1, or 2)
        if idx not in (0, 1):
            return "-"
        val = self.contents[idx]
        return val if val in ("x", "y") else "-"

    def step(self, op):
        # op: "write(x,0)", "write(y,2)", "read(1)", etc.
        if op.startswith("write("):
            inside = op[6:-1]
            try:
                elem, idx = inside.split(",")
                elem = elem.strip()
                idx = int(idx.strip())
            except:
                return "n"
            return self.write(elem, idx)
        elif op.startswith("read("):
            idx = op[5:-1]
            try:
                idx = int(idx.strip())
            except:
                return "-"
            return self.read(idx)
        else:
            return "n"

def generate_sequences(L):
    ops = []
    for elem in ["x", "y"]:
        for idx in range(2):
            ops.append(f"write({elem},{idx})")
    for idx in range(2):
        ops.append(f"read({idx})")
    all_sequences = []
    for seq in product(ops, repeat=L):
        s = LimitedList()
        outputs = []
        for op in seq:
            out = s.step(op)
            outputs.append(out)
        all_sequences.append((seq, tuple(outputs)))
    return all_sequences

# Example usage:
if __name__ == "__main__":
    L = 4
    pairs = generate_sequences(L)
    for inp, out in pairs:
        if len(inp) == 4:  # Print only short sequences for brevity
            print("Input:", inp, "Output:", out)
