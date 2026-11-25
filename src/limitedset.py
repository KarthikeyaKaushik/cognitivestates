from itertools import product

class LimitedSet:
    def __init__(self):
        self.contents = set()
        self.error = False

    def add(self, elem):
        if self.error:
            return "E"
        if elem not in ("x", "y", "z"):
            self.error = True
            return "E"
        if elem in self.contents:
            # Already present, do nothing (classic set behavior)
            return "n"
        self.contents.add(elem)
        return "n"

    def remove(self, elem):
        if self.error:
            return "E"
        if elem not in ("x", "y", "z"):
            self.error = True
            return "E"
        if elem in self.contents:
            self.contents.remove(elem)
            return elem
        else:
            # Do not transition to error state, just return "e"
            return "e"

    def step(self, op):
        if op.startswith("add("):
            elem = op[4:-1]
            return self.add(elem)
        elif op.startswith("remove("):
            elem = op[7:-1]
            return self.remove(elem)
        else:
            self.error = True
            return "E"

def generate_sequences(L):
    ops = ["add(x)", "add(y)", "add(z)", "remove(x)", "remove(y)", "remove(z)"]
    all_sequences = []
    for seq in product(ops, repeat=L):
        s = LimitedSet()
        outputs = []
        for op in seq:
            out = s.step(op)
            outputs.append(out)
        all_sequences.append((seq, tuple(outputs)))
    return all_sequences

# Example usage:
if __name__ == "__main__":
    L = 5
    pairs = generate_sequences(L)
    print(len(pairs))
    # for inp, out in pairs:
    #     print("Input:", inp, "Output:", out)