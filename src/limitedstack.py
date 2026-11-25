from itertools import product
import random
random.seed(42)


class LimitedStack:
    def __init__(self):
        self.stack = []
        self.error = False

    def push(self, elem):
        if self.error:
            return "e"
        if elem not in ("x", "y"):
            self.error = True
            return "e"
        if len(self.stack) >= 3:
            self.error = True
            return "e"
        self.stack.append(elem)
        return "n"

    def pop(self):
        if self.error:
            return "e"
        if not self.stack:
            return "n"
        return self.stack.pop()

    def step(self, op):
        if op == "pop()":
            return self.pop()
        elif op.startswith("push("):
            elem = op[5:-1]
            return self.push(elem)
        else:
            self.error = True
            return "e"

def random_sequence(L):
    ops = ['push(x)', 'push(y)', 'pop()']
    sequence = [random.choice(ops) for _ in range(L)]
    s = LimitedStack()
    outputs = []
    for op in sequence:
        out = s.step(op)
        outputs.append(out)
    sequence = (tuple(sequence), tuple(outputs))
    return sequence


def generate_sequences(L):
    ops = ["push(x)", "push(y)", "pop()"]
    all_sequences = []
    for seq in product(ops, repeat=L):
        s = LimitedStack()
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
        print("Input:", inp, "Output:", out)