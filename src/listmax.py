from itertools import product

def maxlist_output(seq):
    """
    Given a sequence of digits, returns the running maximum at each timestep.
    """
    out = []
    current_max = None
    for s in seq:
        num = int(s)
        if current_max is None or num > current_max:
            current_max = num
        out.append(str(current_max))
    return tuple(out)

def generate_sequences(L):
    tokens = [str(i) for i in range(0, 10)]
    all_pairs = []
    for seq in product(tokens, repeat=L):
        out = maxlist_output(seq)
        all_pairs.append((seq, out))
    return all_pairs

# Example usage:
if __name__ == "__main__":
    L = 4
    pairs = generate_sequences(L)
    for inp, out in pairs:
        print("Input:", inp, "Output:", out)