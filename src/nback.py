from itertools import product

def nback_output(seq, n):
    """
    Given a sequence of tokens and n, returns the n-back output sequence.
    '*' for match to n steps back, '-' otherwise.
    """
    out = []
    for i in range(len(seq)):
        if i < n:
            out.append('-')
        else:
            out.append('*' if seq[i] == seq[i-n] else '-')
    return tuple(out)

def generate_nback_sequences(L, n=3, k=2):
    """
    Generates all input/output pairs for n-back task with n, k stimuli, up to length L.
    Returns list of (input_tuple, output_tuple).
    """
    tokens = [chr(ord('a') + i) for i in range(k)]
    all_pairs = []

    for seq in product(tokens, repeat=L):
        out = nback_output(seq, n)
        all_pairs.append((seq, out))
    return all_pairs

# Example usage:
if __name__ == "__main__":
    n, k, L = 2, 2, 4
    pairs = generate_nback_sequences(n, k, L)
    for inp, out in pairs:
        print("Input:", inp, "Output:", out)