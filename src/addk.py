from typing import List, Tuple
import itertools

def generate_kdigit_add_pairs(k: int, num_digits: int) -> List[Tuple[str, str]]:
    """
    Generate input-output pairs for k-digit addition (LSB-first, interleaved).
    Input: x0, y0, x1, y1, ..., xn, yn (as a string of digits)
    Output: z0, _, z1, _, ..., zn, _ (as a string), where zi = xi + yi + carry (mod k), _ is a placeholder.
    """
    results = []
    digits = [str(d) for d in range(10)]
    # Each input is length 2*num_digits (x0,y0,x1,y1,...)
    for inp_tuple in itertools.product(digits, repeat=2*num_digits):
        inp = "".join(inp_tuple)
        output = []
        carry = 0
        for i in range(num_digits):
            xi = int(inp[2*i])
            yi = int(inp[2*i+1])
            s = xi + yi + carry
            output.append("_")  # Add underscore at odd positions            
            output.append(str(s % k))
            carry = s // k
        results.append((inp, "".join(output)))
    return results