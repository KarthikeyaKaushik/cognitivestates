# generates data for the "majority" cfg 
# the idea here is that the result of a sequence of 1s and 0s is the majority of the bits seen so far, with ties going to 0.
# e.g 1101 -> 1, 1100 -> 0, 0000 -> 0, etc.

import random 
random.seed(42)
from itertools import product



def sample_bounded_majority_sequence(length, max_states):
    seq = []
    state = 0  # #1s - #0s

    for _ in range(length):
        choices = []

        # bit 1 means state += 1
        if state + 1 <= max_states:
            choices.append(1)

        # bit 0 means state -= 1
        if state - 1 >= -max_states:
            choices.append(0)

        bit = random.choice(choices)
        seq.append(bit)

        state += 1 if bit == 1 else -1

    return tuple(seq)


def majority(num_sequences, max_length, max_states):
    sequences = []

    for _ in range(num_sequences):
        seq = sample_bounded_majority_sequence(max_length, max_states)
        sequences.append(seq)

    return sequences



def generate_sequences(num_sequences, max_length, max_states):
    sequences = majority(num_sequences, max_length, max_states)
    io_pairs = []
    for seq in sequences:
        convenient_representation = [1 if bit == 1 else -1 for bit in seq]
        majority_so_far = [sum(convenient_representation[:i+1]) for i in range(len(seq))]
        majority_so_far = [(x > 0)*1 for x in majority_so_far] # convert to boolean: True if majority of bits so far are 1, else False
        # io_pairs.append((seq, (sum(seq) > (len(seq) / 2)) * 1)) # output is 1 if majority of bits are 1, else 0
        io_pairs.append((seq, tuple(majority_so_far)))
    return io_pairs



if __name__ == "__main__":
    num_sequences = 1000
    max_length = 5
    max_states = 2
    # with open('anbn_sequences.pkl', 'wb') as f:
    #     pickle.dump(sequences, f)