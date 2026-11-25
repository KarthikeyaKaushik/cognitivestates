import random 
random.seed(42)

def anbn(num_sequences, max_length):
    base_cases = ['ab', 'aabb', 'aaabbb']#, 'aaaabbbb', 'aaaaabbbbb']
    sequences = []
    while len(sequences) < num_sequences:
        curr_seq = random.choice(base_cases) + '.'
        while len(curr_seq) < max_length:
            to_continue = random.random() < 0.5
            if not to_continue:
                break
            else:
                curr_seq += random.choice(base_cases) + '.'
        if curr_seq not in sequences:
            sequences.append(tuple([x for x in curr_seq]))
    return sequences

def get_outputs(sequences):
    outputs = []
    for seq in sequences:
        count_a = 0
        count_b = 0
        seq_outputs = []
        for char in seq:
            if char == '.':
                seq_outputs.append(-1)
            else:
                if char == 'a':
                    count_a += 1
                elif char == 'b':
                    count_b += 1
                seq_outputs.append(count_a - count_b)
        outputs.append(tuple(seq_outputs))
    return outputs

def generate_sequences(num_sequences, max_length):
    sequences = anbn(num_sequences, max_length)
    outputs = get_outputs(sequences)
    dataset = list(zip(sequences, outputs))
    return dataset


if __name__ == "__main__":
    num_sequences = 1000
    max_length = 20
    dataset = generate_sequences(num_sequences, max_length)
    # with open('anbn_sequences.pkl', 'wb') as f:
    #     pickle.dump(sequences, f)