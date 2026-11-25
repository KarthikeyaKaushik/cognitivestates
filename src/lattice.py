import copy
from itertools import combinations
import numpy as np
from typing import Dict, Iterable, List, Tuple, Set
from src.utils import merge_sets, is_identical
import itertools
import random

class UniqueNumbering:
    def __init__(self):
        self.unique_numbers = set()
        
    def get_next(self):
        self.unique_numbers.add(len(self.unique_numbers))
        return len(self.unique_numbers)  # 1 indexed

class Coloring:
    def __init__(self, coloring, transitions, cognitive_diagram):
        # Ensure the coloring has unique values
        coloring_values = [y for x in coloring for y in x]
        assert len(coloring_values) == len(set(coloring_values)), "Coloring values must be unique"
        self.color2state = {color: set(x) for color, x in enumerate(coloring)}
        self.state2color = {state: color for color in self.color2state for state in self.color2state[color]}
        self.transitions = transitions
        self.cognitive_diagram = cognitive_diagram
        
    def output_probs(self, inputs, outputs):
        # check if attribute exists
        if not hasattr(self, 'statetrans'):
            self.get_statetrans()
        # sequence of inputs
        current_probs = np.zeros(len(self.state2color))
        current_probs[self.state2color[0]] = 1.0  # start wherever state 0 is, forward simulation
        state_vecs = []
        nnl = 0
        for idp, inp in enumerate(inputs):
            # first compute output probs for that particular output
            current_op = outputs[idp]
            new_op_probs = 0.0
            for state in self.state2color:
                current_prob_s = current_probs[state]
                available_op = self.cognitive_diagram.outputtrans[(state, inp)]
                if available_op == current_op:
                    new_op_probs += current_prob_s

            assert new_op_probs <= 1 + 1e-10, "Output probability exceeds 1"
            nnl += -np.log2(new_op_probs) 
            new_state_probs = np.zeros(len(self.state2color))
            for state in self.state2color:
                current_prob_s = current_probs[state]
                new_state_probs += self.statetrans[(state, inp)] * current_prob_s
            current_probs = new_state_probs / np.sum(new_state_probs)
            state_vecs.append(current_probs)
        return state_vecs, nnl

    # def output_probs(self, inputs, outputs):
    #     # sequence of inputs
    #     # start with some c0 = [1, 0, ... 0]
    #     # at each point, apply transition, normalize color probs
    #     # p(y0 | x[0]) = p(c0) * p(y0 | c0, x0)
    #     # normalize to get p(c1)
    #     # repeat 
    #     current_probs = [0 for _ in self.color2state]
    #     current_probs[self.state2color[0]] = 1.0  # start wherever state 0 is, forward simulation
    #     nnl = 0
    #     for idp, inp in enumerate(inputs):
    #         # simulate outputs for all possible colors
    #         # get new color probs, and normalize
    #         new_color_probs = [0 for _ in self.color2state]
    #         new_ops = {y:0 for y in self.cognitive_diagram.Y}
    #         for color in self.color2state:
    #             current_prob_c = current_probs[color]
    #             for color_dest in self.color2state:
    #                 new_color_probs[color_dest] += current_prob_c * self.colortrans[(color, inp)][color_dest]
    #             for y in self.cognitive_diagram.Y:
    #                 new_ops[y] += current_prob_c * self.outputtrans[(color, inp)][y]
    #         # now normalize new_ops to get p(y | x)
    #         total_op = sum(new_ops.values())
    #         new_ops = {k:v/total_op for k,v in new_ops.items()}
    #         # get negative log likelihood contribution
    #         new_nnl = -np.log2(new_ops[outputs[idp]])  # add small value to avoid log(0)
    #         nnl += new_nnl
    #         current_probs = [x/sum(new_color_probs) for x in new_color_probs]
    #     return nnl

    def get_colortrans(self):
        # on some input, how does the state transition between colors, with what probability? 
        self.colortrans: Dict[Tuple[int, str], List[float]] = {}
        parent_cd = self.cognitive_diagram
        for color in self.color2state:
            states = self.color2state[color]
            for input_token in parent_cd.X:
                to_colors = {state:self.state2color[parent_cd.statetrans[(state, input_token)]] for state in states}
                prob_colors = [0 for _ in self.color2state] # count and normalize later
                for state in states:
                    prob_colors[to_colors[state]] += 1
                total = sum(prob_colors)
                prob_colors = [x/total for x in prob_colors]
                self.colortrans[(color, input_token)] = prob_colors    

    def get_statetrans(self):
        # on some input, how does the state transition between states, with what probability? 
        self.statetrans: Dict[Tuple[int, str], List[float]] = {}
        parent_cd = self.cognitive_diagram
        for state in self.state2color:
            for input_token in parent_cd.X:
                prob_states = [0 for _ in self.state2color]
                to_color = self.state2color[parent_cd.statetrans[(state, input_token)]]
                to_states = [s for s in self.color2state[to_color]]
                prob = 1/len(to_states)
                for s in to_states:
                    prob_states[s] = prob
                self.statetrans[(state, input_token)] = np.array(prob_states)

    def get_outputtrans(self):
        self.outputtrans: Dict[Tuple[int, str], str] = {}
        parent_cd = self.cognitive_diagram
        for color in self.color2state:
            states = self.color2state[color]
            for input_token in parent_cd.X:
                output_counts = {y:0 for y in parent_cd.Y}                
                to_outputs = {state:parent_cd.outputtrans[(state, input_token)] for state in states}
                for v in to_outputs.values():
                    output_counts[v] += 1
                total = sum(output_counts.values())
                prob_outputs = {k:v/total for k,v in output_counts.items()}
                self.outputtrans[color, input_token] = prob_outputs        


    def get_sets(self):  # Simply a list of sets
        return list(self.color2state.values())

    def canonicalize(self):
        label_mapping = {}
        next_label = 0
        d = self.state2color
        for k in sorted(d):
            val = d[k]
            if val not in label_mapping:
                label_mapping[val] = next_label
                next_label += 1
        new_color2state = {}
        for color, states in self.color2state.items():
            new_color2state[label_mapping[color]] = states
        self.color2state = new_color2state
        self.state2color = {state: color for color in new_color2state for state in new_color2state[color]}


class ConsistentColoring(Coloring):
    def __init__(self, coloring, transitions, cognitive_diagram):
        super().__init__(coloring, transitions, cognitive_diagram)  # Initialize properties from the parent class
        self.id = None

    def pairwise_matrix(self):
        # create a pairwise matrix of states, with a 0 indicating that they are different, and 1 if they have the same color
        n_states = len(self.state2color)
        matrix = np.zeros((n_states, n_states), dtype=int)
        for i in range(n_states):
            for j in range(n_states):
                if (self.state2color[i] == self.state2color[j]):
                    matrix[i, j] = 0.0
                else:
                    matrix[i, j] = 1.0
        np.fill_diagonal(matrix, 1.0)  # fill diagonal with 0s
        self.pairwise_matrix = matrix
        return


    def get_sets(self): # simply a list of sets
        return list(self.color2state.values())

    def canonicalize(self):
        label_mapping = {}
        pattern = ''
        next_label = 0
        d = self.state2color
        for k in sorted(d):
            val = d[k]
            if val not in label_mapping:
                label_mapping[val] = next_label
                next_label += 1
        new_color2state = {}
        for color, states in self.color2state.items():
            new_color2state[label_mapping[color]] = states
        self.color2state = new_color2state
        self.state2color = {state: color for color in new_color2state for state in new_color2state[color]}
        return
    

    def merge_colors(self, mismatches, color2state):
        new_colors = merge_sets(mismatches)
        new_color2state = {}
        for idc, _ in enumerate(new_colors):
            new_color2state[idc] = set()
            for col in new_colors[idc]:
                new_color2state[idc] = new_color2state[idc].union(color2state[col])
        new_state2color = {state: color for color in new_color2state for state in new_color2state[color]}
        return new_color2state, new_state2color

    def id_mismatches(self, transitions, color2state, state2color):
        mismatches = []
        for i in range(transitions.shape[1]): # for each input
            for color in color2state:
                to_transitions = set()
                for state in color2state[color]:
                    to_transitions.add(transitions[state, i])
                to_colors = set([state2color[x] for x in to_transitions])
                mismatches.append(to_colors)
        unique_mismatches = []
        for x in mismatches:
            if x not in unique_mismatches:
                unique_mismatches.append(x)
        return unique_mismatches


    def make_consistent(self, depth=0):
        assert depth < 20, "Recursion depth exceeded"
        if self.is_consistent():
            self.canonicalize()
            # now call again self.get_outputtrans and self.get_colortrans to update
            self.get_colortrans()
            self.get_outputtrans()
            self.get_statetrans()
            return
        else:
            transitions, color2state, state2color = self.transitions, self.color2state, self.state2color
            mismatches = self.id_mismatches(transitions, color2state, state2color)
            color2state, state2color = self.merge_colors(mismatches, color2state)
            self.color2state = color2state
            self.state2color = state2color
            self.make_consistent(depth + 1)

    def is_consistent(self):
        transitions = self.transitions # for readability
        for i in range(transitions.shape[1]): # for each input
            for color in self.color2state:
                to_transitions = set()
                for state in self.color2state[color]:
                    to_transitions.add(transitions[state, i])
                to_colors = set([self.state2color[x] for x in to_transitions])
                if len(to_colors) > 1:
                    return False
        return True



class Lattice:

    def __init__(self, cognitive_diagram):
        coloring = [[k] for k in cognitive_diagram.labels.keys()]
        self.cd = cognitive_diagram
        self.root = ConsistentColoring(coloring=coloring, transitions=cognitive_diagram.transitions, cognitive_diagram=cognitive_diagram)
        self.nodes = []
        self.relationships = {}
        self.build_lattice()
        self.reformat()

    def get_pairwise_matrices(self):
        for cc in self.nodes:
            cc.pairwise_matrix()
        return

    def get_tikzfig(self):
        text = r"\begin{tikzpicture} \ "
        row_height = 1
        col_width = 1
        n_cols = max([len(x) for x in self.levels])
        node_pos = {}
        for idl, level in enumerate(self.levels):
            # generate all positions
            positions = [x*col_width for x in range(len(level))]
            shift = (n_cols - len(level))/2
            positions = [x + shift for x in positions]
            for idc, cc in enumerate(level):
                node_text = f'\\node ({cc.id}) at ({positions[idc]},{-idl*row_height})' + '{}' + '; \n'
                text += node_text
                node_pos[cc.id] = (positions[idc], -idl*row_height)
        text += '\\begin{pgfonlayer}{background}'
        nodes = [x.id for x in self.nodes]
        for node in sorted(nodes):
            nontransitive_nodes = self.get_nontransitive_nodes(node)
            for nt_n in nontransitive_nodes:
                text += f'\\draw[->, gray!50] ({node}) to ({nt_n});'
        for node in sorted(nodes):
            label_name = str(node) if node != len(self.nodes) else r'$\phi$'
            node_text = f'\\node[circle,draw,minimum size=16pt,inner sep=1pt, fill=white] ({node}) at ({node_pos[node][0]},{node_pos[node][1]})' + ' {' + label_name + '}' + '; \n'
            text += node_text
        text += '\\end{pgfonlayer}'
        text += '\\end{tikzpicture} '
        return text

    def get_latextable(self):
        # generate table with colorings 
        # TODO : 
        pass
            

    def get_nontransitive_nodes(self, id):
        connections = copy.deepcopy(self.relationships[id])
        for x in self.relationships[id]:
            for y in self.relationships[x]:
                if y in connections:
                    connections.remove(y)
        return connections


    def add_node(self, cc):
        if cc not in self.nodes:
            self.nodes.append(cc) 

    def add_relation(self, x, y):
        if x not in self.relationships:
            self.relationships[x] = set([y])
        else:
            self.relationships[x].add(y)


    def get_lub(self, cc1, cc2):
        colors1 = cc1.color2state
        colors2 = cc2.color2state
        new_colors = []

        for c1 in colors1:
            for c2 in colors2:
                if len(colors1[c1].intersection(colors2[c2])) > 0:
                    color_union = colors1[c1].union(colors2[c2])
                    if color_union not in new_colors:
                        new_colors.append(color_union)
        new_colors = merge_sets(new_colors)
        return new_colors
        

    def combine_algebraically(self, ccs):
        old_ccs = []
        new_ccs = [x for x in ccs]
        while len(old_ccs) != len(new_ccs):
            old_ccs = copy.deepcopy(new_ccs)
            for idc1, cc1 in enumerate(old_ccs[:-1]):
                    for idc2, cc2 in enumerate(old_ccs[idc1+1:]):
                        lub = self.get_lub(cc1, cc2)
                        if not any([is_identical(lub, x.get_sets()) for x in new_ccs]):
                            new_cc = ConsistentColoring(lub, old_ccs[0].transitions, self.cd)
                            new_cc.make_consistent()
                            new_ccs.append(new_cc)
        return new_ccs


    def coarser_than(self, cc1, cc2):
        # TODO : Make only one version, keep in utils
        if len(cc1.color2state) > len(cc2.color2state):
            return False
        else: 
            for y in cc2.get_sets():
                flag = False
                for x in cc1.get_sets():
                    if y.issubset(x):
                        flag = True
                        break
                if not flag:
                    return False
            return True
                                    

    def build_lattice(self):
        ccs = self.generate_CCs()
        for cc in ccs:
            self.add_node(cc)

        for cc1 in ccs:
            self.relationships[cc1] = set()
            for cc2 in ccs:
                if cc1 != cc2 and self.coarser_than(cc1, cc2):
                    self.add_relation(cc1, cc2)
        return

    def get_independent_level(self, ccs):
        indie_level = []
        for cc1 in ccs:
            is_coarser = []
            for cc2 in ccs:
                if cc1 == cc2:
                    continue
                if self.coarser_than(cc2, cc1):
                    is_coarser.append(False)
                else:
                    is_coarser.append(True)
            if all(is_coarser):
                indie_level.append(cc1)
        return indie_level


    def reformat(self):
        levels = []
        remaining = [x for x in self.nodes]
        while len(remaining) > 0:
            indie_level = self.get_independent_level(remaining)
            levels.append(indie_level)
            for cc in indie_level:
                remaining.remove(cc)
        numbering = UniqueNumbering()
        for level in levels:
            for cc in level:
                cc.id = numbering.get_next()
        reformatted_relationships = {}
        for key, val in self.relationships.items():
            reformatted_relationships[key.id] = set([x.id for x in val])
        self.relationships = reformatted_relationships
        self.levels = levels
        return 



    def generate_CCs(self):
        pairs = list(combinations([x for x in self.root.color2state.keys()], 2)) 
        canonicals = []
        ccs = [self.root]
        for pair in pairs:
            colors = [[pair[0], pair[1]]] + [[x] for x in self.root.color2state.keys() if x not in pair]
            cc = ConsistentColoring(colors, self.root.transitions, self.cd)
            cc.make_consistent()
            if len(cc.color2state) > 1 and (cc.color2state not in canonicals):
                canonicals.append(cc.color2state)
                ccs.append(cc)
        ccs = self.combine_algebraically(ccs)
        return ccs

def random_partition(collection, s, n_samples=1, seed=1):
    """
    Yield n_samples random partitions of 'collection' into 's' non-empty subsets.
    Each partition is a list of sets.
    """
    random.seed(seed)
    collection = list(collection)
    for _ in range(n_samples):
        random.shuffle(collection)
        # Randomly assign each element to one of s bins
        bins = [[] for _ in range(s)]
        for elem in collection:
            idx = random.randint(0, s-1)
            bins[idx].append(elem)
        # Ensure all bins are non-empty; if not, try again
        if all(len(b) > 0 for b in bins):
            yield [set(b) for b in bins]


def all_partitions(collection, s):
    """
    Generate all possible ways to partition 'collection' into 's' non-empty subsets.
    Each partition is a list of sets.
    """
    collection = list(collection)
    if s < 1 or s > len(collection):
        return
    if s == 1:
        yield [set(collection)]
        return
    if len(collection) == 0:
        return
    first = collection[0]
    rest = collection[1:]
    for smaller in all_partitions(rest, s-1):
        yield [set([first])] + smaller
    for smaller in all_partitions(rest, s):
        for j in range(len(smaller)):
            new_subset = smaller[j] | set([first])
            yield smaller[:j] + [new_subset] + smaller[j+1:]

def unique_partitions(collection, s):
    seen = set()
    for partition in all_partitions(collection, s):
        # Sort subsets by sorted tuple, then convert to tuple of frozensets
        norm = tuple(sorted([frozenset(sub) for sub in partition], key=lambda x: sorted(x)))
        if norm not in seen:
            seen.add(norm)
            yield [set(sub) for sub in norm]
