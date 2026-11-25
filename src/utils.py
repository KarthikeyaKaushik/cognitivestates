import copy


def merge_sets(sets):
    merged = True
    new_sets = copy.deepcopy(sets)
    while merged:
        merged = False
        result = []
        while new_sets:
            current = new_sets.pop(0)
            i = 0
            # Merge with any set that shares an element with 'current'
            while i < len(new_sets):
                if current & new_sets[i]:
                    current |= new_sets.pop(i)
                    merged = True
                else:
                    i += 1
            result.append(current)
        new_sets = result
    return new_sets


def is_identical(s1, s2):
    if not isinstance(s1, (list, set)) or not isinstance(s2, (list, set)):
        raise TypeError("Both arguments must be either lists or sets.")
    return  all([True if x in s2 else False for x in s1]) and all([True if x in s1 else False for x in s2])

def coarser_than(cc1, cc2):
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

def get_l_j(pi_j, pi_j_star):
    # largest number of blocks of pi_j contained in a block of pi_j_star
    l_j = 0
    block_id = 0
    for bid, part_star in enumerate(pi_j_star):
        contained = 0
        for part_j in pi_j:
            if part_j.issubset(part_star):
                contained += 1
        if contained > l_j:
            l_j = contained
            block_id = bid
    assert l_j > 0
    return l_j, block_id



def get_glb(sets):
    assert all(type(y) == set for x in sets for y in x)
    if len(sets) == 1:
        return sets.pop()
    if len(sets) == 2:
        set1, set2 = sets[0], sets[1]
        glb = []
        for s1 in set1:
                for s2 in set2:
                    intersection = s1 & s2  # Compute the intersection of the two sets
                    if intersection:  # Only add non-empty intersections
                        glb.append(intersection)    
        # Merge overlapping sets in the GLB to ensure consistency
        glb = merge_sets(glb)
        return glb
    else:
        glb = sets[0]
        for i in range(1, len(sets)):
            glb = get_glb([glb, sets[i]])
        return glb


