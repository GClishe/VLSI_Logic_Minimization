#Make me an espresso compliment generator in python using the Unate Recursive Complimentation 
#method and bitmaps for vectorization. I already have a tautology checker function if you need one

import numpy as np

# Positional Cube Notation (PCN) Bitmaps:
# 1 = 01 (False), 2 = 10 (True), 3 = 11 (Don't Care)

def complement_single_cube(cube):
    """
    Applies De Morgan's Law to a single cube.
    x*y -> x' + y'. A single cube complement returns a cover of multiple cubes.
    """
    num_vars = len(cube)
    comp_cover = []
    
    for i in range(num_vars):
        if cube[i] == 1:   # Literal is 0, complement is 1 (PCN: 2)
            new_cube = np.full(num_vars, 3)
            new_cube[i] = 2
            comp_cover.append(new_cube)
        elif cube[i] == 2: # Literal is 1, complement is 0 (PCN: 1)
            new_cube = np.full(num_vars, 3)
            new_cube[i] = 1
            comp_cover.append(new_cube)
            
    # Return as a 2D numpy array (cover)
    if comp_cover:
        return np.array(comp_cover, dtype=int)
    return np.empty((0, num_vars), dtype=int)

def binate_select(cover):
    """
    Vectorized selection of the most binate splitting variable.
    Finds the variable that appears most frequently in both true and complement forms.
    """
    ones = np.sum(cover == 1, axis=0)
    twos = np.sum(cover == 2, axis=0)
    
    # Heuristic: maximize the occurrence of literals
    score = ones + twos
    
    # Disqualify variables that are purely 'Don't Cares' (all 3s)
    score[np.all(cover == 3, axis=0)] = -1 
    
    return np.argmax(score)

def cofactor(cover, var_idx, val_pcn):
    """
    Vectorized Shannon Cofactor. 
    val_pcn is 2 (for x=1) or 1 (for x=0).
    """
    # Bitwise AND to check if cubes survive the cofactor intersection
    mask = (cover[:, var_idx] & val_pcn) != 0
    cof = cover[mask].copy()
    
    # For surviving cubes, the splitting variable becomes a 'Don't Care' (3)
    if len(cof) > 0:
        cof[:, var_idx] = 3
        
    return cof

def urp_complement(cover):
    """
    Main Unate Recursive Paradigm complementation function.
    """
    num_vars = cover.shape[1] if len(cover) > 0 else 0

    # Termination 1: Empty cover -> Universal cube (all Don't Cares)
    if len(cover) == 0:
        return np.array([np.full(num_vars, 3)])

    # Termination 2: Universal cube present -> Empty cover
    if np.any(np.all(cover == 3, axis=1)):
        return np.empty((0, num_vars), dtype=int)

    # Termination 3: Single cube -> De Morgan's
    if len(cover) == 1:
        return complement_single_cube(cover[0])

    # [OPTIONAL: Insert your custom tautology checker here]
    # if is_tautology(cover): 
    #     return np.empty((0, num_vars), dtype=int)

    # Recursive Step: Select most binate variable and split
    var_idx = binate_select(cover)

    # Cofactor against x=1 (PCN 2) and x=0 (PCN 1)
    C_1 = cofactor(cover, var_idx, 2)
    C_0 = cofactor(cover, var_idx, 1)

    # Recurse down the tree
    comp_1 = urp_complement(C_1)
    comp_0 = urp_complement(C_0)

    # AND the split variables back into the resulting complementary covers
    if len(comp_1) > 0:
        comp_1[:, var_idx] = 2  # AND with x=1
    if len(comp_0) > 0:
        comp_0[:, var_idx] = 1  # AND with x=0

    # Merge covers back together
    if len(comp_1) == 0:
        return comp_0
    if len(comp_0) == 0:
        return comp_1
        
    return np.vstack((comp_1, comp_0))

# --- Example Usage ---
if __name__ == "__main__":
    # Let's define F(x, y) = x * y
    # Cover has 1 cube: x=1, y=1 (PCN: 2, 2)
    F = np.array([
        [2, 2]
    ])
    
    print("Original Cover F (x AND y):")
    print(F)
    
    F_comp = urp_complement(F)
    
    print("\nComplement Cover F' (x' OR y'):")
    print(F_comp)
    # Expected output: 
    # [[1, 3],  -> x=0, y=-
    #  [3, 1]]  -> x=-, y=0