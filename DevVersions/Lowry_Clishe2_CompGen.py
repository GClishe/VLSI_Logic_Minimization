#Bench      1, 0.830    2, 0.291    3, 0.543    4, 2.060    5, 3.721    6, 0.088    7, 3.858    8, 5.889    9, 2.177    10, 288.6    11, 105.8
#Cubes      5609        0           0           58          24313       0           711642      1426983     0           89375        0
#Notes      

"""
def to_bitmask(cube):
        mask = 0
        for i, val in enumerate(cube):
            if val == 1:   # Logic 0
                mask |= (1 << (2 * i))
            elif val == 2: # Logic 1
                mask |= (2 << (2 * i))
            elif val == 3: # Don't Care
                mask |= (3 << (2 * i))
        return mask
"""
from Lowry_Clishe_TautCheck import is_tautology, parse_espresso, Tracker
import time
import sys
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

def complement(cover):
    """
    Main Unate Recursive Paradigm complementation function.
    """
    num_vars = cover.shape[1]

    # Termination 1: Empty cover -> Universal cube (all Don't Cares)
    if len(cover) == 0:
        return np.array([np.full(num_vars, 3)])

    # Termination 3: Single cube -> De Morgan's
    if len(cover) == 1:
        return complement_single_cube(cover[0])

    dummy_vars = list(range(num_vars))
    if is_tautology(cover, dummy_vars, 0, Tracker())[0]:
        return np.empty((0, num_vars), dtype=np.uint8)
    
    # Recursive Step: Select most binate variable and split
    var_idx = binate_select(cover)

    # Cofactor against x=1 (PCN 2) and x=0 (PCN 1)
    C_1 = cofactor(cover, var_idx, 2)
    C_0 = cofactor(cover, var_idx, 1)

    # Recurse down the tree
    comp_1 = complement(C_1)
    comp_0 = complement(C_0)

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


def export_espresso(cover, ilb, ob, filepath):
    with open(filepath, 'w') as f:
        f.write(f".i {len(ilb) if ilb else cover.shape[1]}\n")
        f.write(f".o {len(ob) if ob else 1}\n")
        f.write(f".p {len(cover)}\n")
        if ilb: f.write(f".ilb {' '.join(ilb)}\n")
        if ob: f.write(f".ob {' '.join(ob)}\n")
        for row in cover:
            cube_str = "".join(['0' if v == 1 else '1' if v == 2 else '-' for v in row])
            f.write(f"{cube_str} 1\n")
        f.write(".e\n")

if __name__ == "__main__":
    start_time = time.perf_counter()
    input_file = sys.argv[1]
    cover, ilb, ob = parse_espresso(input_file)
    print(f"\nParsed {input_file} containing {len(cover)} cubes in {(time.perf_counter() - start_time):.3f} sec")

    start_time = time.perf_counter()
    comp_cover = complement(cover)
    comp_time = time.perf_counter() - start_time
    
    print(f"\n--- Complementation Results ---")
    print(f"Original Cubes: {len(cover)}")
    print(f"Complement Cubes (SCC Minimal): {len(comp_cover)}")
    print(f"Time Taken: {comp_time:.3f} sec")

    output_file = f"Lowry_Clishe_Complement_Covers/{input_file.split('/')[-1]}_compl"
    export_espresso(comp_cover, ilb, ob, output_file)
    print(f"Exported complement to: {output_file}\n")