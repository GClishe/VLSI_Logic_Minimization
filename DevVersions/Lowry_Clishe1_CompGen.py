#Bench      1, 3.846    2, 0.291    3, 0.548    4, 2.093    5, 48.22    6, 0.088    7, 1145.    8, 0.000    9, 2.173    10, 0.000    11, 105.8
#Cubes      5443        0           0           58          23357       0           101482      NA          0           NA           0
#Notes      

import sys
import time
import numpy as np
from Lowry_Clishe_TautCheck import is_tautology, Tracker, parse_espresso

def scc_minimal(cover):
    if len(cover) <= 1: return cover
    # Sort descending by number of dashes so larger cubes are processed first
    num_dashes = np.sum(cover == 3, axis=1)
    sorted_cover = cover[np.argsort(-num_dashes)]
    
    kept = []
    for row in sorted_cover:
        if not kept:
            kept.append(row)
            continue
        kept_arr = np.array(kept)
        # Check if the row is contained by any already kept maximal cube
        # A kept_row contains row if for all cols: kept_row == 3 OR kept_row == row
        is_contained = np.any(np.all((kept_arr == 3) | (kept_arr == row), axis=1))
        if not is_contained:
            kept.append(row)
    return np.array(kept, dtype=np.uint8)

def complement_single_cube(cube):
    n = len(cube)
    res = []
    for i in range(n):
        if cube[i] == 1:
            new_cube = np.full(n, 3, dtype=np.uint8)
            new_cube[i] = 2
            res.append(new_cube)
        elif cube[i] == 2:
            new_cube = np.full(n, 3, dtype=np.uint8)
            new_cube[i] = 1
            res.append(new_cube)
    return np.array(res, dtype=np.uint8)

def complement(F):
    n_vars = F.shape[1]
    
    # Base Cases
    if F.size == 0: 
        return np.full((1, n_vars), 3, dtype=np.uint8)
    if any((F == 3).all(axis=1)): 
        return np.empty((0, n_vars), dtype=np.uint8)
        
    # Tautology Pruning 
    dummy_vars = list(range(n_vars))
    if is_tautology(F, dummy_vars, 0, Tracker())[0]:
        return np.empty((0, n_vars), dtype=np.uint8)
        
    if len(F) == 1:
        return complement_single_cube(F[0])

    # Common Cube Extraction 
    has_0 = (F == 1).all(axis=0)
    has_1 = (F == 2).all(axis=0)
    C = np.full(n_vars, 3, dtype=np.uint8)
    C[has_0] = 1
    C[has_1] = 2
    
    if np.any(C != 3): 
        F_c = F.copy()
        for i in range(n_vars):
            if C[i] != 3: F_c[:, i] = 3
        
        comp_C = complement_single_cube(C)
        comp_Fc = complement(F_c)
        res = np.vstack((comp_C, comp_Fc)) if len(comp_Fc) > 0 else comp_C
        return scc_minimal(res)

    # Splitting Variable Selection
    count_0 = np.sum(F == 1, axis=0)
    count_1 = np.sum(F == 2, axis=0)
    binate_mask = (count_0 > 0) & (count_1 > 0)
    scores = count_0 + count_1
    
    # Unate Complementation Optimization 
    if not np.any(binate_mask):
        split_var = np.argmax(scores) 
        is_pos = (count_1[split_var] > 0) 

        if is_pos:
            F_p = F[F[:, split_var] != 1].copy(); F_p[:, split_var] = 3
            F_n = F[F[:, split_var] == 3].copy(); F_n[:, split_var] = 3
            comp_P, comp_N = complement(F_p), complement(F_n)
            if len(comp_N) > 0: comp_N[:, split_var] = 1
            res = np.vstack((comp_P, comp_N)) if len(comp_P)>0 and len(comp_N)>0 else (comp_P if len(comp_P)>0 else comp_N)
            return scc_minimal(res)
        else:
            F_p = F[F[:, split_var] == 3].copy(); F_p[:, split_var] = 3
            F_n = F[F[:, split_var] != 2].copy(); F_n[:, split_var] = 3
            comp_P, comp_N = complement(F_p), complement(F_n)
            if len(comp_P) > 0: comp_P[:, split_var] = 2
            res = np.vstack((comp_P, comp_N)) if len(comp_P)>0 and len(comp_N)>0 else (comp_P if len(comp_P)>0 else comp_N)
            return scc_minimal(res)

    # Standard Binate Split
    binate_scores = np.where(binate_mask, scores, -1)
    split_var = np.argmax(binate_scores)
    
    F_p = F[F[:, split_var] != 1].copy(); F_p[:, split_var] = 3
    F_n = F[F[:, split_var] != 2].copy(); F_n[:, split_var] = 3
    
    comp_P = complement(F_p)
    comp_N = complement(F_n)
    
    if len(comp_P) > 0: comp_P[:, split_var] = 2
    if len(comp_N) > 0: comp_N[:, split_var] = 1
    
    res = np.vstack((comp_P, comp_N)) if len(comp_P)>0 and len(comp_N)>0 else (comp_P if len(comp_P)>0 else comp_N)
    return scc_minimal(res)

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