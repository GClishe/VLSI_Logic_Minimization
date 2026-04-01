#Write a complement generator in Python that accepts files in the espresso format. The resulting complement should be SCC minimal, 
#and exported as an espresso file. Provided is a stand alone tautology checker. You can integrate the main checker function into 
#the the complement generator if you need to check tautology. Also attached is a PDF with lecture notes on Unate Complements you 
#can use for reference if needed. I would suggest using the bit packing method it outlines, but the tautology checker doesn't use 
#this, so checking tautology would require converting to the format the checker uses.

import sys
import time
import numpy as np

# --- Tautology Checker Components (Integrated) ---

class Tracker:
    def __init__(self):
        self.max_depth = 0
        self.null_cover_count = 0
        self.all_unique_cubes_count = 0
        self.univeral_cube_count = 0
        self.all_1_or_0_count = 0
        self.all_unate_count = 0
        self.no_uni_in_unate_sub_count = 0
        self.unate_reduction_count = 0
        self.binate_split_count = 0
        self.null_cover_time = 0
        self.all_unique_cubes_time = 0
        self.universal_cube_time = 0
        self.find_unate_binate_time = 0
        self.all_unate_time = 0
        self.GUR_fail_time = 0
        self.GUR_time = 0
        self.binate_split_time = 0

def parse_espresso(filepath):
    cover = []
    ilb, ob = [], []
    with open(filepath, 'r') as f: lines = f.readlines()
    for line in lines:
        line = line.strip()
        if not line: continue
        if line[0] in '01-': cover.append([1 if c == '0' else 2 if c == '1' else 3 for c in line.split()[0]])
        elif line.startswith('.ilb '): ilb = line.split()[1:]
        elif line.startswith('.ob '): ob = line.split()[1:]
        elif line.startswith('.e'): break
    cover = np.array(cover, dtype=np.uint8)
    return cover, ilb, ob

def is_tautology(cover, vars, depth, tracker):
    t = time.perf_counter()
    tracker.max_depth = max(tracker.max_depth, depth)
    if cover.size == 0:
        tracker.null_cover_count += 1
        tracker.null_cover_time += time.perf_counter() - t
        return False, {v: 0 for v in vars}
    tracker.null_cover_time += time.perf_counter() - t
    t = time.perf_counter()

    n = len(vars)
    if 3 not in cover:
        unique_cubes = frozenset(tuple(c) for c in cover)
        if len(unique_cubes) == (2**n):
            tracker.all_unique_cubes_count += 1
            tracker.all_unique_cubes_time += time.perf_counter() - t
            return True, None
        for i in range(2**n):
            c_tuple = tuple(2 if ((i >> (n - 1 - j)) & 1) else 1 for j in range(n))
            if c_tuple not in unique_cubes:
                wit = {vars[j]: 1 if c_tuple[j] == 2 else 0 for j in range(n)}
                tracker.all_unique_cubes_count += 1
                tracker.all_unique_cubes_time += time.perf_counter() - t
                return False, wit
    tracker.all_unique_cubes_time += time.perf_counter() - t
    t = time.perf_counter()

    if any((cover == 3).all(axis=1)):
        tracker.univeral_cube_count += 1
        tracker.universal_cube_time += time.perf_counter() - t
        return True, None
    tracker.universal_cube_time += time.perf_counter() - t
    t = time.perf_counter()

    has_0 = (cover == 1).any(axis=0)
    has_1 = (cover == 2).any(axis=0)
    has_dashes = (cover == 3).any(axis=0)
    binate_cols_mask = has_0 & has_1
    unate_cols_mask = ~binate_cols_mask
    pure_unate = ~(binate_cols_mask | has_dashes)
    all_1s = pure_unate & has_1
    if any(all_1s):
        tracker.all_1_or_0_count += 1
        tracker.find_unate_binate_time += time.perf_counter() - t
        return False, {v: 0 for v in vars}
    all_0s = pure_unate & has_0
    if any(all_0s):
        tracker.all_1_or_0_count += 1
        tracker.find_unate_binate_time += time.perf_counter() - t
        return False, {v: 1 for v in vars}
    tracker.find_unate_binate_time += time.perf_counter() - t
    t = time.perf_counter()

    unate_cols = np.where(unate_cols_mask)[0]
    binate_cols = np.where(binate_cols_mask)[0]
    if len(binate_cols) == 0:
        tracker.all_unate_count += 1
        wit = {vars[i]: 0 if has_1[i] else 1 for i in range(n)}
        tracker.all_unate_time += time.perf_counter() - t
        return False, wit
    tracker.all_unate_time += time.perf_counter() - t
    t = time.perf_counter()

    if len(unate_cols) > 0:
        reduced_cover = cover[(cover[:, unate_cols] == 3).all(axis=1)][:, binate_cols]
        if len(reduced_cover) == 0:
            tracker.no_uni_in_unate_sub_count += 1
            wit = {v: 0 for v in vars}
            for i in unate_cols: wit[vars[i]] = 0 if has_1[i] else 1
            tracker.GUR_fail_time += time.perf_counter() - t
            return False, wit
        else:
            tracker.GUR_fail_time += time.perf_counter() - t
            t = time.perf_counter()
            cur_times = [tracker.null_cover_time, tracker.all_unique_cubes_time, tracker.universal_cube_time, tracker.find_unate_binate_time, tracker.all_unate_time, tracker.GUR_time, tracker.GUR_fail_time, tracker.binate_split_time]
            tracker.unate_reduction_count += 1
            new_vars = [vars[i] for i in binate_cols]
            res, sub_wit = is_tautology(reduced_cover, new_vars, depth + 1, tracker)
            diff = sum([getattr(tracker, a) for a in ["null_cover_time", "all_unique_cubes_time", "universal_cube_time", "find_unate_binate_time", "all_unate_time", "GUR_fail_time", "GUR_time", "binate_split_time"]]) - sum(cur_times)
            tracker.GUR_time += time.perf_counter() - t + diff
            if res: return True, None
            wit = {vars[i]: (0 if has_1[i] else 1) if i in unate_cols else sub_wit[vars[i]] for i in range(n)}
            return False, wit
    tracker.GUR_time += time.perf_counter() - t
    t = time.perf_counter()

    cur_times = [tracker.null_cover_time, tracker.all_unique_cubes_time, tracker.universal_cube_time, tracker.find_unate_binate_time, tracker.all_unate_time, tracker.GUR_time, tracker.GUR_fail_time, tracker.binate_split_time]
    tracker.binate_split_count += 1
    dash_counts = np.sum(cover[:, binate_cols] == 3, axis=0)
    best_col = binate_cols[np.argmin(dash_counts)]
    split_var = vars[best_col]
    new_vars = [vars[i] for i in range(n) if i != best_col]
    
    cofactor_1 = np.delete(cover[cover[:, best_col] != 1], best_col, axis=1)
    res1, wit1 = is_tautology(cofactor_1, new_vars, depth + 1, tracker)
    diff = sum([getattr(tracker, a) for a in ["null_cover_time", "all_unique_cubes_time", "universal_cube_time", "find_unate_binate_time", "all_unate_time", "GUR_fail_time", "GUR_time", "binate_split_time"]]) - sum(cur_times)
    if not res1:
        wit1[split_var] = 1
        tracker.binate_split_time += time.perf_counter() - t + diff
        return False, wit1
        
    cofactor_0 = np.delete(cover[cover[:, best_col] != 2], best_col, axis=1)
    res0, wit0 = is_tautology(cofactor_0, new_vars, depth + 1, tracker)
    diff = sum([getattr(tracker, a) for a in ["null_cover_time", "all_unique_cubes_time", "universal_cube_time", "find_unate_binate_time", "all_unate_time", "GUR_fail_time", "GUR_time", "binate_split_time"]]) - sum(cur_times)
    tracker.binate_split_time += time.perf_counter() - t + diff
    if not res0:
        wit0[split_var] = 0
        return False, wit0
        
    return True, None


# --- Complementation Components ---

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
    if len(sys.argv) < 2:
        print("Usage: python3 Lowry_Clishe_Complement.py <cover_file.pla>")
        sys.exit(1)

    input_file = sys.argv[1]
    
    start_time = time.perf_counter()
    cover, ilb, ob = parse_espresso(input_file)
    print(f"Parsed {input_file} containing {len(cover)} cubes in {(time.perf_counter() - start_time):.3f} sec")

    start_time = time.perf_counter()
    comp_cover = complement(cover)
    comp_time = time.perf_counter() - start_time
    
    print(f"\n--- Complementation Results ---")
    print(f"Original Cubes: {len(cover)}")
    print(f"Complement Cubes (SCC Minimal): {len(comp_cover)}")
    print(f"Time Taken: {comp_time:.3f} sec")

    output_file = f"{input_file.split('/')[-1].replace('.pla', '')}_complement.pla"
    export_espresso(comp_cover, ilb, ob, output_file)
    print(f"Exported complement to: {output_file}\n")