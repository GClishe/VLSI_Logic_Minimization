#Given Lowry_Clishe2_tautcheck
#Here is my Tautology Checker code. Could you integrate NumPy to improve the speed? 

#https://gemini.google.com/share/4104e6fb71a1

import sys
import time
import numpy as np

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
        self.null_cover_time = 0.0
        self.all_unique_cubes_time = 0.0
        self.universal_cube_time = 0.0
        self.find_unate_binate_time = 0.0
        self.all_unate_time = 0.0
        self.GUR_fail_time = 0.0
        self.GUR_time = 0.0
        self.binate_split_time = 0.0
        self.binate_split_setup_time = 0.0

    def get_total_time(self):
        # Helper to simplify subtraction of nested recursion times
        return (self.null_cover_time + self.all_unique_cubes_time + 
                self.universal_cube_time + self.find_unate_binate_time + 
                self.all_unate_time + self.GUR_fail_time + 
                self.GUR_time + self.binate_split_time)

def parse_espresso(filepath):
    cover = []
    ilb = []
    ob = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line[0] in '01-':
                cube_str = line.split()[0]
                # Convert to integer mapping: 0->0, 1->1, '-'->2
                cube = [0 if c == '0' else 1 if c == '1' else 2 for c in cube_str]
                cover.append(cube)
            elif line.startswith('.ilb '):
                ilb = line.split()[1:]
            elif line.startswith('.ob '):
                ob = line.split()[1:]
            elif line.startswith('.e'):
                break
    
    # Convert cover to a NumPy array for vectorized processing
    cover_np = np.array(cover, dtype=np.int8) if cover else np.empty((0, len(ilb)), dtype=np.int8)
    return cover_np, ilb, ob

def is_tautology(cover, vars, depth, tracker):
    t = time.perf_counter()
    tracker.max_depth = max(tracker.max_depth, depth)
    n = len(vars)
    
    # 1. Null Cover Check
    if cover.shape[0] == 0:
        tracker.null_cover_count += 1
        tracker.null_cover_time += time.perf_counter() - t
        return False, {v: 0 for v in vars}
    tracker.null_cover_time += time.perf_counter() - t
    t = time.perf_counter()

    # 2. No Dashes Check (2^n Unique Cubes)
    if not np.any(cover == 2):
        unique_cubes = np.unique(cover, axis=0)
        if unique_cubes.shape[0] == (1 << n):
            tracker.all_unique_cubes_count += 1
            tracker.all_unique_cubes_time += time.perf_counter() - t
            return True, None
        
        # Fast missing cube calculation using binary base dot product
        powers = 1 << np.arange(n - 1, -1, -1)
        cube_ints = unique_cubes.dot(powers)
        
        expected_set = set(range(1 << n))
        actual_set = set(cube_ints)
        missing_int = (expected_set - actual_set).pop()
        
        wit = {}
        for j in range(n):
            wit[vars[j]] = (missing_int >> (n - 1 - j)) & 1
            
        tracker.all_unique_cubes_count += 1
        tracker.all_unique_cubes_time += time.perf_counter() - t
        return False, wit
    tracker.all_unique_cubes_time += time.perf_counter() - t
    t = time.perf_counter()

    # 3. Universal Cube Check (Row of all 2s)
    if np.any(np.all(cover == 2, axis=1)):
        tracker.univeral_cube_count += 1
        tracker.universal_cube_time += time.perf_counter() - t
        return True, None
    tracker.universal_cube_time += time.perf_counter() - t
    t = time.perf_counter()

    # 4. Find Unate/Binate & Check Pure Unate Columns
    has_0 = np.any(cover == 0, axis=0)
    has_1 = np.any(cover == 1, axis=0)

    # Column of all 1s (No 0s and no dashes)
    all_1s = np.all(cover == 1, axis=0)
    if np.any(all_1s):
        tracker.all_1_or_0_count += 1
        col_idx = np.where(all_1s)[0][0]
        wit = {v: 0 for v in vars}
        wit[vars[col_idx]] = 0
        tracker.find_unate_binate_time += time.perf_counter() - t
        return False, wit

    # Column of all 0s (No 1s and no dashes)
    all_0s = np.all(cover == 0, axis=0)
    if np.any(all_0s):
        tracker.all_1_or_0_count += 1
        col_idx = np.where(all_0s)[0][0]
        wit = {v: 0 for v in vars}
        wit[vars[col_idx]] = 1
        tracker.find_unate_binate_time += time.perf_counter() - t
        return False, wit
        
    tracker.find_unate_binate_time += time.perf_counter() - t
    t = time.perf_counter()

    # 5. Check If All Columns Are Unate
    unate_cols_mask = ~has_0 | ~has_1
    binate_cols_mask = has_0 & has_1
    
    unate_cols = np.where(unate_cols_mask)[0]
    binate_cols = np.where(binate_cols_mask)[0]

    if len(binate_cols) == 0:
        tracker.all_unate_count += 1
        wit = {}
        for i in range(n):
            wit[vars[i]] = 0 if has_1[i] else 1
        tracker.all_unate_time += time.perf_counter() - t
        return False, wit
    tracker.all_unate_time += time.perf_counter() - t
    t = time.perf_counter()

    # 6. General Unate Reduction (GUR)
    if len(unate_cols) > 0:
        # Check rows where ALL unate columns are dashes (2)
        mask = np.all(cover[:, unate_cols] == 2, axis=1)
        reduced_cover = cover[mask][:, binate_cols]
        
        if reduced_cover.shape[0] == 0:
            tracker.no_uni_in_unate_sub_count += 1
            wit = {v: 0 for v in vars}
            for i in unate_cols:
                wit[vars[i]] = 0 if has_1[i] else 1
            tracker.GUR_fail_time += time.perf_counter() - t
            return False, wit
        else:
            tracker.GUR_fail_time += time.perf_counter() - t
            t = time.perf_counter()
            inner_start_time = tracker.get_total_time()
            
            tracker.unate_reduction_count += 1
            new_vars = [vars[i] for i in binate_cols]
            
            res, sub_wit = is_tautology(reduced_cover, new_vars, depth + 1, tracker)
            
            inner_time_spent = tracker.get_total_time() - inner_start_time
            
            if res:
                tracker.GUR_time += (time.perf_counter() - t) - inner_time_spent
                return True, None
            else:
                wit = {}
                for i in range(n):
                    if i in unate_cols:
                        wit[vars[i]] = 0 if has_1[i] else 1
                    else:
                        wit[vars[i]] = sub_wit[vars[i]]
                tracker.GUR_time += (time.perf_counter() - t) - inner_time_spent
                return False, wit
    tracker.GUR_time += time.perf_counter() - t
    t = time.perf_counter()

    # 7. Binate Split
    inner_start_time = tracker.get_total_time()
    tracker.binate_split_count += 1
    
    # Fast approach to find variable with least '-' in binate columns
    dash_counts = np.sum(cover[:, binate_cols] == 2, axis=0)
    best_binate_idx = np.argmin(dash_counts)
    best_col = binate_cols[best_binate_idx]
    
    split_var = vars[best_col]
    new_vars = [vars[i] for i in range(n) if i != best_col]

    # Cofactor 1: Split on positive
    mask_1 = cover[:, best_col] != 0
    cofactor_1 = np.delete(cover[mask_1], best_col, axis=1)
    
    res1, wit1 = is_tautology(cofactor_1, new_vars, depth + 1, tracker)
    if not res1:
        wit1[split_var] = 1
        inner_time_spent = tracker.get_total_time() - inner_start_time
        tracker.binate_split_time += (time.perf_counter() - t) - inner_time_spent
        return False, wit1

    # Cofactor 0: Split on negative
    mask_0 = cover[:, best_col] != 1
    cofactor_0 = np.delete(cover[mask_0], best_col, axis=1)
    
    res0, wit0 = is_tautology(cofactor_0, new_vars, depth + 1, tracker)
    if not res0:
        wit0[split_var] = 0
        inner_time_spent = tracker.get_total_time() - inner_start_time
        tracker.binate_split_time += (time.perf_counter() - t) - inner_time_spent
        return False, wit0

    inner_time_spent = tracker.get_total_time() - inner_start_time
    tracker.binate_split_time += (time.perf_counter() - t) - inner_time_spent
    return True, None

def export_witness(witness_dict, ilb, ob, output_filepath):
    with open(output_filepath, 'w') as f:
        f.write(f".i {len(ilb)}\n")
        f.write(f".o {len(ob)}\n")
        f.write(".p 1\n")
        f.write(f".ilb {' '.join(ilb)}\n")
        f.write(f".ob {' '.join(ob)}\n")
        cube_str = "".join(str(witness_dict[var]) for var in ilb)
        f.write(f"{cube_str} 1\n")
        f.write(".e\n")
    return cube_str

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 Lowry_Clishe_tautcheck.py <filepath>")
        sys.exit(1)
        
    input_file = sys.argv[1]
    cover, ilb, ob = parse_espresso(input_file)
    tracker = Tracker()
    
    start_time = time.perf_counter()
    is_taut, witness = is_tautology(cover, ilb, 0, tracker)

    print(f"\n--- Results for {input_file} ---\n")
    print(f"Is Tautology?: {'YES' if is_taut else 'NO'}")        
    print("\n--- Algorithm Statistics ---")
    print(f"Cover Is Null:                             {tracker.null_cover_count} times in {tracker.null_cover_time:.3f} sec")
    print(f"No Dashes Check For 2^n Unique Cubes:      {tracker.all_unique_cubes_count} times in {tracker.all_unique_cubes_time:.3f} sec")
    print(f"Found Universal Cube:                      {tracker.univeral_cube_count} times in {tracker.universal_cube_time:.3f} sec")
    print(f"Found Column of All 1's or All 0's:        {tracker.all_1_or_0_count} times in {tracker.find_unate_binate_time:.3f} sec")
    print(f"All Columns Unate Without Universal Cube:  {tracker.all_unate_count} times in {tracker.all_unate_time:.3f} sec")
    print(f"No Universal Cube in Unate Sub-Cover:      {tracker.no_uni_in_unate_sub_count} times in {tracker.GUR_fail_time:.3f} sec")
    print(f"Unate Reduction:                           {tracker.unate_reduction_count} times in {tracker.GUR_time:.3f} sec")
    print(f"Binate Split:                              {tracker.binate_split_count} times in {tracker.binate_split_time:.3f} sec")
    print(f"Max Recursive Layers:                      {tracker.max_depth} layers")
    print(f"Time to Check Tautology:                   {(time.perf_counter() - start_time):.3f} sec\n")

    print(tracker.binate_split_setup_time)
    
    if not is_taut:
        start_time = time.perf_counter()
        print("--- Witness Information ---")
        # Ensure directory exists or adjust to your system setup
        output_file = f"Lowry_Clishe_Tautology_Witnesses/{input_file.split('/')[-1]}_off_cube"
        try:
            witness_str = export_witness(witness, ilb, ob, output_file)
            print(f"Witness Cube:             {witness_str}")
            print(f"Witness file exported to: {output_file}")
        except FileNotFoundError:
            print("Warning: Directory 'Lowry_Clishe_Tautology_Witnesses/' not found. Make sure it exists to export.")
            print(f"Witness Cube:             {''.join(str(witness[var]) for var in ilb)}")
            
        print(f"Time to Generate Witness: {(time.perf_counter() - start_time):.3f} sec\n")