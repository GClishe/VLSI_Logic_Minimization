#Since the last AI generated version, I have learned that the binary representation doesn't work great for tautology checking
#which is mostly moving values arround, something made much easier by not packing the bits.
#I also found that general unate reduction is pretty much always faster and more effective than the unate split

#https://gemini.google.com/share/a3dd79475c16

#Write me a python script to import a cover, in the espresso format and determine if it is a tautology. If it isn't produce a witness cube as an exported file in the espresso format. 
#
#The algorithm should check the following special cases in the provided order:
#1. If a universal cube of all dashes exists, the cover is a tautology.
#2. If there are no dashes in the cover, it is a tautology if and only if 2^n unique cubes exist. n is the number of variables in the cover.
#3. If a column of all 1's or all 0's exists, the cover is not a tautology.
#4. If all columns are unate (any mix of positive or negative) it is not a tautology. (Unless there is a universal cube but that was checked first).
#5. If you put all the unate variables on their own, and there is no universal cube, the cover is not a tautology.
#
#If none of the special cases produce an answer, use general unate reduction to reduce the problem. This is where you take the variables of each cube and look at the ones that are part of a unate variable. If those terms are all dashes, place the cube in a new cover and but without the unate variables. This new cover should be passed to a recursive instance of the tautology checker. The parent cover is a tautology if and only if this new cover is a tautology.
#
#Finally, if you get this point, execute a binate split. To do this, first pick the most binate variable. Pass its positive then its negative cofactor as a recursive instance of the tautology checker. If either returns false, the parent cover is not a tautology, otherwise, the parent cover is a tautology. 
#
#Track the max number of recursive layers, how many times each special case was used, how many times unate reduction was used, how many times a binate split was used, the time taken to do the tautology check, and the time taken to generate the negative witness and its espresso file. Make sure the output witness contains the variable names (.ilb and .ob)
#
#It may be useful to represent the 0's as 01, 1's as 10, and - as 11. It can help to execute operations as vectors this way. 
#
#Also make sure that when you recursively pass sub covers that you actually make a sub cover with less cubes and less variables. 
#Attached are 2 example files. TC_T1 is a tautology. TC_T5 is not. 

import sys
import time

# Positional Cube Notation (PCN) mappings
V_0 = (0, 1)
V_1 = (1, 0)
V_DASH = (1, 1)

class Tracker:
    def __init__(self):
        self.max_depth = 0
        self.case1_count = 0
        self.case2_count = 0
        self.case3_count = 0
        self.case4_count = 0
        self.case5_count = 0
        self.unate_red_count = 0
        self.binate_split_count = 0

def parse_espresso(filepath):
    """Parses an Espresso file and returns the cover matrix, ilb, and ob."""
    ilb = []
    ob = []
    cover = []
    num_inputs = 0
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
            
        if line.startswith('.i '):
            num_inputs = int(line.split()[1])
        elif line.startswith('.o '):
            pass # Mostly 1 for these checks
        elif line.startswith('.ilb '):
            ilb = line.split()[1:]
        elif line.startswith('.ob '):
            ob = line.split()[1:]
        elif line.startswith('.p '):
            pass
        elif line.startswith('.e'):
            break
        elif line[0] in '01-':
            # Extract only the input part of the cube
            input_part = line.split()[0]
            if len(input_part) > num_inputs and num_inputs != 0:
                input_part = input_part[:num_inputs]
                
            cube = []
            for char in input_part:
                if char == '0':
                    cube.append(V_0)
                elif char == '1':
                    cube.append(V_1)
                elif char == '-':
                    cube.append(V_DASH)
            cover.append(cube)
            
    # Default labels if missing
    if not ilb and cover:
        ilb = [f"v{i}" for i in range(len(cover[0]))]
    if not ob:
        ob = ["out"]
        
    return cover, ilb, ob

def is_tautology(cover, vars, depth, tracker):
    tracker.max_depth = max(tracker.max_depth, depth)
    n = len(vars)

    # Base case for empty cover (Not a Tautology)
    if not cover:
        return False, {v: 0 for v in vars}

    # 1. Universal Cube check
    for cube in cover:
        if all(val == V_DASH for val in cube):
            tracker.case1_count += 1
            return True, None

    # 2. No dashes in the cover
    has_dash = any(V_DASH in cube for cube in cover)
    if not has_dash:
        tracker.case2_count += 1
        unique_cubes = set(tuple(c) for c in cover)
        if len(unique_cubes) == (1 << n):
            return True, None
        else:
            # Find the missing minterm (witness)
            for i in range(1 << n):
                # Build tuple mapping bit 0 -> V_0, bit 1 -> V_1
                c_tuple = tuple(V_1 if ((i >> (n - 1 - j)) & 1) else V_0 for j in range(n))
                if c_tuple not in unique_cubes:
                    wit = {vars[j]: 1 if c_tuple[j] == V_1 else 0 for j in range(n)}
                    return False, wit

    # 3. Column of all 1s or all 0s
    for i in range(n):
        col = [c[i] for c in cover]
        if all(v == V_1 for v in col):
            tracker.case3_count += 1
            wit = {v: 0 for v in vars}
            wit[vars[i]] = 0 # Missing phase
            return False, wit
        if all(v == V_0 for v in col):
            tracker.case3_count += 1
            wit = {v: 0 for v in vars}
            wit[vars[i]] = 1 # Missing phase
            return False, wit

    # Find unate and binate columns
    unate_cols = []
    binate_cols = []
    for i in range(n):
        col_vals = set(c[i] for c in cover)
        if V_0 in col_vals and V_1 in col_vals:
            binate_cols.append(i)
        else:
            unate_cols.append(i)

    # 4. All columns are unate
    if len(binate_cols) == 0:
        tracker.case4_count += 1
        wit = {}
        for i in range(n):
            col_vals = set(c[i] for c in cover)
            wit[vars[i]] = 0 if V_1 in col_vals else 1
        return False, wit

    # 5. General Unate Reduction (Isolate unate variables)
    if len(unate_cols) > 0:
        reduced_cover = []
        for cube in cover:
            # Check if all unate variables in this cube are dashes
            if all(cube[i] == V_DASH for i in unate_cols):
                reduced_cover.append([cube[i] for i in binate_cols])

        if len(reduced_cover) == 0:
            tracker.case5_count += 1
            wit = {v: 0 for v in vars}
            for i in unate_cols:
                col_vals = set(c[i] for c in cover)
                wit[vars[i]] = 0 if V_1 in col_vals else 1
            return False, wit
        else:
            tracker.unate_red_count += 1
            new_vars = [vars[i] for i in binate_cols]
            res, sub_wit = is_tautology(reduced_cover, new_vars, depth + 1, tracker)
            
            if res:
                return True, None
            else:
                wit = {}
                for i in range(n):
                    if i in unate_cols:
                        col_vals = set(c[i] for c in cover)
                        wit[vars[i]] = 0 if V_1 in col_vals else 1
                    else:
                        wit[vars[i]] = sub_wit[vars[i]]
                return False, wit

    # 6. Binate Split
    tracker.binate_split_count += 1

    # Pick the most binate variable (Maximize non-dash occurrences)
    best_col = -1
    max_binate_score = -1
    for i in binate_cols:
        score = sum(1 for c in cover if c[i] != V_DASH)
        if score > max_binate_score:
            max_binate_score = score
            best_col = i

    split_i = best_col
    split_var = vars[split_i]
    new_vars = [vars[i] for i in range(n) if i != split_i]

    # Positive Cofactor (x = 1)
    T_1 = []
    for cube in cover:
        if cube[split_i] == V_1 or cube[split_i] == V_DASH:
            T_1.append([cube[i] for i in range(n) if i != split_i])

    res1, wit1 = is_tautology(T_1, new_vars, depth + 1, tracker)
    if not res1:
        wit1[split_var] = 1
        return False, wit1

    # Negative Cofactor (x = 0)
    T_0 = []
    for cube in cover:
        if cube[split_i] == V_0 or cube[split_i] == V_DASH:
            T_0.append([cube[i] for i in range(n) if i != split_i])

    res0, wit0 = is_tautology(T_0, new_vars, depth + 1, tracker)
    if not res0:
        wit0[split_var] = 0
        return False, wit0

    return True, None

def export_witness(witness_dict, ilb, ob, output_filepath):
    """Exports the negative witness to an Espresso format file."""
    with open(output_filepath, 'w') as f:
        f.write(f".i {len(ilb)}\n")
        f.write(f".o {len(ob)}\n")
        f.write(f".ilb {' '.join(ilb)}\n")
        f.write(f".ob {' '.join(ob)}\n")
        f.write(".p 1\n")
        
        # Build the cube string exactly in the order of original .ilb
        cube_str = "".join(str(witness_dict[var]) for var in ilb)
        f.write(f"{cube_str} 1\n")
        f.write(".e\n")

def main():
    if len(sys.argv) < 2:
        print("Usage: python tautology_checker.py <input_espresso_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    
    # 1. Parse File
    cover, ilb, ob = parse_espresso(input_file)
    tracker = Tracker()

    # 2. Check Tautology (Timer started)
    start_time_check = time.perf_counter()
    is_taut, witness = is_tautology(cover, ilb, 0, tracker)
    end_time_check = time.perf_counter()
    check_duration_ms = (end_time_check - start_time_check) * 1000

    print(f"--- Results for {input_file} ---")
    print(f"Is Tautology?: {'YES' if is_taut else 'NO'}")

    # 3. Handle Witness & Timing
    witness_duration_ms = 0
    if not is_taut:
        start_time_wit = time.perf_counter()
        output_file = f"witness_{input_file.split('/')[-1]}"
        export_witness(witness, ilb, ob, output_file)
        end_time_wit = time.perf_counter()
        witness_duration_ms = (end_time_wit - start_time_wit) * 1000
        print(f"Witness cube exported to: {output_file}")
    
    # 4. Print Metrics
    print("\n--- Algorithm Statistics ---")
    print(f"Max Recursive Layers : {tracker.max_depth}")
    print(f"Case 1 (Universal)   : {tracker.case1_count} hits")
    print(f"Case 2 (No Dashes)   : {tracker.case2_count} hits")
    print(f"Case 3 (All 1/All 0) : {tracker.case3_count} hits")
    print(f"Case 4 (All Unate)   : {tracker.case4_count} hits")
    print(f"Case 5 (Unate Failed): {tracker.case5_count} hits")
    print(f"Unate Reductions Used: {tracker.unate_red_count} times")
    print(f"Binate Splits Exec   : {tracker.binate_split_count} times")
    print(f"Time (Check Tautology): {check_duration_ms:.4f} ms")
    if not is_taut:
        print(f"Time (Gen/Export Wit) : {witness_duration_ms:.4f} ms")

if __name__ == "__main__":
    main()