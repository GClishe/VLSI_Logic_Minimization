#Write me a python script to import a cover, in the espresso format and determine if it is a tautology. 
#If it isn't produce a witness cube as an exported file in the espresso format. 
#
#The algorithm should check the following special cases in the provided order:
#1. If there are no dashes in the cover, it is a tautology if and only if 2^n unique cubes exist. 
#   n is the number of variables in the cover.
#2. If a column of all 1's or all 0's exists, the cover is not a tautology.
#3. If a universal cube of all dashes exists, the cover is a tautology.
#4. If all columns are unate (any mix of positive or negative) it is not a tautology. 
#   (Unless there is a universal cube but that was checked first).
#5. If you put all the unate variables on their own, and there is no universal cube, the cover is not a tautology.
#
#If none of the special cases produce an answer, there are two options to reduce the problem. 
#Pick which one to use based on which produces a smaller cover for the next recursion (or produces a result). 
#Also add an option to manually choose one.
#1. If a positive unate variable exists the cover is a tautology if and only if its negative cofactor is a tautology. 
#   If a negative unate variable exists, the cover is a tautology if and only if its positive cofactor is a tautology. 
#   The cofactors here would be passed to a recursive instance of the tautology checker.
#2. General unate reduction. This is where you take the variables of each cube and look at the ones that are part of a unate variable. 
#   If those terms are all dashes, place the cube in a new cover and but without the unate variables. 
#   This new cover should be passed to a recursive instance of the tautology checker. 
#   The parent cover is a tautology if and only if this new cover is a tautology.
#
#Finally, if you get this point, execute a binate split. To do this, first pick the most binate variable. 
#Pass its positive then its negative cofactor as a recursive instance of the tautology checker. 
#If either returns false, the parent cover is not a tautology, otherwise, the parent cover is a tautology. 
#
#Track the max number of recursive layers, how many times each special case was used, 
#how many times unate reduction was used, how many times a binate split was used, 
#the time taken to do the tautology check, and the time taken to generate the negative witness and its espresso file.
#
#Attached are 2 example files. TC_T1 is a tautology. TC_T5 is not

#https://gemini.google.com/share/ab5c6aa6112d

#Option A doesn't work and is pretty much always slower. Remove it to save a bit of time not chekcing it
#Doesn't copy .ild or .ob to witness file

import argparse
import time
import sys
import random

class TautologyChecker:
    def __init__(self, unate_mode='auto'):
        self.unate_mode = unate_mode
        self.stats = {
            'max_depth': 0,
            'case_1_no_dashes': 0,
            'case_2_all_0s_or_1s': 0,
            'case_3_universal': 0,
            'case_4_all_unate': 0,
            'case_5_unate_isolated': 0,
            'unate_reduction_A': 0,
            'unate_reduction_B': 0,
            'binate_splits': 0
        }

    def parse_espresso(self, filepath):
        """Parses an espresso file and returns the number of variables and the cover."""
        n_inputs = 0
        cover = []
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'): continue
                if line.startswith('.i '):
                    n_inputs = int(line.split()[1])
                elif line.startswith('.') and not line.startswith('.e'):
                    continue  # Ignore .o, .p, .ilb, .ob
                elif line == '.e':
                    break
                else:
                    parts = line.split()
                    if len(parts) >= 1:
                        cube = parts[0]
                        if len(cube) == n_inputs:
                            cover.append(cube)
        return n_inputs, cover

    def export_witness(self, witness_cube, filepath):
        """Exports a negative witness cube to an espresso format file."""
        n = len(witness_cube)
        with open(filepath, 'w') as f:
            f.write(f".i {n}\n")
            f.write(f".o 1\n")
            f.write(f".p 1\n")
            f.write(f"{witness_cube} 1\n")
            f.write(".e\n")

    def fill_witness(self, path_dict, n):
        """Converts a partial assignment dictionary to a full witness string."""
        witness = ['0'] * n  # Default unassigned binate/free variables to '0'
        for k, v in path_dict.items():
            witness[k] = v
        return "".join(witness)

    def is_tautology(self, cover, path, depth, n):
        self.stats['max_depth'] = max(self.stats['max_depth'], depth)
        
        if not cover:
            return False, self.fill_witness(path, n)

        # Count literal occurrences
        c0_counts = [0] * n
        c1_counts = [0] * n
        for cube in cover:
            for j, val in enumerate(cube):
                if val == '0': c0_counts[j] += 1
                elif val == '1': c1_counts[j] += 1

        active_vars = [j for j in range(n) if j not in path]

        # ----------------------------------------------------------------
        # 1. Base Case: No dashes in the cover
        # ----------------------------------------------------------------
        if not any('-' in cube for cube in cover):
            k = len(active_vars)
            unique_cubes = set(cover)
            if len(unique_cubes) == 2**k:
                self.stats['case_1_no_dashes'] += 1
                return True, None
            else:
                self.stats['case_1_no_dashes'] += 1
                # Find the missing assignment efficiently
                existing = {"".join(c[v] for v in active_vars) for c in unique_cubes}
                missing_bits = None
                if k < 20: # Linear search
                    for i in range(2**k):
                        bits = bin(i)[2:].zfill(k)
                        if bits not in existing:
                            missing_bits = bits
                            break
                else: # Random sampling for massive k
                    while True:
                        i = random.getrandbits(k)
                        bits = bin(i)[2:].zfill(k)
                        if bits not in existing:
                            missing_bits = bits
                            break
                
                witness = path.copy()
                for idx, v in enumerate(active_vars):
                    witness[v] = missing_bits[idx]
                return False, witness

        # ----------------------------------------------------------------
        # 2. Base Case: Column of all 1's or all 0's
        # ----------------------------------------------------------------
        for j in active_vars:
            if c1_counts[j] == len(cover):
                self.stats['case_2_all_0s_or_1s'] += 1
                witness = path.copy()
                witness[j] = '0'
                return False, witness
            if c0_counts[j] == len(cover):
                self.stats['case_2_all_0s_or_1s'] += 1
                witness = path.copy()
                witness[j] = '1'
                return False, witness

        # ----------------------------------------------------------------
        # 3. Base Case: Universal cube exists
        # ----------------------------------------------------------------
        universal_cube = '-' * n
        if universal_cube in cover:
            self.stats['case_3_universal'] += 1
            return True, None

        # ----------------------------------------------------------------
        # 4. Base Case: All columns unate
        # ----------------------------------------------------------------
        all_unate = all(not (c0_counts[j] > 0 and c1_counts[j] > 0) for j in active_vars)
        if all_unate:
            self.stats['case_4_all_unate'] += 1
            witness = path.copy()
            for j in active_vars:
                witness[j] = '0' if c1_counts[j] > 0 else '1'
            return False, witness

        # ----------------------------------------------------------------
        # 5. Base Case: Unate variables on their own
        # ----------------------------------------------------------------
        unate_vars = [j for j in active_vars if (c0_counts[j] == 0 or c1_counts[j] == 0)]
        if unate_vars:
            has_univ_for_unate = any(all(cube[j] == '-' for j in unate_vars) for cube in cover)
            if not has_univ_for_unate:
                self.stats['case_5_unate_isolated'] += 1
                witness = path.copy()
                for j in unate_vars:
                    witness[j] = '0' if c1_counts[j] > 0 else '1'
                return False, witness

        # ----------------------------------------------------------------
        # Unate Reductions
        # ----------------------------------------------------------------
        if unate_vars:
            # OPTION A: Cofactor single unate variable
            var_A = unate_vars[0]
            val_A = '0' if c1_counts[var_A] > 0 else '1' # Take worst-case cofactor
            
            cover_A = []
            if self.unate_mode in ['A', 'auto']:
                for cube in cover:
                    if cube[var_A] == val_A: continue # Drop opposites
                    new_cube = list(cube)
                    new_cube[var_A] = '-'
                    cover_A.append("".join(new_cube))

            # OPTION B: General Unate Reduction (Cofactor all at once)
            cover_B = []
            if self.unate_mode in ['B', 'auto']:
                for cube in cover:
                    if all(cube[j] == '-' for j in unate_vars):
                        cover_B.append(cube)

            # Pick the strategy
            chosen_opt = self.unate_mode
            if chosen_opt == 'auto':
                chosen_opt = 'B' if len(cover_B) <= len(cover_A) else 'A'

            if chosen_opt == 'B':
                self.stats['unate_reduction_B'] += 1
                new_path = path.copy()
                for j in unate_vars:
                    new_path[j] = '0' if c1_counts[j] > 0 else '1'
                return self.is_tautology(cover_B, new_path, depth + 1, n)
            else:
                self.stats['unate_reduction_A'] += 1
                new_path = path.copy()
                new_path[var_A] = val_A
                return self.is_tautology(cover_A, new_path, depth + 1, n)

        # ----------------------------------------------------------------
        # Binate Split
        # ----------------------------------------------------------------
        binate_vars = [j for j in active_vars if c0_counts[j] > 0 and c1_counts[j] > 0]
        
        # Heuristic: Pick variable with highest presence (c0 + c1), tie-break using min(c0, c1)
        best_var = max(binate_vars, key=lambda j: (c0_counts[j] + c1_counts[j], min(c0_counts[j], c1_counts[j])))
        
        self.stats['binate_splits'] += 1

        # 1. Positive Cofactor
        pos_cover = []
        for cube in cover:
            if cube[best_var] == '0': continue
            new_cube = list(cube); new_cube[best_var] = '-'; pos_cover.append("".join(new_cube))
        
        pos_path = path.copy()
        pos_path[best_var] = '1'
        res_pos, wit_pos = self.is_tautology(pos_cover, pos_path, depth + 1, n)
        if not res_pos:
            return False, wit_pos

        # 2. Negative Cofactor
        neg_cover = []
        for cube in cover:
            if cube[best_var] == '1': continue
            new_cube = list(cube); new_cube[best_var] = '-'; neg_cover.append("".join(new_cube))

        neg_path = path.copy()
        neg_path[best_var] = '0'
        res_neg, wit_neg = self.is_tautology(neg_cover, neg_path, depth + 1, n)
        if not res_neg:
            return False, wit_neg

        return True, None

def main():
    parser = argparse.ArgumentParser(description="Espresso Tautology Checker")
    parser.add_argument("input_file", help="Path to the input Espresso file.")
    parser.add_argument("--out", "-o", default="witness.txt", help="Path for the output witness file (if not a tautology).")
    parser.add_argument("--unate-mode", choices=['A', 'B', 'auto'], default='auto', 
                        help="Choose Unate Reduction technique. 'A'=single, 'B'=general/all, 'auto'=smallest cover.")
    
    args = parser.parse_args()

    checker = TautologyChecker(unate_mode=args.unate_mode)
    n_inputs, cover = checker.parse_espresso(args.input_file)
    
    print(f"Loaded cover with {n_inputs} variables and {len(cover)} cubes.")

    # Time tautology check
    start_tautology = time.perf_counter()
    is_taut, witness_dict = checker.is_tautology(cover, {}, 0, n_inputs)
    end_tautology = time.perf_counter()

    tautology_time = end_tautology - start_tautology

    if is_taut:
        print("\n=> Result: The cover IS a tautology.")
    else:
        print("\n=> Result: The cover is NOT a tautology.")
        
        # Time witness generation and export
        start_export = time.perf_counter()
        witness_cube = checker.fill_witness(witness_dict, n_inputs)
        checker.export_witness(witness_cube, args.out)
        end_export = time.perf_counter()
        
        export_time = end_export - start_export
        print(f"   Negative Witness generated: {witness_cube}")
        print(f"   Witness exported to: {args.out}")
        print(f"   Time to generate/export witness: {export_time:.6f} seconds")

    print(f"\n--- Statistics ---")
    print(f"Time Taken (Algorithm):     {tautology_time:.6f} seconds")
    print(f"Max Recursion Depth:        {checker.stats['max_depth']}")
    print(f"Case 1 (No Dashes):         {checker.stats['case_1_no_dashes']}")
    print(f"Case 2 (All 0s or 1s):      {checker.stats['case_2_all_0s_or_1s']}")
    print(f"Case 3 (Universal Cube):    {checker.stats['case_3_universal']}")
    print(f"Case 4 (All Columns Unate): {checker.stats['case_4_all_unate']}")
    print(f"Case 5 (Unate Isolated):    {checker.stats['case_5_unate_isolated']}")
    print(f"Unate Reduction (Option A): {checker.stats['unate_reduction_A']}")
    print(f"Unate Reduction (Option B): {checker.stats['unate_reduction_B']}")
    print(f"Binate Splits:              {checker.stats['binate_splits']}")

if __name__ == "__main__":
    main()