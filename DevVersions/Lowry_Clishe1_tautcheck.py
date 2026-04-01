#Use 2 bit data type: 0=01, 1=10, -=11, void=00
#Keep separate data sets to track what row/column is which cube/variable
#Use bit packing for vectorized logic operations

#Check special cases:
#If no dashes in cover, it's a tautology if an only if there are 2^n unique cubes
#If a column of all 1's or all 0's exists, it's not a tautology
#If a universal cube of all dashes esists, it is a tautology
#If all columns are unate (any mix of + and - unate), it's not a tautology unless universal cube (but that was checked first so just fail here)
#If you put all the unate columns in their own cover and no row has a univeral cube it's not a tautology. If there is one, it doens't mean anything. Save this for later

#General:
#Take the unate sub cover from before
#Sort its cubes (rows) so all the universal cubes are on the bottom
#Make a cover with the unate cubes removed and sort it's cubes to match the resulting order of the unate sub cover
#Look at the bottom cubes in the non unate cover. The whole cover is a tautology if and only if they are a tautology
#Maybe sorting and copying stuff isn't ideal, might be a better way to track this 
#Take the variables of each cube and look at the ones that are part of a unate variable. 
#If those terms are all dashes, place the cube in a new cover but without the unate variables

#Else:
#Pick the most binate variable (trying to get smallest and most evenly split cofactors) (last dashes followed by closest to 50/50 split of 1's and 0's)
#Pass the positive then negative cofactor of F using this variable through new instances. If either fail, return false. If both true, return true

#Originally based on Gemini1_tautcheck then modified

import time
import random
#import numpy as np

def is_tautology(cover, path, depth, n):
    global max_depth, universal_cube_count, n_unique_cubes_count, pure_column_count, all_unate_columns_count, no_unate_universal_count, unate_reduction_count, binate_split_count
    max_depth = max(max_depth, depth)

    #Check for any universal cubes
    universal_cube = '-' * n                                                                            #Define universal cube for this n
    if universal_cube in cover:                                                                         #If the universal cube exists in the cover
        universal_cube_count += 1                                                                       #Increment the counter
        return True, None                                                                               #Return that this is a tautology
    
    # See if there are no dashes in the cover. If so, then if the number of unique cubes equals 2^n, it's a tautology, otherwise, it's not a tautology
    active_vars = [j for j in range(n) if j not in path]                                                #List variables that are used in this sub cover
    if not any('-' in cube for cube in cover):
        n_unique_cubes_count += 1
        k = len(active_vars)
        unique_cubes = set(cover)
        if len(unique_cubes) == 2**k: return True, None
        else:
            # Find the missing assignment efficiently
            existing = {"".join(c[v] for v in active_vars) for c in unique_cubes}
            missing_bits = None
            for i in range(2**k):
                bits = bin(i)[2:].zfill(k)
                if bits not in existing:
                    missing_bits = bits
                    break
            
            witness = path.copy()
            for idx, v in enumerate(active_vars):
                witness[v] = missing_bits[idx]
            return False, witness
    
    # 3. Base Case: Column of all 1's or all 0's
    global timer_01
    t = time.perf_counter()
    # Count literal occurrences
    c0_counts = [0] * n #np.zeros(n)
    c1_counts = [0] * n #np.zeros(n)
    for cube in cover:
        for j, val in enumerate(cube):
            if val == '0': c0_counts[j] += 1
            elif val == '1': c1_counts[j] += 1
    for j in active_vars:
        if c1_counts[j] == len(cover):
            pure_column_count += 1
            witness = path.copy()
            witness[j] = '0'
            timer_01 += time.perf_counter() - t
            return False, witness
        if c0_counts[j] == len(cover):
            pure_column_count += 1
            witness = path.copy()
            witness[j] = '1'
            timer_01 += time.perf_counter() - t
            return False, witness
    timer_01 += time.perf_counter() - t

    # ----------------------------------------------------------------
    # 4. Base Case: All columns unate
    # ----------------------------------------------------------------
    all_unate = all(not (c0_counts[j] > 0 and c1_counts[j] > 0) for j in active_vars)
    if all_unate:
        all_unate_columns_count += 1
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
            no_unate_universal_count += 1
            witness = path.copy()
            for j in unate_vars:
                witness[j] = '0' if c1_counts[j] > 0 else '1'
            return False, witness

    # ----------------------------------------------------------------
    # Unate Reduction
    # ----------------------------------------------------------------
    if unate_vars:
        cover_B = []
        for cube in cover:
            if all(cube[j] == '-' for j in unate_vars):
                cover_B.append(cube)

        unate_reduction_count += 1
        new_path = path.copy()
        for j in unate_vars:
            new_path[j] = '0' if c1_counts[j] > 0 else '1'
        return is_tautology(cover_B, new_path, depth + 1, n)

    # ----------------------------------------------------------------
    # Binate Split
    # ----------------------------------------------------------------
    binate_vars = [j for j in active_vars if c0_counts[j] > 0 and c1_counts[j] > 0]
    
    # Heuristic: Pick variable with highest presence (c0 + c1), tie-break using min(c0, c1)
    best_var = max(binate_vars, key=lambda j: (c0_counts[j] + c1_counts[j], min(c0_counts[j], c1_counts[j])))
    
    binate_split_count += 1

    # 1. Positive Cofactor
    pos_cover = []
    for cube in cover:
        if cube[best_var] == '0': continue
        new_cube = list(cube); new_cube[best_var] = '-'; pos_cover.append("".join(new_cube))
    
    pos_path = path.copy()
    pos_path[best_var] = '1'
    res_pos, wit_pos = is_tautology(pos_cover, pos_path, depth + 1, n)
    if not res_pos:
        return False, wit_pos

    # 2. Negative Cofactor
    neg_cover = []
    for cube in cover:
        if cube[best_var] == '1': continue
        new_cube = list(cube); new_cube[best_var] = '-'; neg_cover.append("".join(new_cube))

    neg_path = path.copy()
    neg_path[best_var] = '0'
    res_neg, wit_neg = is_tautology(neg_cover, neg_path, depth + 1, n)
    if not res_neg:
        return False, wit_neg

    return True, None

#1, 0.06; 2, 1.05; 3, 485; 4, 145; 5, 0.0186; 6, 0.639; 7, 33.1; 8, 1.07
#Input the espresso file
FILE_NAME = "TC_T2"                                                                                     #Define name of espresso cover file
if FILE_NAME[3] == "T": INPUT_FILE = "Tautology-Checking-Tests/" + FILE_NAME                            #If it's a test file, set the input folder accordingly
else: INPUT_FILE = "Tautology-Checking-Benchmarks/" + FILE_NAME                                         #Otherwise, assume it's a benchmark file and set the input folder accordingly
OUTPUT_FILE = "Lowry_Clishe_Tautology_Witnesses/" + FILE_NAME + "_off_cube"                             #Set the output file to be in the witness folder with the name of the file plus "_off_cube" appended
cover = []                                                                                              #Create list to append with cubes in the cover
with open(INPUT_FILE, 'r') as file:                                                                     #Read the input file
    for line in file:                                                                                   #Go through each line
        line = line.strip()                                                                             #Get the line as a string
        if line.startswith('.i '): NUM_INPUTS = int(line.split()[1])                                    #If the line starts with .i cast everything after .i to an integer and save it as the number of inputs
        elif line.startswith('.ilb '): ILB = line                                                       #If the line starts with .ilb save it as the variable names
        elif line.startswith('.ob '): OB = line                                                         #If the line starts with .ob save it as the output names
        elif line == '.e': break                                                                        #If the line starts with .e, you've reached the end
        elif line.startswith('.'): continue                                                             #If the line starts with . and isn't one we've looked at already, skip to the next one
        else: cover.append(line.split()[0])                                                             #The line doesn''t start with a . so it must be part of the cover. Take everything in the line before the first space (the cube ignoring the output) and append it to the cover list
#cover = np.array(cover, dtype=f"U{NUM_INPUTS}")

#Create variables to track stats
max_depth = 0
universal_cube_count = 0
n_unique_cubes_count = 0
pure_column_count = 0
all_unate_columns_count = 0
no_unate_universal_count = 0
unate_reduction_count = 0
binate_split_count = 0

print(f"Loaded cover with {NUM_INPUTS} variables and {len(cover)} cubes...\n")          

timer_01 = 0

START_TAUTOLOGY = time.perf_counter()
IS_TAUT, WITNESS_DICT = is_tautology(cover, {}, 0, NUM_INPUTS)
TAUTOLOGY_TIME = time.perf_counter() - START_TAUTOLOGY

print(timer_01)

print(f"Universal Cube:                         {universal_cube_count}")
print(f"2^n Unique Cubes:                       {n_unique_cubes_count}")
print(f"Column of All 1's or 0's:               {pure_column_count}")
print(f"All Columns Unate:                      {all_unate_columns_count}")
print(f"No Universal Cube in Unate Sub-Cover:   {no_unate_universal_count}")

print(f"\nUnate Reductions:                       {unate_reduction_count}")
print(f"Binate Splits:                          {binate_split_count}")
print(f"Max Recursion Depth:                    {max_depth}")
print(f"Time To Tautology:                      {TAUTOLOGY_TIME:.3f} seconds")

if IS_TAUT:
    print("\nResult: The cover is a tautology")
else:
    START_EXPORT = time.perf_counter()

    """Converts a partial assignment dictionary to a full witness string."""
    witness = ['0'] * NUM_INPUTS  # Default unassigned binate/free variables to '0'
    for k, v in WITNESS_DICT.items():
        witness[k] = v
    WITNESS_CUBE = "".join(witness)

    n = len(WITNESS_CUBE)
    with open(OUTPUT_FILE, 'w') as f:
        f.write(f".i {n}\n")
        f.write(f".o 1\n")
        f.write(f".p 1\n")
        f.write(f"{ILB}\n")
        f.write(f"{OB}\n")
        f.write(f"{WITNESS_CUBE} 1\n")
        f.write(".e\n")
 
    EXPORT_TIME = time.perf_counter() - START_EXPORT
    print("\nResult: The cover is NOT a tautology\n")    
    print(f"Negative Witness generated: {WITNESS_CUBE}")
    print(f"Witness exported to: {OUTPUT_FILE}")
    print(f"Time to generate/export witness: {EXPORT_TIME:.6f} seconds\n")