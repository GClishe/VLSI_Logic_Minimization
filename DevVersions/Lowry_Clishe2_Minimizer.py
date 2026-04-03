#Test       1           2           3           4           5           6
#Target     255         28          120         495         8192        65535
#Cubes      130         28          124         504
#Time       1.989       0.318       9.804       138.1
#Bench      1           2           3           4           5           6           7           8           9           10
#Target     255         330         799         NA          6435        2943        769         4095        65535       115
#Cubes      255         336         804
#Time       2.168       18.46       124.0

# Lowry_Clishe_Minimizer.py
# Usage: python3 Lowry_Clishe_Minimizer.py <input_file.espresso>

#Irredundant before cover gen
#Combine this, gemini3, and gemini4
#Global flipper goes crazy on Test 1, global rand does nothing

import sys
import time
import numpy as np

# Import your preexisting tools
from Lowry_Clishe_TautCheck import is_tautology, parse_espresso, Tracker
from Lowry_Clishe_CompGen import complement, export_espresso, ComplTracker

def cofactor_cube(cover, cube):
    """
    Takes the Shannon cofactor of a cover with respect to an entire cube.
    It sequentially cofactors the cover against each literal in the cube.
    """
    if len(cover) == 0:
        return cover
        
    cof = cover.copy()
    for var_idx in range(len(cube)):
        val_pcn = cube[var_idx]
        if val_pcn != 3: # If not a Don't Care
            # Keep cubes that intersect with the cofactor literal
            mask = (cof[:, var_idx] & val_pcn) != 0
            cof = cof[mask]
            
            if len(cof) == 0:
                break
            
            # The cofactored variable becomes a 'Don't Care' (3) in the surviving cubes
            cof[:, var_idx] = 3
            
    return cof

def reduce_cover(cover, vars_list):
    """
    Tries to shrink cubes by replacing '-' (3) with '0' (1) or '1' (2).
    """
    reduced_cover = cover.copy()
    num_cubes, num_vars = reduced_cover.shape

    dash_counts = np.sum(reduced_cover == 3, axis=1)
    sort_indices = np.argsort(dash_counts)[::-1] # [::-1] reverses the array
    reduced_cover = reduced_cover[sort_indices]
    
    for i in range(num_cubes):
        c = reduced_cover[i].copy()
        # Create a sub-cover excluding the current cube
        rest_cover = np.delete(reduced_cover, i, axis=0)
        
        for v in np.random.permutation(num_vars): #range(num_vars):
            if c[v] == 3: # If it's a '-'
                # Test replacing '-' with '0' (PCN: 1)
                c_test = c.copy()
                c_test[v] = 1 
                cof = cofactor_cube(rest_cover, c_test)
                if is_tautology(cof, vars_list, 0, Tracker())[0]:
                    c[v] = 2 # The 0-part is redundant, so shrink to '1' (PCN: 2)
                    continue
                    
                # Test replacing '-' with '1' (PCN: 2)
                c_test[v] = 2
                cof = cofactor_cube(rest_cover, c_test)
                if is_tautology(cof, vars_list, 0, Tracker())[0]:
                    c[v] = 1 # The 1-part is redundant, so shrink to '0' (PCN: 1)
                    
        reduced_cover[i] = c # Update in place to help reduce subsequent cubes
        
    return reduced_cover

def expand_cover(cover, off_set):
    """
    Tries to grow cubes by replacing '0' (1) and '1' (2) with '-' (3).
    A variable can be expanded if the resulting cube does NOT intersect the OFF-set.
    """
    expanded_cover = cover.copy()
    num_cubes, num_vars = expanded_cover.shape

    dash_counts = np.sum(expanded_cover == 3, axis=1)
    sort_indices = np.argsort(dash_counts)
    expanded_cover = expanded_cover[sort_indices]
    
    for i in range(num_cubes):
        c = expanded_cover[i].copy()
        
        for v in np.random.permutation(num_vars): #range(num_vars):
            if c[v] != 3:
                orig_val = c[v]
                # Invert the target variable (1 -> 2, 2 -> 1)
                inverted_val = 2 if orig_val == 1 else 1
                
                c_test = c.copy()
                c_test[v] = inverted_val
                
                # Fast vectorized intersection check against the entire OFF-set.
                # Intersection happens if for ALL variables, the bitwise AND is != 0.
                overlap = (off_set & c_test) != 0
                intersects = np.any(np.all(overlap, axis=1))
                
                if not intersects:
                    c[v] = 3 # Safe to expand to '-'
                    
        expanded_cover[i] = c
        
    return expanded_cover

def make_irredundant(cover, vars_list):
    """
    Removes cubes that are completely covered by the remaining cubes in the cover.
    Achieves both SCC-minimal status and general irredundancy.
    """
    num_cubes = len(cover)
    keep_mask = np.ones(num_cubes, dtype=bool)
    
    for i in range(num_cubes): #np.random.permutation(num_cubes): #range(num_cubes):
        if not keep_mask[i]:
            continue
            
        c = cover[i]
        # Build the remaining cover from currently active cubes
        rest_cover = np.array([cover[j] for j in range(num_cubes) if keep_mask[j] and j != i])
        
        if len(rest_cover) == 0:
            continue
            
        cof = cofactor_cube(rest_cover, c)
        
        # If the cofactor is a tautology, the cube 'c' is redundant
        if is_tautology(cof, vars_list, 0, Tracker())[0]:
            keep_mask[i] = False
            
    return cover[keep_mask]

def last_gasp(cover, off_set, vars_list):
    """
    Attempts to find 'Super-Primes'. It expands every cube in the current 
    cover to its absolute maximal prime implicant and then runs irredundant 
    to see if these new, larger primes can replace multiple smaller ones.
    """
    num_vars = len(vars_list)
    prime_candidates = []
    
    # 1. Expand every cube to its maximal prime against the OFF-set
    for c in cover:
        prime = c.copy()
        # Try to turn every 0 or 1 into a dash (3)
        for v in range(num_vars):
            if prime[v] != 3:
                temp = prime.copy()
                temp[v] = 3
                # Intersection check with OFF-set: (off_set & temp) != 0
                # A cube intersects a row if ALL variables have a bitwise overlap
                overlap = (off_set & temp) != 0
                if not np.any(np.all(overlap, axis=1)):
                    prime[v] = 3
        prime_candidates.append(prime)
    
    # 2. Combine unique maximal primes with the current cover
    unique_primes = np.unique(np.array(prime_candidates), axis=0)
    combined_cover = np.vstack((cover, unique_primes))
    
    # 3. Let Irredundant pick the best subset from this 'super-set'
    return make_irredundant(combined_cover, vars_list)

def minimize(cover, vars_list, max_in_a_row):
    t = time.perf_counter()
    print("\nGenerating Complement...")
    off_set = complement(cover, 0, ComplTracker())
    print(f"Generated complement with {len(off_set)} cubes in {(time.perf_counter() - t):.3f} sec\n")

    t = time.perf_counter()
    print("Initial Irredundant Pass...") # To guarantee a baseline SCC minimal cover before the heavy lifting
    prev_cube_count = len(cover)
    in_a_row = 0
    iteration = 1
    cover = make_irredundant(cover, vars_list)
    print(f"Generated initial irredundant cover with {len(cover)} cubes in {(time.perf_counter() - t):.3f} sec\n")

    while in_a_row <= max_in_a_row:   
        print(f"Iteration {iteration}...")

        #idx = np.random.permutation(len(cover))
        #cover = cover[idx]
        cover = np.flip(cover, axis=1)

        t = time.perf_counter()
        cover = reduce_cover(cover, vars_list)
        print(f"Reduction finished in {(time.perf_counter() - t):.3f} sec")
        
        t = time.perf_counter()
        cover = expand_cover(cover, off_set)
        print(f"Expansion finished in {(time.perf_counter() - t):.3f} sec")
        
        t = time.perf_counter()
        cover = make_irredundant(cover, vars_list)
        new_cube_count = len(cover)
        print(f"Irredundant finished. Cubes: {new_cube_count} (took {(time.perf_counter() - t):.3f} sec)")
        
        if new_cube_count >= prev_cube_count: in_a_row += 1
        else: in_a_row = 0
        prev_cube_count = len(cover)
        iteration += 1

    return cover

if __name__ == "__main__":
    start_time = time.perf_counter()
    input_file = sys.argv[1]
    cover, ilb, ob = parse_espresso(input_file)
    print(f"Parsed {input_file} containing {len(cover)} cubes in {(time.perf_counter() - start_time):.3f}.")
            
    start_time = time.perf_counter()
    np.random.seed(1234)
    minimized_cover = minimize(cover, ilb, 5)
    min_time = time.perf_counter() - start_time
    
    start_time = time.perf_counter()
    output_filepath = f"Lowry_Clishe_Minimized/{input_file.split('/')[-1]}_minimal"
    export_espresso(minimized_cover, ilb, ob, output_filepath)
    
    print(f"\nReduced from {len(cover)} cubes to {len(minimized_cover)} cubes in {min_time:.3f} sec")
    print(f"Cover exported to {output_filepath} in {(time.perf_counter() - start_time):.3f} sec\n")