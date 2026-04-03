import numpy as np
import time
import sys
from Lowry_Clishe_TautCheck import is_tautology, parse_espresso, Tracker
from Lowry_Clishe_CompGen import complement, export_espresso, ComplTracker

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
    Heuristic: Sort cubes by size (number of dashes) DESCENDING.
    Reducing the largest cubes first leaves more 'logical space' for others.
    """
    # Sort: Largest cubes (most 3s) first
    dash_counts = np.sum(cover == 3, axis=1)
    cover = cover[np.argsort(dash_counts)[::-1]]
    
    reduced_cover = cover.copy()
    num_cubes, num_vars = reduced_cover.shape
    for i in range(num_cubes):
        c = reduced_cover[i].copy()
        rest_cover = np.delete(reduced_cover, i, axis=0)
        for v in range(num_vars):
            if c[v] == 3:
                # Test '0' (1)
                c_test = c.copy(); c_test[v] = 1 
                if is_tautology(cofactor_cube(rest_cover, c_test), vars_list, 0, Tracker())[0]:
                    c[v] = 2; continue
                # Test '1' (2)
                c_test[v] = 2
                if is_tautology(cofactor_cube(rest_cover, c_test), vars_list, 0, Tracker())[0]:
                    c[v] = 1
        reduced_cover[i] = c
    return reduced_cover

def expand_cover(cover, off_set):
    """
    Heuristic: Sort cubes by size ASCENDING.
    Expanding the smallest cubes first gives them a chance to swallow 
    others before the space is taken by larger ones.
    """
    # Sort: Smallest cubes (fewest 3s) first
    dash_counts = np.sum(cover == 3, axis=1)
    cover = cover[np.argsort(dash_counts)]
    
    expanded_cover = cover.copy()
    num_cubes, num_vars = expanded_cover.shape
    for i in range(num_cubes):
        c = expanded_cover[i].copy()
        # Randomizing variable order here can also help break local minima, but ordered is ideal too
        for v in range(num_vars): #np.random.permutation(num_vars):
            if c[v] != 3:
                c_test = c.copy()
                c_test[v] = 3 # Try to expand to dash
                overlap = (off_set & c_test) != 0
                if not np.any(np.all(overlap, axis=1)):
                    c[v] = 3
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

def minimize(cover, vars_list, max_iterations=15):
    print(f"\n[Step 1] Generating OFF-set...")
    off_set = complement(cover, ComplTracker())
    current_cover = make_irredundant(cover, vars_list)
    
    best_count = len(current_cover)
    no_improvement_streak = 0

    for iteration in range(1, max_iterations + 1):
        prev_count = len(current_cover)
        
        current_cover = reduce_cover(current_cover, vars_list)
        current_cover = expand_cover(current_cover, off_set)
        current_cover = make_irredundant(current_cover, vars_list)
        
        new_count = len(current_cover)
        print(f"Iteration {iteration}: {new_count} cubes")

        if new_count < best_count:
            best_count = new_count
            no_improvement_streak = 0
        else:
            no_improvement_streak += 1

        # --- LAST GASP TRIGGER ---
        # If we haven't improved in 2 iterations, trigger Last Gasp
        if no_improvement_streak >= 2:
            print("!! Stalled. Triggering Last Gasp...")
            current_cover = last_gasp(current_cover, off_set, vars_list)
            # If Last Gasp didn't change anything, we are truly done
            if no_improvement_streak >= 4:
                print("Convergence reached.")
                break
            no_improvement_streak = 0 # Reset and try the loop again with new primes

    return current_cover

if __name__ == "__main__":
    start_time = time.perf_counter()
    input_file = sys.argv[1]
    cover, ilb, ob = parse_espresso(input_file)
    print(f"Parsed {input_file} containing {len(cover)} cubes in {(time.perf_counter() - start_time):.3f}.")
            
    start_time = time.perf_counter()
    np.random.seed(1234)
    minimized_cover = minimize(cover, ilb, 50)
    min_time = time.perf_counter() - start_time
    
    start_time = time.perf_counter()
    output_filepath = f"Lowry_Clishe_Minimized/{input_file.split('/')[-1]}_minimal"
    export_espresso(minimized_cover, ilb, ob, output_filepath)
    
    print(f"\nReduced from {len(cover)} cubes to {len(minimized_cover)} cubes in {min_time:.3f} sec")
    print(f"Cover exported to {output_filepath} in {(time.perf_counter() - start_time):.3f} sec\n")