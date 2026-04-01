import numpy as np
import time
import sys
from Lowry_Clishe_TautCheck import is_tautology, parse_espresso, Tracker
from Lowry_Clishe_CompGen import complement, export_espresso, ComplTracker

# --- CORE TOOLS ---
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

def get_primes(cover, off_set):
    """Expands every cube in the cover to its maximal prime implicant."""
    primes = []
    num_vars = cover.shape[1]
    for c in cover:
        p = c.copy()
        for v in range(num_vars):
            if p[v] != 3:
                p_test = p.copy()
                p_test[v] = 3
                # Intersection check with OFF-set
                overlap = (off_set & p_test) != 0
                if not np.any(np.all(overlap, axis=1)):
                    p[v] = 3
        primes.append(p)
    return np.unique(np.array(primes), axis=0)

def find_essentials(primes, vars_list):
    """
    Identifies Essential Prime Implicants. 
    A prime is essential if it covers a minterm no other prime covers.
    """
    essentials_mask = np.zeros(len(primes), dtype=bool)
    for i in range(len(primes)):
        p = primes[i]
        rest = np.delete(primes, i, axis=0)
        # If rest-cover cofactor w.r.t p is NOT a tautology, p is essential
        if not is_tautology(cofactor_cube(rest, p), vars_list, 0, Tracker())[0]:
            essentials_mask[i] = True
    return primes[essentials_mask], primes[~essentials_mask]

# --- IMPROVED HEURISTICS ---

def expand_with_weighting(cover, off_set):
    """
    Heuristic: Expand variables that appear most frequently in the cover.
    This encourages merges by expanding toward the 'center' of the logic.
    """
    num_cubes, num_vars = cover.shape
    # Count occurrences of 0s and 1s to see which literals are most 'crowded'
    weights = np.sum((cover == 1) | (cover == 2), axis=0)
    # Sort variables: expand most common literals first
    var_order = np.argsort(weights)[::-1]
    
    expanded = cover.copy()
    for i in range(num_cubes):
        c = expanded[i].copy()
        for v in var_order:
            if c[v] != 3:
                c_test = c.copy(); c_test[v] = 3
                overlap = (off_set & c_test) != 0
                if not np.any(np.all(overlap, axis=1)):
                    c[v] = 3
        expanded[i] = c
    return expanded

def irredundant_smart(candidates, essentials, vars_list):
    """
    Sorts candidates by size. Larger cubes are more likely to cover others, 
    so we try to remove the smaller ones first.
    """
    # Sort by number of dashes (ascending: check smaller cubes first)
    dash_counts = np.sum(candidates == 3, axis=1)
    candidates = candidates[np.argsort(dash_counts)]
    
    keep = np.ones(len(candidates), dtype=bool)
    for i in range(len(candidates)):
        curr = candidates[i]
        others = candidates[keep]
        # Ensure we don't include ourselves in the check
        others = others[~np.all(others == curr, axis=1)]
        # Combine with essentials for the tautology check context
        full_rest = np.vstack((essentials, others)) if len(essentials) > 0 else others
        
        if is_tautology(cofactor_cube(full_rest, curr), vars_list, 0, Tracker())[0]:
            keep[i] = False
    return candidates[keep]

# --- Clasic Heuristics ---
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
    return irredundant_smart(combined_cover, vars_list)

# --- MAIN LOOP ---

def minimize_aggressive(cover, ilb):
    off_set = complement(cover, ComplTracker())
    
    print(f"[*] Initial Cover: {len(cover)} cubes")
    
    # 1. Expand all to Primes and find Essentials
    all_primes = get_primes(cover, off_set)
    essentials, candidates = find_essentials(all_primes, ilb)
    print(f"[*] Found {len(essentials)} Essential Primes. Minimizing {len(candidates)} candidates...")

    best_cover = np.vstack((essentials, candidates)) if len(candidates) > 0 else essentials
    
    # 2. Heuristic Loop
    for it in range(1, 15):
        prev_count = len(best_cover)
        
        # We only manipulate the 'candidates' part of the cover
        candidates = reduce_cover(candidates, ilb) # Use your existing reduce
        candidates = expand_with_weighting(candidates, off_set)
        candidates = irredundant_smart(candidates, essentials, ilb)
        
        # Every few iterations, re-check if any candidates became essential
        if it % 3 == 0:
            current_all = np.vstack((essentials, candidates))
            essentials, candidates = find_essentials(get_primes(current_all, off_set), ilb)
        
        best_cover = np.vstack((essentials, candidates)) if len(candidates) > 0 else essentials
        print(f"Iteration {it}: {len(best_cover)} cubes")
        
        #if len(best_cover) >= prev_count and it > 5:   #Broken
        #    # Shake the tree one last time
        #    candidates = last_gasp(candidates, off_set, ilb)
        #    if len(np.vstack((essentials, candidates))) >= prev_count:
        #        print("[!] Convergence reached.")
        #        break
                
    return best_cover

# Run the aggressive minimizer
if __name__ == "__main__":
    start_time = time.perf_counter()
    input_file = sys.argv[1]
    cover, ilb, ob = parse_espresso(input_file)
    print(f"Parsed {input_file} containing {len(cover)} cubes in {(time.perf_counter() - start_time):.3f}.")
            
    start_time = time.perf_counter()
    np.random.seed(1234)
    minimized_cover = minimize_aggressive(cover, ilb)
    min_time = time.perf_counter() - start_time
    
    start_time = time.perf_counter()
    output_filepath = f"Lowry_Clishe_Minimized/{input_file.split('/')[-1]}_minimal"
    export_espresso(minimized_cover, ilb, ob, output_filepath)
    
    print(f"\nReduced from {len(cover)} cubes to {len(minimized_cover)} cubes in {min_time:.3f} sec")
    print(f"Cover exported to {output_filepath} in {(time.perf_counter() - start_time):.3f} sec\n")