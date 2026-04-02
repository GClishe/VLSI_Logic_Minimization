#Test       1           2           3           4           5           6
#Target     255         28          120         495         8192        65535
#Cubes      308         28          122         499
#Time       2.241       0.249       3.482       59.63
#Bench      1           2           3           4           5           6           7           8           9           10
#Target     255         330         799         NA          6435        2943        769         4095        65535       115
#Cubes      301         333         797
#Time       1.973       9.288       25.43

#Flipping helps Test and Bench 1 but hurts everything else. Need to find what else is different about LCMinimizer.
#Also remove last gasp it doen't do much and way increases time
#Test Gemini 3 and 4 to see if they have any secret sauce
#Save this current one to Github too

import numpy as np
import time
import sys
from Lowry_Clishe_TautCheck import is_tautology, parse_espresso, Tracker
from Lowry_Clishe_CompGen import complement, export_espresso, ComplTracker

def cofactor_cube(cover, cube):
    """
    Takes the Shannon cofactor of a cover with respect to an entire cube.
    It sequentially cofactors the cover against each literal in the cube.
    """
    if len(cover) == 0: return cover 
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

def reduce(cover, vars_list):
    """
    Tries to shrink cubes by replacing '-' (3) with '0' (1) or '1' (2).
    """
    num_cubes, num_vars = cover.shape
    """
    Heuristic: Sort cubes by size (number of dashes) DESCENDING.
    Reducing the largest cubes first leaves more 'logical space' for others.
    """
    # Sort: Largest cubes (most 3s) first
    #cover = cover[np.argsort(np.sum(cover == 3, axis=1))[::-1]] # [::-1] reverses the array    #No regressions, some improvment and speedup, had one regression this time, bad
    
    for i in np.random.permutation(len(cover)): #range(num_cubes): #Seems even better than sorted, still holds keep3
        c = cover[i].copy()
        # Create a sub-cover excluding the current cube
        rest_cover = np.delete(cover, i, axis=0)
        
        for v in range(num_vars): #np.random.permutation(num_vars): #Rand breaks even on help vs hurt, considering it did nothing originally I think it's just messing with the seed. Oo regression, bad
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
        cover[i] = c # Update in place to help reduce subsequent cubes
    return cover

def expand(cover, off_set):
    """
    Tries to grow cubes by replacing '0' (1) and '1' (2) with '-' (3).
    A variable can be expanded if the resulting cube does NOT intersect the OFF-set.
    """
    """
    Heuristic: Sort cubes by size ASCENDING.
    Expanding the smallest cubes first gives them a chance to swallow 
    others before the space is taken by larger ones.
    """
    # Sort: Smallest cubes (fewest 3s) first
    #cover = cover[np.argsort(np.sum(cover == 3, axis=1))]   #Consistent regression bad
    
    num_cubes, num_vars = cover.shape
    for i in np.random.permutation(len(cover)): #range(num_cubes): #Rand Minor improvements, no regressions, keep2
        c = cover[i].copy()
        # Randomizing variable order here can also help break local minima, the ordering is cube order. Does this break the cover since the vars aren't tracked?
        for v in np.random.permutation(num_vars): #range(num_vars): #Rand has huge benefits and somehow doesn't break anything, keep1
            if c[v] != 3:
                c_test = c.copy()
                c_test[v] = 3 # Try to expand to dash
                overlap = (off_set & c_test) != 0
                if not np.any(np.all(overlap, axis=1)):
                    c[v] = 3
        cover[i] = c
    return cover

def irredundant(cover, vars_list):
    """
    Removes cubes that are completely covered by the remaining cubes in the cover.
    Achieves both SCC-minimal status and general irredundancy.
    """
    """
    Sorts candidates by size. Larger cubes are more likely to cover others, 
    so we try to remove the smaller ones first.
    """
    # Sort by number of dashes (ascending: check smaller cubes first)
    #cover = cover[np.argsort(np.sum(cover == 3, axis=1))]   #helps a tiny bit (never worse) and improves time a good amount on some. Made it consistently worse this time, yep still do bad
    #Or random
    #cover = cover[np.random.permutation(len(cover))] #Not as much speed up, sometimes helps more than sort, sometimes makes it worse. Makes it consistently worse this time too. yep still do bad
    keep = np.ones(len(cover), dtype=bool)
    for i, cube in enumerate(cover):
        others = cover[keep]
        others = others[~np.all(others == cube, axis=1)]
        # Ensure we don't include ourselves in the check
        
        if is_tautology(cofactor_cube(others, cube), vars_list, 0, Tracker())[0]:
            keep[i] = False
    return cover[keep]

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
    return irredundant(combined_cover, vars_list)

def minimize(cover, vars_list, max_in_a_row):
    t = time.perf_counter()
    print("Initial Irredundant Pass...") # To guarantee a baseline SCC minimal cover before the heavy lifting
    cover = np.array(list({tuple(cube) for cube in cover}), dtype=np.uint8) #Remove duplicate cubes
    in_a_row = 0
    #best_cover_count = -1
    iteration = 1
    cover = irredundant(cover, vars_list)
    prev_cube_count = len(cover)
    print(f"Generated initial irredundant cover with {prev_cube_count} cubes in {(time.perf_counter() - t):.3f} sec\n")

    t = time.perf_counter()
    print("Generating Complement...")
    off_set = complement(cover, 0, ComplTracker())
    print(f"Generated complement with {len(off_set)} cubes in {(time.perf_counter() - t):.3f} sec\n")

    while True:   
        print(f"Iteration {iteration}...")

        #idx = np.random.permutation(len(cover))    #Do mixing and reordering in the functions not here
        #cover = cover[idx]
        #cover = np.flip(cover, axis=1)

        t = time.perf_counter()
        cover = reduce(cover, vars_list)
        print(f"Reduction finished in {(time.perf_counter() - t):.3f} sec")
        
        t = time.perf_counter()
        cover = expand(cover, off_set)
        print(f"Expansion finished in {(time.perf_counter() - t):.3f} sec")
        
        t = time.perf_counter()
        cover = irredundant(cover, vars_list)
        new_cube_count = len(cover)
        print(f"Irredundant finished in {(time.perf_counter() - t):.3f} sec with {new_cube_count} cubes\n")
        
        if new_cube_count >= prev_cube_count: in_a_row += 1
        else: in_a_row = 0
        prev_cube_count = len(cover)
        iteration += 1

        if in_a_row >= max_in_a_row:
            ##If last gasp hasn't been run, or improvement was seen over best, run last gasp
            #if best_cover_count > new_cube_count or best_cover_count == -1:
            #    best_cover = cover
            #    best_cover_count = len(cover)
            #    cover = last_gasp(cover, off_set, vars_list)
            #    prev_cube_count = len(cover)
            #    in_a_row = 0
            #else:
            #    return best_cover
            return cover

if __name__ == "__main__":
    start_time = time.perf_counter()
    input_file = sys.argv[1]
    cover, ilb, ob = parse_espresso(input_file)
    print(f"Parsed {input_file} containing {len(cover)} cubes in {(time.perf_counter() - start_time):.3f} sec\n")
            
    start_time = time.perf_counter()
    np.random.seed(1234)
    minimized_cover = minimize(cover, ilb, 5)
    print(f"Reduced from {len(cover)} cubes to {len(minimized_cover)} cubes in {(time.perf_counter() - start_time):.3f} sec\n")
    
    start_time = time.perf_counter()
    output_filepath = f"Lowry_Clishe_Minimized/{input_file.split('/')[-1]}_minimal"
    export_espresso(minimized_cover, ilb, ob, output_filepath)
    print(f"Cover exported to {output_filepath} in {(time.perf_counter() - start_time):.3f} sec\n")