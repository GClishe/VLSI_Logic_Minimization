#https://gemini.google.com/share/8a105cdce656
#Make me a complete espresso logic minimizer in python. Here are some preexisting functions I have for complement
#generation and tautology checking. The function should first generate an off set. Then, reduce the cover. This
#is where you try to replace all '-' (represented as 3 in my code) with a 1 or a 0. This can be done by creating
#a copy of the cover with the cube removed, and taking it's cofactor with respect to a modified version of the
#cube that has the target '-' replaced by a '0' (represented by 1 in my code. If this co factored sub cover is
#a tautology, the '-' in the original cube can be replaced with a '1'. If not try again but modify the cube to
#replace the '-' with a '1' (represented by 3 in my code). If neither work, just move on. Next you would expand.
#This is where you try to replace all the '1' and '0' with '-'. This is done by taking the target cube and
#inverting the target variable. If the modified cube does not intersect the off set (can use bit wise AND), then
#the target variable can become a '-'. The result of this step should be SCC (single cube containment) minimal.
#This is where you make sure a cube is not contained by the rest of the cover by checking if the cofactor of the
#cover with respect to each cube is not a tautology. If it is, the cube can be removed. Last is irredundant.
#This is where you go through all the cubes and try to remove some. This is done by removing the cube form the
#cover and seeing if the cofactor of this with respect to the cube is a tautology. If it is, the cube can be
#removed from the cover. Also provided are two example espresso files. LM_T3 should be reduced to 120 cubes, 
#and LM_T2 should be reduced to 28 cubes.

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
    
    for i in range(num_cubes):
        c = reduced_cover[i].copy()
        # Create a sub-cover excluding the current cube
        rest_cover = np.delete(reduced_cover, i, axis=0)
        
        for v in range(num_vars):
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
    
    for i in range(num_cubes):
        c = expanded_cover[i].copy()
        
        for v in range(num_vars):
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
    
    for i in range(num_cubes):
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

def minimize(cover, vars_list, max_iterations=10):
    """
    The main heuristic minimization loop.
    """
    print("\n[Step 1] Generating OFF-set (Complement)...")
    t_start = time.perf_counter()
    off_set = complement(cover, ComplTracker())
    print(f"OFF-set generated: {len(off_set)} cubes in {(time.perf_counter() - t_start):.3f} sec")

    # To guarantee a baseline SCC minimal cover before the heavy lifting
    current_cover = make_irredundant(cover, vars_list)
    print(f"\nInitial Irredundant Cover: {len(current_cover)} cubes")

    iteration = 1
    while iteration <= max_iterations:
        prev_cube_count = len(current_cover)
        print(f"\n--- Iteration {iteration} ---")
        
        # 1. Reduce
        t0 = time.perf_counter()
        current_cover = reduce_cover(current_cover, vars_list)
        print(f"Reduce finished in {(time.perf_counter() - t0):.3f} sec")
        
        # 2. Expand
        t0 = time.perf_counter()
        current_cover = expand_cover(current_cover, off_set)
        print(f"Expand finished in {(time.perf_counter() - t0):.3f} sec")
        
        # 3. Irredundant
        t0 = time.perf_counter()
        current_cover = make_irredundant(current_cover, vars_list)
        new_cube_count = len(current_cover)
        print(f"Irredundant finished. Cubes: {new_cube_count} (took {(time.perf_counter() - t0):.3f} sec)")
        
        # Check for convergence
        if new_cube_count >= prev_cube_count:
            print(f"\nConvergence reached. No further minimization possible.")
            break
            
        iteration += 1

    return current_cover

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 Lowry_Clishe_Minimizer.py <input_file>")
        sys.exit(1)
        
    input_file = sys.argv[1]
    
    # 1. Parse
    start_time = time.perf_counter()
    cover, ilb, ob = parse_espresso(input_file)
    print(f"Parsed {input_file} containing {len(cover)} cubes.")
    
    # If variable names weren't explicitly supplied, generate dummy names
    num_vars = cover.shape[1]
    if not ilb:
        ilb = [f"x{i}" for i in range(num_vars)]
        
    # 2. Minimize
    minimized_cover = minimize(cover, ilb)
    
    total_time = time.perf_counter() - start_time
    
    # 3. Export & Summary
    output_filepath = f"Minimized_{input_file.split('/')[-1]}"
    export_espresso(minimized_cover, ilb, ob, output_filepath)
    
    print(f"\n=== Minimization Summary ===")
    print(f"Original Cubes:  {len(cover)}")
    print(f"Minimized Cubes: {len(minimized_cover)}")
    print(f"Total Time:      {total_time:.3f} sec")
    print(f"Exported to:     {output_filepath}\n")