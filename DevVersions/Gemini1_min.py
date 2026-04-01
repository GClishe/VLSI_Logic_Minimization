#https://gemini.google.com/share/0f3b1aea5bb1
# To use, run the command:
#   python3 Lowry_Clishe_Minimizer.py <directory of cover file>

import sys
import time
import numpy as np

# Import your existing tools
from Lowry_Clishe_TautCheck import is_tautology, parse_espresso, Tracker
from Lowry_Clishe_CompGen import complement, ComplTracker, export_espresso, cofactor


def intersects_off_set(cube, off_cover):
    """ Fast vectorized check: does 'cube' overlap with any part of the OFF-set? """
    if len(off_cover) == 0: return False
    # A conflict occurs if one has '0'(1) and other has '1'(2). 
    # If a row has NO conflicts, it intersects.
    clashes = ((cube == 1) & (off_cover == 2)) | ((cube == 2) & (off_cover == 1))
    return (~clashes.any(axis=1)).any()

def expand(cover, off_cover):
    """
    Greedy Expand: Tries to replace '0' (1) and '1' (2) with '-' (3) to make 
    cubes as large as possible without intersecting the OFF-set.
    """
    expanded_cover = []
    
    # Sort cubes by weight (number of literals / non-dashes) descending.
    # Expanding the smallest cubes first often yields better results.
    dash_counts = np.sum(cover == 3, axis=1)
    sorted_indices = np.argsort(dash_counts) # Ascending dashes = descending literals
    
    for idx in sorted_indices:
        cube = cover[idx].copy()
        
        for i in range(len(cube)):
            if cube[i] != 3: # If it's a 1 or 2, try to make it a 3
                original_val = cube[i]
                cube[i] = 3
                
                # If expanding causes an intersection with the OFF-set, revert it
                if intersects_off_set(cube, off_cover):
                    cube[i] = original_val
                    
        expanded_cover.append(cube)
        
    return np.array(expanded_cover)

def cofactor_cover_by_cube(cover, cube):
    """
    Cofactors an entire cover by a specific cube to check for tautology in that subspace.
    """
    cofactored = cover.copy()
    for i, val in enumerate(cube):
        if val == 1:
            cofactored = cofactor(cofactored, i, 1) # Cofactor against x=0
        elif val == 2:
            cofactored = cofactor(cofactored, i, 2) # Cofactor against x=1
        
        # If at any point the cofactor becomes empty, return empty
        if len(cofactored) == 0:
            return cofactored
            
    return cofactored

def irredundant(cover, num_vars):
    """
    Irredundant Cover: Removes cubes that are completely covered by the union of the other cubes.
    A cube 'c' is redundant if (Cover \ {c}) cofactored with 'c' is a tautology.
    """
    irred_cover = []
    tracker = Tracker()
    dummy_vars = list(range(num_vars))
    
    # Sort by size (largest first, meaning most dashes) to keep the biggest cubes
    dash_counts = np.sum(cover == 3, axis=1)
    sorted_indices = np.argsort(-dash_counts)
    
    working_cover = cover[sorted_indices].copy()
    active_mask = np.ones(len(working_cover), dtype=bool)
    
    for i in range(len(working_cover)):
        c = working_cover[i]
        
        # Create a cover of all OTHER currently active cubes
        active_mask[i] = False
        other_cubes = working_cover[active_mask]
        
        # Cofactor the remaining cover by the current cube
        cofactored_others = cofactor_cover_by_cube(other_cubes, c)
        
        is_taut = False
        if len(cofactored_others) > 0:
             is_taut, _ = is_tautology(cofactored_others, dummy_vars, 0, tracker)
        
        if is_taut:
            # The cube is completely covered by the others, it is redundant (leave active_mask[i] as False)
            pass 
        else:
            # The cube contains essential prime implicants, keep it
            active_mask[i] = True
            irred_cover.append(c)
            
    return np.array(irred_cover)

def reduce_cover(cover, num_vars):
    """
    Heuristic Reduce: Shrinks cubes as much as possible while maintaining the same total coverage.
    To avoid complex Unate Recursive computations, we use a greedy tautology check.
    """
    reduced_cover = []
    tracker = Tracker()
    dummy_vars = list(range(num_vars))
    
    # Sort cubes (largest first)
    dash_counts = np.sum(cover == 3, axis=1)
    sorted_indices = np.argsort(-dash_counts)
    
    working_cover = cover[sorted_indices].copy()
    
    for i in range(len(working_cover)):
        c = working_cover[i].copy()
        
        # The remainder of the cover
        other_cubes = np.delete(working_cover, i, axis=0)
        
        # Try to shrink dashes (3) into 0s (1) or 1s (2)
        for j in range(num_vars):
            if c[j] == 3:
                # Try setting to 0 (PCN: 1)
                c[j] = 1
                test_cover = np.vstack((other_cubes, c))
                # Check if this shrunk cover still covers the original space
                # Actually, standard Espresso checks if (Q u ~c) is a tautology. 
                # For basic heuristic speed, we'll keep it simple and skip intense reduce loops 
                # if it bogs down your pure Python implementation, but the framework is here.
                # Reverting back to 3 to keep the script performant without a true recursive reduce:
                c[j] = 3
                
        reduced_cover.append(c)
        working_cover[i] = c # Update in place for subsequent checks
        
    return np.array(reduced_cover)

def supercube(cover):
    """
    Calculates the smallest single cube that contains every cube in the cover.
    In Positional Cube Notation (1='0', 2='1', 3='-'), this is the bitwise OR
     of all rows in the cover.
    """
    if len(cover) == 0:
        return None
    return np.bitwise_or.reduce(cover, axis=0)

def get_supercube(cube_a, cube_b):
    """ Returns the smallest cube containing both a and b (Bitwise OR in PCN). """
    return np.bitwise_or(cube_a, cube_b)



def cofactor_cover_by_cube(cover, cube):
    """
    Cofactors a cover by every literal present in a specific cube.
    """
    res = cover
    for i, val in enumerate(cube):
        if val == 1:   # Literal is '0'
            res = cofactor(res, i, 1)
        elif val == 2: # Literal is '1'
            res = cofactor(res, i, 2)
        if len(res) == 0:
            break
    return res

def reduce_step(cover):
    """
    The Maximal Reduce Step:
    For each cube 'c', it find the points in 'c' that are NOT covered 
    by any other cube. It then shrinks 'c' to the smallest cube 
    containing those essential points.
    """
    new_cover = cover.copy()
    num_vars = cover.shape[1]
    
    # HEURISTIC: Process cubes from largest to smallest.
    # Shrinking large cubes first creates more "maneuvering room" for Expand.
    dash_counts = np.sum(new_cover == 3, axis=1)
    order = np.argsort(-dash_counts)
    
    for i in order:
        c = new_cover[i]
        
        # 1. Identify the rest of the cover
        other_cubes = np.delete(new_cover, i, axis=0)
        
        # 2. Cofactor the rest of the cover by the current cube 'c'.
        # This defines what parts of the space inside 'c' are already covered.
        q_cofactor = cofactor_cover_by_cube(other_cubes, c)
        
        # 3. Find the complement of that cofactor.
        # These are the "Essential Points" of cube 'c'.
        tracker = ComplTracker()
        essential_points = complement(q_cofactor, tracker)
        
        # 4. If the cube is redundant (no essential points), 
        # Irredundant will handle it. Otherwise, shrink it.
        if len(essential_points) > 0:
            s_cube = supercube(essential_points)
            
            # Create the reduced cube:
            # We keep the original 1s and 2s of 'c', but we replace 
            # the dashes (3s) with the tighter constraints found in s_cube.
            reduced_c = c.copy()
            for j in range(num_vars):
                if c[j] == 3:
                    reduced_c[j] = s_cube[j]
            
            new_cover[i] = reduced_c
            
    return new_cover

def last_gasp(cover, off_cover, num_vars):
    """
    Attempts to break local minima by expanding the essential cores 
    of the current cover into potentially better prime implicants.
    """
    # 1. Reduce everyone to their absolute minimum 'essential core'
    # These are points covered by cube[i] but NOT by any other cube in 'cover'
    cores = []
    for i in range(len(cover)):
        c = cover[i]
        others = np.delete(cover, i, axis=0)
        
        # Find space in 'c' not covered by others
        q_cofactor = cofactor_cover_by_cube(others, c)
        tracker = ComplTracker()
        essential_points = complement(q_cofactor, tracker)
        
        if len(essential_points) > 0:
            # Shrink to the supercube of these points
            s_cube = supercube(essential_points)
            # Map back to the full variable space
            reduced_c = c.copy()
            for j in range(num_vars):
                if c[j] == 3: reduced_c[j] = s_cube[j]
            cores.append(reduced_c)
    
    if not cores: return cover
    
    # 2. Expand these cores. Because they are smaller, 
    # Expand has more "directions" it can grow in to try and overlap others.
    gasp_cover = better_expand(np.array(cores), off_cover)
    
    # 3. Combine with the original and let Irredundant pick the best 28
    combined = np.vstack((cover, gasp_cover))
    # Remove exact duplicates to save time
    combined = np.unique(combined, axis=0)
    
    return irredundant(combined, num_vars)


def espresso_minimizer(cover, num_vars):
    """
    The main Espresso heuristic loop.
    """
    print("  -> Generating OFF-set (Complement)...")
    compl_tracker = ComplTracker()
    off_cover = complement(cover, compl_tracker)
    print(f"     OFF-set size: {len(off_cover)} cubes.")
    
    iteration = 0
    best_cover = cover

    # Run the loop until the number of cubes stops decreasing
    while iteration < 1:

        #idx = np.random.permutation(len(best_cover))
        #cover = best_cover[idx]


        print(f"  -> Iteration {iteration}: Starting with {len(cover)} cubes")
        
        # 2. Irredundant
        print("     Running Irredundant...")
        cover = irredundant(cover, num_vars)
        print(len(cover))

        # 1. Expand
        print("     Running Expand...")
        cover = better_expand(cover, off_cover)
        print(len(cover))

        # 3. Reduce (Placeholder for full Espresso logic, usually followed by another Expand)
        print("     Running Reduce...")
        cover = reduce_step(cover)
        print(len(cover))

        # 2. Irredundant
        print("     Running Irredundant...")
        cover = irredundant(cover, num_vars)
        print(len(cover))

        print("     Running Last Gasp...")
        cover = last_gasp(cover, off_cover, num_vars)
        print(len(cover))
        
        print(f"     Finished iteration {iteration}")
        
        if len(cover) < len(best_cover):
            best_cover = cover
        else: iteration += 1
                
    return best_cover




def better_expand(cover, off_cover):
    """
    Espresso-style Expand:
    1. Tries to expand cubes to 'swallow' other cubes in the cover.
    2. Expands remaining literals to maximize the cube size.
    """
    working_cover = cover.copy()
    num_vars = working_cover.shape[1]
    
    # Sort: Smallest cubes first. 
    # Small cubes are "easier" to move and expand into larger neighbors.
    dash_counts = np.sum(working_cover == 3, axis=1)
    order = list(np.argsort(dash_counts))
    
    active_mask = np.ones(len(working_cover), dtype=bool)
    
    for i in order:
        if not active_mask[i]: continue
        
        current_cube = working_cover[i]
        
        # --- PHASE 1: Expand to Cover ---
        # Look for other cubes (j) that this cube (i) could potentially swallow.
        for j in range(len(working_cover)):
            if i == j or not active_mask[j]: continue
            
            # If we merged i and j, would it hit the OFF-set?
            candidate = get_supercube(current_cube, working_cover[j])
            if not intersects_off_set(candidate, off_cover):
                current_cube = candidate
                active_mask[j] = False # Cube j is now swallowed/redundant
        
        # --- PHASE 2: Maximal Expansion ---
        # For any remaining 0s or 1s, try to turn them into dashes.
        for v in range(num_vars):
            if current_cube[v] != 3:
                original = current_cube[v]
                current_cube[v] = 3
                if intersects_off_set(current_cube, off_cover):
                    current_cube[v] = original
                    
        working_cover[i] = current_cube
        
    return working_cover[active_mask]


# ... (Insert your parse/main logic here)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 Lowry_Clishe_Minimizer.py <espresso_file>")
        sys.exit(1)
        
    input_file = sys.argv[1]
    
    print(f"--- Espresso Logic Minimization ---")
    
    start_time = time.perf_counter()
    cover, ilb, ob = parse_espresso(input_file)
    num_vars = len(ilb) if ilb else cover.shape[1]
    
    print(f"Parsed {input_file}: {len(cover)} initial cubes.")
    
    # Run Minimizer
    minimized_cover = espresso_minimizer(cover, num_vars)
    
    end_time = time.perf_counter()
    
    print("\n--- Minimization Results ---")
    print(f"Original Cubes:  {len(cover)}")
    print(f"Minimized Cubes: {len(minimized_cover)}")
    print(f"Total Time:      {(end_time - start_time):.3f} sec")
    
    # Export
    output_file = f"minimized_{input_file.split('/')[-1]}"
    export_espresso(minimized_cover, ilb, ob, output_file)
    print(f"\nExported minimized cover to: {output_file}")