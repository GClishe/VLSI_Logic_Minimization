#To use, run the command 
#   python3 Lowry_Clishe_Minimizer.py <directory of cover file>
#An espresso file for the minimized cover will be placed in the folder Lowry_Clishe_Minimized
#Make sure this folder exists before running the script

#Test       1           2           3           4           5           6
#Target     255         28          120         495         8192        65535
#Cubes      255         28          122         499
#Time       2.241       0.249       3.482       59.63
#Bench      1           2           3           4           5           6           7           8           9           10
#Target     255         330         799         NA          6435        2943        769         4095        65535       115
#Cubes      255         333         810
#Time       1.973       9.288       15.73

#This script employs a the espresso heuristic minimizer algorithim
#It first imports the cover file as a uint8 numpy array with '0' '1' and '-' represented by 1, 2, and 3
#The minimization proceess has 3 main functions:
#1. Reduce: This funtion looks for cubes that can have '-' replaced with a '1' or '0'
#   This stops cubes from covering minterms that are already covered by other cubes, allowing the expand funtion to 
#   Have more ways to expand. To test if a '-'' can be replaced, a copy of the cube is made with the '-'' replaced by '0'
#   If the cofactor of the cover with the original cube removed with respect to the modified cube is a tautlogy
#   It means the '0' is covered by another cube and the '-' can be replaced with a '1'
#   It loops through all cubes in the cover all each variable in each cube. The order the variables are checked
#   Doesn't matter, but the order the cubes are checked does. Sorting so cubes with more dashes are sorted first
#   Helps, but going in a random order can allow the algorithim to break out of local minimums
#2. Expand: This function is the oposite of reduce, it looks for '1' and '0' and tries to replace them with '-'
#   This makes it so each cube covers as many minterms as possible to increase the chance that it is removed
#   In the irredundant step. This is done by iterating through each cube and variable in a random order
#   Theoretically going in order of least '-' is optimal, but the randomness helps avoid local minimums
#   If the variable is not a '-', make a copy of the cube, change the variable to a '-', and AND it with the off_set
#   If the result is that there is no overlap with the off_set, the expansion is valid
#3. Irredundant checks if a cube is covered by other cubes. It does this by iterating through all the cubes
#   Theoretically, going in order of last to most '-' is best since smaller cubes are more likely to be covered
#   But we did not find this to do much. Additionally, going in a random order didn't affect it either
#   For each cube, it is a sub cover is made that does not include it or any other cubes removed during this
#   Irredundant pass. If the cofactor of this sub cover with respect to the cube is a tautology, it is removed from the main cover
#The minimize function puts these three operatations together. It first removes any duplicate cubes in the cover
#It does this by casting the array to a set of tuples, then back to a numpy array
#This was found to be faster than simply using np.unique
#It then runs a modified version of the irredundant function called irredundat_init
#The only difference is that init uses a different method of building the others array
#instead of using a conditional for loop to create a list and casting it back to an np array
#It applies the keep mask to the cover, then takes that, creates another mask of any cubes that don't equal the current one
#And applies that mask. This method should produce the same result, but it doesn't
#It's not known why or how it's different, but the results are usually lower quality
#That said, it's much faster, so it's used for this initial pass since it just needs to get the size down
#Before the main loop, not neccesarilly be super optimal. The reduced cover is then used of generate a complement
#Now we start the main loop, where it does reduce, expand, irredundatn in that order. Due to the randomness, the exit condition is different
#Since it's possible to have multiple in a row that don't improve, followed by an improvement from
#One of the randomizers finding a good order, the loop ends when there are a certain number of same length in a row
#Once this value is reached, it tries one more thing before giving up
#Online it was suggested to try running the Last Gasp function at this point. This function
#Takes anything that's close to expanding and forces it to by adding cubes to the cover
#This gives it a chance to find a new solution and escape a local minimum. It saves the 
#Best result before doing this, and will either do it again or stop if it gets back to this point
#With an improvement or a regression/stagnation. The last gasp didn't ever do much though
#Often just dragging out run time to propose a worse result
#Once the algorithim runs out though, it returns and exports the cover

import numpy as np
import time
import sys
from Lowry_Clishe_TautCheck import is_tautology, parse_espresso, Tracker
from Lowry_Clishe_CompGen import complement, export_espresso, ComplTracker

def cofactor_cube(cover, cube):                                                                     #Takes the Shannon cofactor of a cover with respect to an entire cube by cofactoring the cover against each literal in the cube
    if len(cover) == 0: return cover                                                                #If the cover is empty, just return the cover
    for var_idx in range(len(cube)):                                                                #For each variable in the cube
        val = cube[var_idx]                                                                         #Get the value
        if val != 3:                                                                                #If the value isn't '-' (3)
            cover = cover[(cover[:, var_idx] & val) != 0]                                           #Keep cubes that intersect with the cofactor literal
            if len(cover) == 0: break                                                               #If the resulting cover is empty, break the loop
            cover[:, var_idx] = 3                                                                   #The cofactored variable becomes a 'Don't Care' (3) in the surviving cubes
    return cover                                                                                    #Return the cofactored cover

def reduce(cover, vars_list):                                                                       #Tries to shrink cubes by replacing '-' (3) with '0' (1) or '1' (2). Legal if tautology if the cover with the cube removed is cofactor with respect to the oposite of the proposed cube
    num_cubes, num_vars = cover.shape                                                               #Get the number of cubes and number of variables
    for i in np.random.permutation(num_cubes):                                                      #Increment through the cubes in a random order    
        others = np.delete(cover, i, axis=0)                                                        #Create a sub-cover excluding the current cube
        for v in range(num_vars):                                                                   #Increment through each variable in the cube
            if cover[i][v] == 3:                                                                    #If the value is '-' (3)
                test_cube = cover[i].copy()                                                         #Make a copy of the cube
                test_cube[v] = 1                                                                    #Test replacing '-' with '0' (1)
                if is_tautology(cofactor_cube(others, test_cube), vars_list, 0, Tracker())[0]:      #If the cofactor of the cover without original the cube with respect to the test cube is a tautology
                    cover[i][v] = 2                                                                 #The '0' (1) part of the '-' (3) is redundant, so shrink to '1' (2)
                    continue                                                                        #Skip to the next iteration variable in the cube
                test_cube[v] = 2                                                                    #If it's not, test replacing '-' with '1' (PCN: 2)
                if is_tautology(cofactor_cube(others, test_cube), vars_list, 0, Tracker())[0]:      #If the cofactor of the cover without original the cube with respect to the test cube is a tautology
                    cover[i][v] = 1                                                                 #The '1' (2) part of the '-' (3) is redundant, so shrink to '0' (1)
    return cover                                                                                    #Return the expanded cover

def expand(cover, off_set):                                                                         #Tries to grow cubes by replacing '0' (1) and '1' (2) with '-' (3). Legal if resulting cube doesn't intersect OFF-set
    num_cubes, num_vars = cover.shape                                                               #Get the number of cubes and number of variables
    for i in np.random.permutation(num_cubes):                                                      #Step through the cubes in a random order
        for v in np.random.permutation(num_vars):                                                   #Step through each variable in each cube in a random order
            if cover[i][v] != 3:                                                                    #If the variable is a '-' (3)
                test_cube = cover[i].copy()                                                         #Make a copy of the cube for testing
                test_cube[v] = 3                                                                    #Try to expand the variable to a '-' (3) in the test cube
                overlap = (off_set & test_cube) != 0                                                #Create a mask for where the test cube overlaps with the off_set
                if not np.any(np.all(overlap, axis=1)): cover[i][v] = 3                             #If there is no intersection, commit the '-' (3) to the cover                                                        
    return cover                                                                                    #Return the expanded cover

def irredundant(cover, vars_list):                                                                  #Remove any cubes covered by other cubes. This is legal if the cofactor of the cover without the cube with respect to the cube is a tautology
    num_cubes = len(cover)                                                                          #Get the number of cubes in the cover
    keep = np.ones(num_cubes, dtype=bool)                                                           #Create a list of 'True' with an entry for each cube
    for i, cube in enumerate(cover):                                                                #Iterate through all the cubes
        others = np.array([cover[j] for j in range(num_cubes) if keep[j] and j != i], dtype=np.uint8)   #Create a sub cover without the cube, and without any other cubes that were already marked for removal
        if is_tautology(cofactor_cube(others, cube), vars_list, 0, Tracker())[0]: keep[i] = False   #If the cofactor of the sub cover with respect to the cube is a tautology, mark it for removal
    return cover[keep]                                                                              #Return the irredundant cover

def irredundant_init(cover, vars_list):                                                             #Remove any cubes covered by other cubes. This is legal if the cofactor of the cover without the cube with respect to the cube is a tautology
    num_cubes = len(cover)                                                                          #Get the number of cubes in the cover
    keep = np.ones(num_cubes, dtype=bool)                                                           #Create a list of 'True' with an entry for each cube
    for i, cube in enumerate(cover):                                                                #Iterate through all the cubes
        others = cover[keep]                                                                        #Create a sub cover without any other cubes that were already marked for removal
        others = others[(others != cube).any(axis=1)]                                               #Remove the curent cube form the sub cover (can't use np.delete since indexis is funny)
        if is_tautology(cofactor_cube(others, cube), vars_list, 0, Tracker())[0]: keep[i] = False   #If the cofactor of the sub cover with respect to the cube is a tautology, mark it for removal
    return cover[keep]                                                                              #Return the irredundant cover

def minimize(cover, vars_list, max_in_a_row):
    t = time.perf_counter()
    print("Initial Irredundant Pass...") # To guarantee a baseline SCC minimal cover before the heavy lifting
    cover = np.array(list({tuple(cube) for cube in cover}), dtype=np.uint8) #Remove duplicate cubes
    in_a_row = 0
    iteration = 1
    cover = irredundant_init(cover, vars_list)
    prev_cube_count = len(cover)
    print(f"Generated initial irredundant cover with {prev_cube_count} cubes in {(time.perf_counter() - t):.3f} sec\n")

    t = time.perf_counter()
    print("Generating Complement...")
    off_set = complement(cover, 0, ComplTracker())
    print(f"Generated complement with {len(off_set)} cubes in {(time.perf_counter() - t):.3f} sec\n")

    while True:   
        print(f"Iteration {iteration}...")

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

        #Add a time buget thing where it calculates number of iterations to fill an hour
        #loop count = (budget - initial setup time) / (first loop time)
        #For now max in a row is fine, but looking at Test 4 where it gets to 499 with 5, but 496 with 31, they can always use more iterations

        if in_a_row >= max_in_a_row:
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