#To use, run the command 
#   python3 Lowry_Clishe_tautcheck.py <directory of cover file>

#Test       1, 0.036    2, 0.455    3, 195.9    4, 62.35    5, 0.010    6, 0.301    7, 14.38    8, 0.469
#Bench      1, 0.024    2, 0.248    3, 1.692    4, 0.110    5, 0.106    6, 0.054    7, 0.205    8, 0.025    9, 2.122    10, 62.39    11, 164.2
#Tatology   NO          YES         YES         NO          NO          YES         NO          NO          YES         NO             YES
#Notes      Col of 1/0              No GUR/2^n  No GUR      1x 2^n, 12x split       Col of 1/0  No Universal Cube in Unate Sub-Cover

#Originally based on Gemini2_tautcheck

import sys
import time

class Tracker:                                                                                      #Create class object to track statistics
    def __init__(self):
        self.max_depth = 0
        self.null_cover_count = 0
        self.all_unique_cubes_count = 0
        self.univeral_cube_count = 0
        self.all_1_or_0_count = 0
        self.all_unate_count = 0
        self.no_uni_in_unate_sub_count = 0
        self.unate_reduction_count = 0
        self.binate_split_count = 0
        self.null_cover_time = 0
        self.all_unique_cubes_time = 0
        self.universal_cube_time = 0
        self.find_unate_binate_time = 0
        self.all_unate_time = 0
        self.GUR_fail_time = 0
        self.GUR_time = 0
        self.binate_split_time = 0
        self.binate_split_setup_time = 0

def parse_espresso(filepath):                                                                       #Parses an Espresso file and returns the cover matrix, ilb, and ob
    cover = []                                                                                      #Create empty list for the cover
    with open(filepath, 'r') as f: lines = f.readlines()                                            #Open the file and import the lines
    for line in lines:                                                                              #Cycle through the lines
        line = line.strip()                                                                         #Extract the individual line
        if line[0] in '01-': cover.append(line.split()[0])                                          #If the line starts with 0 1 or - it's a cube. Split off the input part of the cube and append it to the cover
        elif line.startswith('.ilb '): ilb = line.split()[1:]                                       #If it's the .ild line, save everything after the .ilb as the variable names
        elif line.startswith('.ob '): ob = line.split()[1:]                                         #If it's the .ob line, save everything after the .ob as the output names
        elif line.startswith('.e'): break                                                           #If it's the .e line, it's the end of the file
    return cover, ilb, ob                                                                           #Return the cover and the variable names

def is_tautology(cover, vars, depth, tracker):                                                      #Provide the cover, the variable names, the current recursion depth, and the stat tracker object to determine tautology
    t = time.perf_counter()                                                                         #Mark start time
    tracker.max_depth = max(tracker.max_depth, depth)                                               #Update max recursion depth
    if not cover:                                                                                   #If it's a null cover
        tracker.null_cover_count += 1                                                               #Add it to the tracker
        tracker.null_cover_time += time.perf_counter() - t                                          #Add the time to the tracker
        return False, {v: 0 for v in vars}                                                          #If an empty cover is provided, it's not a tautology. Return the witness dictionary of each variable with a 0
    tracker.null_cover_time += time.perf_counter() - t                                              #If valid cover provided, add time to tracker for time spent checking
    t = time.perf_counter()                                                                         #Reset start time for next segment

    #If there are no dashes in the cover, then if the unique cube count equals 2^n it's a tautology, if it doesn't equal 2^n it's not a tautology
    n = len(vars)                                                                                   #Find the number of variables in the cover
    if not any('-' in cube for cube in cover):                                                      #If there are no dashes in the cover                                                                          
        unique_cubes = set(tuple(c) for c in cover)                                                 #Convert the cover into a set of cubes
        if len(unique_cubes) == (2**n):                                                             #If the length is of the set (number of unique cubes) is 2^n
            tracker.all_unique_cubes_count += 1                                                     #Add one to the tracker
            tracker.all_unique_cubes_time += time.perf_counter() - t                                #Add time spent to the tracker
            return True, None                                                                       #Return that the cover is a tautology                                                                                    
        for i in range(2**n):                                                                       #If the cover isn't a tautology, cycle through all possible cubes to find the missing minterm (witness)
            c_tuple = tuple('1' if ((i >> (n - 1 - j)) & 1) else '0' for j in range(n))             #Build a tuple for the possible cube
            if c_tuple not in unique_cubes:                                                         #See if it's not in the set of unique cubes (one eventually won't be)
                wit = {vars[j]: 1 if c_tuple[j] == '1' else 0 for j in range(n)}                    #Reorganize the tuple to be in a dictionary object of all the variables
                tracker.all_unique_cubes_count += 1                                                 #Add one the tracker
                tracker.all_unique_cubes_time += time.perf_counter() - t                            #Add time spent to the timer
                return False, wit                                                                   #Return that the cover isn't a tautology with the witness dictionary
    tracker.all_unique_cubes_time += time.perf_counter() - t                                        #If there are dashes in the cover, add time spend checking to the tracker
    t = time.perf_counter()                                                                         #Restart time for next segment

    #If there is a universal cube, the cover is a tautology
    for cube in cover:                                                                              #Increment through all cubes in the cover
        if all(val == '-' for val in cube):                                                         #If all of it's values are -
            tracker.univeral_cube_count += 1                                                        #Add one to the tracker
            tracker.universal_cube_time += time.perf_counter() - t                                  #Add time spent figuring that out to the tracker
            return True, None                                                                       #Return that the cover is a tautology
    tracker.universal_cube_time += time.perf_counter() - t                                          #If not, add time spent figuring it out to the tracker
    t = time.perf_counter()                                                                         #Reset time for next segment

    #Find unate and binate columns. If any pure unate columns exist, it's not a tautology. Faster to do this check during unate / binate column identification but it makes time reporting a bit ambiguous
    cover_column = tuple(zip(*cover))                                                               #Create a version of the cover that is indexed by column
    cover_column_sets = tuple(frozenset(col) for col in cover_column)                               #Create a set of what symbols are in the column

    unate_cols = []                                                                                 #Create an empty list for unate columns
    binate_cols = []                                                                                #Create an empty list for binate columns
    u_append = unate_cols.append                                                                    #Alias append since it's referenced often in a loop
    b_append = binate_cols.append                                                                   #Alias append since it's referenced often in a loop
    for i, column_set in enumerate(cover_column_sets):                                              #Iterate through each column in the cover
        if '0' not in column_set:                                                                   #If the column is positive unate (no 0's)
            u_append(i)                                                                             #Add it to the unate list
            if '-' not in column_set:                                                               #If it also doesn't have any - it's pure unate
                tracker.all_1_or_0_count += 1                                                       #Add it to the tracker
                wit = {v: 0 for v in vars}                                                          #Make an empty witness dictionary
                wit[vars[i]] = 0                                                                    #Make the variable associated with this column a 0 in the witness cube
                tracker.find_unate_binate_time += time.perf_counter() - t                           #Add time spent checking this to tracker
                return False, wit                                                                   #Return that the cover is not a tautology
        elif '1' not in column_set:                                                                 #If the column is negative unate (no 1's)
            u_append(i)                                                                             #Add it to the unate list
            if '-' not in column_set:                                                               #If it also doesn't have any - it's pure unate
                tracker.all_1_or_0_count += 1                                                       #Add it to the tracker
                wit = {v: 0 for v in vars}                                                          #Make an empty witness dictionary
                wit[vars[i]] = 1                                                                    #Make the variable associated with this column a 1 in the witness cube
                tracker.find_unate_binate_time += time.perf_counter() - t                           #Add time spent checking this to the tracker
                return False, wit                                                                   #Return that the cover is not a tautology
        else: b_append(i)                                                                           #Add column to binate list
    tracker.find_unate_binate_time += time.perf_counter() - t                                       #If no pure unate columns were found, add time spent checking to the tracker
    t = time.perf_counter()                                                                         #Reset time for next segment

    #If all columns are unate and there isn't a univeral cube, it's not a tautology. Since universal cube check happened first, it can be skipped here
    if len(binate_cols) == 0:                                                                       #See if all the columns are unate (no binate ones)
        tracker.all_unate_count += 1                                                                #Add the instance to the tracker
        wit = {}                                                                                    #Create an empty dictionary object for the witness
        for i in range(n):                                                                          #Cycle through all the columns
            wit[vars[i]] = 0 if '1' in cover_column_sets[i] else 1                                  #If there's a 1 in the column value set, that variable is a 0, otherwise it's a 1
        tracker.all_unate_time += time.perf_counter() - t                                           #Add time spent to tracker
        return False, wit                                                                           #Return that the cover is not a tautology
    tracker.all_unate_time += time.perf_counter() - t                                               #If check fails, add time spent to tracker
    t = time.perf_counter()                                                                         #Reset time for next segment

    #General Unate Reduction organizes the columns such that unate are on the left, and any universal cubes within the unate sub-cover are on the bottom. The bottom right corner (F2) can determine tautology  
    if len(unate_cols) > 0:                                                                         #If there are unate columns
        reduced_cover = []                                                                          #Create list for reduced cover (F2)
        for cube in cover:                                                                          #Iterate through all the cubes
            if all(cube[i] == '-' for i in unate_cols):                                             #Look for rows where all unate variables are -
                reduced_cover.append([cube[i] for i in binate_cols])                                #Add the non unate variables of these rows to the reduced cover
        if len(reduced_cover) == 0:                                                                 #If there are no cubes in this sub-cover, there are no universal cubes in the unate sub-cover
            tracker.no_uni_in_unate_sub_count += 1                                                  #Add this to the tracker
            wit = {v: 0 for v in vars}                                                              #Create witness dictionary of all 0's
            for i in unate_cols:                                                                    #Go through all the unate variables
                wit[vars[i]] = 0 if '1' in cover_column_sets[i] else 1                              #If there's a '1' set the variable in the witness dictionary to 0. If '1' or '-', set it to 1
            tracker.GUR_fail_time += time.perf_counter() - t                                        #Add time taken to the tracker
            return False, wit                                                                       #Return that the cover was not a tautology
        else:                                                                                       #If there are cubes in the sub-cover
            tracker.GUR_fail_time += time.perf_counter() - t                                        #Log the time used to attempt. This is unflated as it includes setup for GUR
            t = time.perf_counter()                                                                 #Reset time for main GUR segment
            cur_null_cover_time = tracker.null_cover_time                                           #Since GUR can insite a recursion, log the current values of timers to subtract out and isolate GUR time
            cur_all_unique_cubes_time = tracker.all_unique_cubes_time
            cur_universal_cube_time = tracker.universal_cube_time
            cur_find_unate_binate_time = tracker.find_unate_binate_time
            cur_all_unate_time = tracker.all_unate_time
            cur_GUR_time = tracker.GUR_time
            cur_GUR_fail_time = tracker.GUR_fail_time
            cur_binate_split_time = tracker.binate_split_time
            tracker.unate_reduction_count += 1                                                      #Increment tracker
            new_vars = [vars[i] for i in binate_cols]                                               #Create new variable tag list that just contains the variables in the reduced cover
            res, sub_wit = is_tautology(reduced_cover, new_vars, depth + 1, tracker)                #Check the reduced cover for tautology
            if res:                                                                                 #If the result is that it is a tautology, track time
                tracker.GUR_time += time.perf_counter() - t + (cur_null_cover_time - tracker.null_cover_time) + (cur_universal_cube_time - tracker.universal_cube_time) + (cur_all_unique_cubes_time - tracker.all_unique_cubes_time) + (cur_find_unate_binate_time - tracker.find_unate_binate_time) + (cur_all_unate_time - tracker.all_unate_time) + (cur_GUR_fail_time - tracker.GUR_fail_time) + (cur_GUR_time - tracker.GUR_time) + (cur_binate_split_time - tracker.binate_split_time)
                return True, None                                                                   #And return that the parent cover is also a tautology
            else:                                                                                   #If it's not a tautology
                wit = {}                                                                            #Create an empty dictionary for the witness
                for i in range(n):                                                                  #Iterate through all the variables
                    if i in unate_cols:                                                             #For the unate columns
                        wit[vars[i]] = 0 if '1' in cover_column_sets[i] else 1                      #If the set has a '1' set that variable in the witness to a 0. 1 otherwise
                    else:                                                                           #If it's not a unate column
                        wit[vars[i]] = sub_wit[vars[i]]                                             #Use the witness from the sub cover
                tracker.GUR_time += time.perf_counter() - t + (cur_null_cover_time - tracker.null_cover_time) + (cur_universal_cube_time - tracker.universal_cube_time) + (cur_all_unique_cubes_time - tracker.all_unique_cubes_time) + (cur_find_unate_binate_time - tracker.find_unate_binate_time) + (cur_all_unate_time - tracker.all_unate_time) + (cur_GUR_fail_time - tracker.GUR_fail_time) + (cur_GUR_time - tracker.GUR_time) + (cur_binate_split_time - tracker.binate_split_time)
                return False, wit                                                                   #Track the time, return that the parent cover isn't a tautology
    tracker.GUR_time += time.perf_counter() - t                                                     #If there were no unate columns, track the time spent figuring it out
    t = time.perf_counter()                                                                         #Reset the time for the next segment

    #If all else fails, execute a binate split. This is where the cover is split into positive and negative cofactors. If either isn't a tautology, neither is the parent
    cur_null_cover_time = tracker.null_cover_time                                                   #Since this process insites a recursion, log the current values of times to subtract out and isolate the split time
    cur_all_unique_cubes_time = tracker.all_unique_cubes_time
    cur_universal_cube_time = tracker.universal_cube_time                                           #Instead of picking the most binate variable to minimize overlap in cofactors
    cur_find_unate_binate_time = tracker.find_unate_binate_time                                     #Find variable with least '-'. Not strictly most binate column
    cur_all_unate_time = tracker.all_unate_time                                                     #But just as good at minimizing repeat cubes in cofactors
    cur_GUR_time = tracker.GUR_time                                                                 #Not as good at keeping the cofactors even, but it's way faster to calculate
    cur_GUR_fail_time = tracker.GUR_fail_time
    cur_binate_split_time = tracker.binate_split_time
    tracker.binate_split_count += 1                                                                 #Increment the tracker for binate splits
    min_dashes = len(cover)                                                                         #Initialize min dahses as max possible value
    for i in binate_cols:                                                                           #Iterate through the binate columns
        score = cover_column[i].count('-')                                                          #Count how many '-' are present
        if score < min_dashes:                                                                      #If it's less than the min known
            min_dashes = score                                                                      #Record as new best
            best_col = i                                                                            #And note which column it was
    split_var = vars[best_col]                                                                      #Get the variable name associated with the best column
    new_vars = [vars[i] for i in range(n) if i != best_col]                                         #Create a new set of variable names without it
    cofactor_1 = [cube[:best_col] + cube[best_col+1:] for cube in cover if cube[best_col] in ('1', '-')]    #Create the positive cofactor (x = 1)
    res1, wit1 = is_tautology(cofactor_1, new_vars, depth + 1, tracker)                             #Pass it recursively to the tautology checker
    if not res1:                                                                                    #If it's not a tautology
        wit1[split_var] = 1                                                                         #The witness is simply anything with the best variable as 1
        tracker.binate_split_time += time.perf_counter() - t + (cur_null_cover_time - tracker.null_cover_time) + (cur_universal_cube_time - tracker.universal_cube_time) + (cur_all_unique_cubes_time - tracker.all_unique_cubes_time) + (cur_find_unate_binate_time - tracker.find_unate_binate_time) + (cur_all_unate_time - tracker.all_unate_time) + (cur_GUR_fail_time - tracker.GUR_fail_time) + (cur_GUR_time - tracker.GUR_time) + (cur_binate_split_time - tracker.binate_split_time)
        return False, wit1                                                                          #Return that the partent cover is not a tautology and add time to the tracker
    cofactor_0 = [cube[:best_col] + cube[best_col+1:] for cube in cover if cube[best_col] in ('0', '-')]    #Create the negative cofactor (x = 0)
    res0, wit0 = is_tautology(cofactor_0, new_vars, depth + 1, tracker)                             #Pass it recursively to the tautology checker
    if not res0:                                                                                    #If it's not a tautology
        wit0[split_var] = 0                                                                         #The witness is simply anything with the best variable as 1
        tracker.binate_split_time += time.perf_counter() - t + (cur_null_cover_time - tracker.null_cover_time) + (cur_universal_cube_time - tracker.universal_cube_time) + (cur_all_unique_cubes_time - tracker.all_unique_cubes_time) + (cur_find_unate_binate_time - tracker.find_unate_binate_time) + (cur_all_unate_time - tracker.all_unate_time) + (cur_GUR_fail_time - tracker.GUR_fail_time) + (cur_GUR_time - tracker.GUR_time) + (cur_binate_split_time - tracker.binate_split_time)
        return False, wit0                                                                          #Return that the partent cover is not a tautology and add time to the tracker
    tracker.binate_split_time += time.perf_counter() - t + (cur_null_cover_time - tracker.null_cover_time) + (cur_universal_cube_time - tracker.universal_cube_time) + (cur_all_unique_cubes_time - tracker.all_unique_cubes_time) + (cur_find_unate_binate_time - tracker.find_unate_binate_time) + (cur_all_unate_time - tracker.all_unate_time) + (cur_GUR_fail_time - tracker.GUR_fail_time) + (cur_GUR_time - tracker.GUR_time) + (cur_binate_split_time - tracker.binate_split_time)
    return True, None                                                                               #If both passed, the parent is a tautology

def export_witness(witness_dict, ilb, ob, output_filepath):                                         #Create a witness cube and espresso file
    with open(output_filepath, 'w') as f:                                                           #Create an empty file at the file path
        f.write(f".i {len(ilb)}\n")                                                                 #Add .i line with number of variables
        f.write(f".o {len(ob)}\n")                                                                  #Add .ob line with number of outputs
        f.write(".p 1\n")                                                                           #Add .p line with number of cubes (just 1 for a witness file)
        f.write(f".ilb {' '.join(ilb)}\n")                                                          #Add .ilb line with variable names
        f.write(f".ob {' '.join(ob)}\n")                                                            #Add .ob line with output names
        cube_str = "".join(str(witness_dict[var]) for var in ilb)                                   #Build the cube string in the order of original .ilb by using the provided dictionary object
        f.write(f"{cube_str} 1\n")                                                                  #Add cube line
        f.write(".e\n")                                                                             #Add end line
    return cube_str                                                                                 #Return the cube string

input_file = sys.argv[1]                                                                            #Get input file from terminal
cover, ilb, ob = parse_espresso(input_file)                                                         #Parse the file into a cover
tracker= Tracker()                                                                                  #Create tracker object
start_time = time.perf_counter()                                                                    #Start timer for tautology checking
is_taut, witness = is_tautology(cover, ilb, 0, tracker)                                             #Start tautology checking

print(f"\n--- Results for {input_file} ---\n")                                                      #Print results
print(f"Is Tautology?: {'YES' if is_taut else 'NO'}")        
print("\n--- Algorithm Statistics ---")
print(f"Cover Is Null:                             {tracker.null_cover_count} times in {tracker.null_cover_time:.3f} sec")
print(f"No Dashes Check For 2^n Unique Cubes:      {tracker.all_unique_cubes_count} times in {tracker.all_unique_cubes_time:.3f} sec")
print(f"Found Universal Cube:                      {tracker.univeral_cube_count} times in {tracker.universal_cube_time:.3f} sec")
print(f"Found Column of All 1's or All 0's:        {tracker.all_1_or_0_count} times in {tracker.find_unate_binate_time:.3f} sec")
print(f"All Columns Unate Without Universal Cube:  {tracker.all_unate_count} times in {tracker.all_unate_time:.3f} sec")
print(f"No Universal Cube in Unate Sub-Cover:      {tracker.no_uni_in_unate_sub_count} times in {tracker.GUR_fail_time:.3f} sec")
print(f"Unate Reduction:                           {tracker.unate_reduction_count} times in {tracker.GUR_time:.3f} sec")
print(f"Binate Split:                              {tracker.binate_split_count} times in {tracker.binate_split_time:.3f} sec")
print(f"Max Recursive Layers:                      {tracker.max_depth} layers")
print(f"Time to Check Tautology:                   {(time.perf_counter() - start_time):.3f} sec\n")

print(tracker.binate_split_setup_time)
if not is_taut:                                                                                     #If not a tautology, generate witness report 
    start_time = time.perf_counter()
    print("--- Witness Information ---")
    output_file = f"Lowry_Clishe_Tautology_Witnesses/{input_file.split('/')[-1]}_off_cube"
    print(f"Witness Cube:             {export_witness(witness, ilb, ob, output_file)}")
    print(f"Witness file exported to: {output_file}")
    print(f"Time to Generate Witness: {(time.perf_counter() - start_time):.3f} sec\n")