#To use, run the command 
#   python3 Lowry_Clishe_CompGen.py <directory of cover file>
#An espresso file for the complement cover will be placed in the folder Lowry_Clishe_Complement_Covers
#Make sure this folder exists before running the script

#Bench      1, 0.773    2, 0.291    3, 0.543    4, 2.027    5, 3.215    6, 0.088    7, 3.392    8, 5.540    9, 2.177    10, 288.4    11, 105.8
#Cubes      5609        0           0           58          24313       0           711642      1426983     0           89375        0

#This is a relatively simple espresso complement generator. It expresses '0', '1', and '-' to leverage the uint8 data type
#And exploit bitwise operations. It starts be importing the cover to a numpy array
#It then checks if the cover is empty, in which case it simply returns a cube of all '-'
#If the cover has just one cube, it does a simple DeMorgan complement
#It then checks if the cover is a tautology since tautologies would return an empty cover
#If all the special cases fail, it picks the most binate variable by selecting the column with the least '-'
#Strictly speaking this isn't the most binate, but it is faster to calculate and
#Will produce the least total cubes in the resulting cofactors, even if they aren't ballanced
#It then cofactors the cover based on the selected variable
#For some reason deleting the selected variable from the cofactors didn't work like in the tautology cheker
#To work arround this, the selected variable is just set to all '-' in the cofactor
#The two cofactors are then recursicely complemented
#Then to satisfy F' = x * F(x)' + x' * F(x')' the cofactors are AND with the original variable where all values are 1 or 0 then vstacked for the OR
#The unate complement steps seemed unneccesary considering most of the execution time is tautology checking as is
#I tried an SCC minimal function but it took forever and didn't ever seem to reduce the cover 
#in the tests that ran in a reasonable amount of time, so I'm assuming the covers are SCC minimal on their own somehow

from Lowry_Clishe_TautCheck import is_tautology, parse_espresso, Tracker
import time
import sys
import numpy as np

class ComplTracker:                                                                                 #Create class object to track statistics
    def __init__(self):
        self.max_depth = 0
        self.empty_array_time = 0
        self.single_cube_time = 0
        self.tautology_time = 0
        self.cofactor_time = 0
        self.empty_array_count = 0
        self.single_cube_count = 0
        self.tautology_count = 0
        self.cofactor_count = 0

def complement_single_cube(cube, num_vars):                                                         #Use DeMorgan's law to complement a cube
    comp_cover = []                                                                                 #Create an empty list
    for i in range(num_vars):                                                                       #For each variable in the cube
        if cube[i] == 1:                                                                            #If the value is '0' (1)
            new_cube = np.full(num_vars, 3)                                                         #Create a new cube of all '-' (3)
            new_cube[i] = 2                                                                         #Fill in the variable with a '1' (2)
            comp_cover.append(new_cube)                                                             #Append the new cube to the cover list
        elif cube[i] == 2:                                                                          #If the value is '1' (2)
            new_cube = np.full(num_vars, 3)                                                         #Create a new cube of all '-' (3)
            new_cube[i] = 1                                                                         #Fill in the variable with a '0' (1)
            comp_cover.append(new_cube)                                                             #Append the new cube to the cover list
    if comp_cover: return np.array(comp_cover, dtype=np.uint8)                                      #If there's stuff in the new cover, return it as an np array
    return np.empty((0, num_vars), dtype=np.uint8)                                                  #If the array is empty, it means the cube was all '3', return a null cube

def cofactor(cover, var_idx, val_pcn):                                                              #Create a cofactor of a given cover
    cover = cover[(cover[:, var_idx] & val_pcn) != 0]                                               #Bitwise AND each val in the cofactor variable with the cofacotr value. Anything that's not null survives
    if len(cover) > 0:                                                                              #For surviving cubes
        cover[:, var_idx] = 3                                                                       #The splitting variable becomes all '-' (3) (In the tautology checker I remove the variable, not sure why but that didn't work here)
    return cover                                                                                    #Return the cofactored sub cover

def complement(cover, depth, compl_tracker):                                                        #Complement the cover
    t = time.perf_counter()                                                                         #Start timer
    compl_tracker.max_depth = max(compl_tracker.max_depth, depth)                                   #Update max recursion depth
    num_cubes, num_vars = cover.shape                                                               #Get the number of cubes and variables in the cover
    if num_cubes == 0:                                                                              #If the cover has no cubes
        compl_tracker.empty_array_count += 1                                                        #Add an instance to the tracker
        cover = np.array([np.full(num_vars, 3)])                                                    #Make the cover a universal cube
        compl_tracker.empty_array_time += time.perf_counter() - t                                   #Increment the time
        return cover                                                                                #Return the complemented cover
    compl_tracker.empty_array_time += time.perf_counter() - t                                       #If not, add time it took to figure it out

    t = time.perf_counter()                                                                         #Reset timer
    if len(cover) == 1:                                                                             #If the cover has 1 cube
        cover = complement_single_cube(cover[0], num_vars)                                          #use DeMorgan's law to make the complement
        compl_tracker.single_cube_count += 1                                                        #Add one to the tracker
        compl_tracker.single_cube_time += time.perf_counter() - t                                   #Increment the time
        return cover                                                                                #Return the complemented cover
    compl_tracker.single_cube_time += time.perf_counter() - t                                       #If not, record time it took to figure it out

    t = time.perf_counter()                                                                         #Reset timer
    if is_tautology(cover, list(range(num_vars)), 0, Tracker())[0]:                                 #If the cover is a tautology
        compl_tracker.tautology_count += 1                                                          #Add an instance of tautology checking to the tracker
        cover = np.empty((0, num_vars), dtype=np.uint8)                                             #Create an empty array to return
        compl_tracker.tautology_time += time.perf_counter() - t                                     #Increment the timer
        return cover                                                                                #Return the complemented cover
    compl_tracker.tautology_time += time.perf_counter() - t                                         #If not, add the time it took to figure it out
    
    t = time.perf_counter()                                                                         #Reset the timer
    compl_tracker.cofactor_count +=1                                                                #Increment the tracker
    var_idx = np.argmin(np.sum(cover == 3, axis=0))                                                 #Get the column with the fewest '-' (3) as a stand in for most binate
    cofactor_1 = cofactor(cover, var_idx, 2)                                                        #Cofactor against '1' (2)
    cofactor_0 = cofactor(cover, var_idx, 1)                                                        #Cofactor against '0' (1)
    compl_tracker.cofactor_time += time.perf_counter() - t                                          #Increment timer

    comp_1 = complement(cofactor_1, depth + 1, compl_tracker)                                       #Recurse with cofactor 1
    comp_0 = complement(cofactor_0, depth + 1, compl_tracker)                                       #Recurse with cofactor 0

    if len(comp_1) > 0: comp_1[:, var_idx] = 2                                                      #Cofactor 1 AND x = 1
    if len(comp_0) > 0: comp_0[:, var_idx] = 1                                                      #Cofactor 0 AND x = 0
        
    return np.vstack((comp_1, comp_0))                                                              #Return comp_1 + comp_0

def export_espresso(cover, ilb, ob, filepath):                                                      #Export resulting cover in espresso format
    with open(filepath, 'w') as f:                                                                  #Create an epty file at the file path
        f.write(f".i {len(ilb)}\n")                                                                 #Add .i line with number of variables
        f.write(f".o {len(ob)}\n")                                                                  #Add .ob line with number of outputs
        f.write(f".p {len(cover)}\n")                                                               #Add .p line with number of cubes
        if ilb: f.write(f".ilb {' '.join(ilb)}\n")                                                  #Add .ilb line with variable names
        if ob: f.write(f".ob {' '.join(ob)}\n")                                                     #Add .ob line with output names
        for cube in cover:                                                                          #For each cube in the cover
            cube_str = "".join(['0' if v == 1 else '1' if v == 2 else '-' for v in cube])           #Create cube line, translate form 1, 2, 3, to '0' '1' '-'
            f.write(f"{cube_str} 1\n")                                                              #Add cube line
        f.write(".e\n")                                                                             #Add end line

if __name__ == "__main__":
    start_time = time.perf_counter()
    input_file = sys.argv[1]
    cover, ilb, ob = parse_espresso(input_file)
    cover = np.array(list({tuple(cube) for cube in cover}), dtype=np.uint8)
    print(f"\nParsed {input_file} containing {len(cover)} cubes in {(time.perf_counter() - start_time):.3f} sec")

    start_time = time.perf_counter()
    compl_tracker = ComplTracker()
    comp_cover = complement(cover, 0, compl_tracker)
    
    print(f"\n--- Complementation Results ---")
    print(f"Original Cubes: {len(cover)}")
    print(f"Complement Cubes (SCC Minimal): {len(comp_cover)}")
    print("\n--- Algorithm Statistics ---")
    print(f"Cover Is Empty:                 {compl_tracker.empty_array_count} times in {compl_tracker.empty_array_time:.3f} sec")
    print(f"Cover Was a Single Cube:        {compl_tracker.single_cube_count} times in {compl_tracker.single_cube_time:.3f} sec")
    print(f"Cover Was a Tautology:          {compl_tracker.tautology_count} times in {compl_tracker.tautology_time:.3f} sec")
    print(f"Complement Via Cofactor:        {compl_tracker.cofactor_count} times in {compl_tracker.cofactor_time:.3f} sec")
    print(f"Max Recursion Depth:            {compl_tracker.max_depth}")
    print(f"Time to Generate Complement:    {(time.perf_counter() - start_time):.3f} sec\n")

    start_time = time.perf_counter()
    output_file = f"Lowry_Clishe_Complement_Covers/{input_file.split('/')[-1]}_compl"
    export_espresso(comp_cover, ilb, ob, output_file)
    print(f"Exported complement to {output_file} in {(time.perf_counter() - start_time):.3f} sec\n")