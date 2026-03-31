# The logic minimization and tautology check benchmarks and tests all use the ESPRESSO PLA format. 
# The line denoted .i specifies the number of input variables,
# the line denoted .o specifies the number of output variables (all 1 in this case),
# the line denoted .ilb specifies the labels for all of the input variables,
# the line denoted .ob specifies the labels for all of the output variables,
# the line denoted .p specifies the number of cubes that follows. 

# Each line consists of an input and output pattern. 
# The input symbols can be 0 (low), 1(high), and -(dont care). 
# The output symbols can be 0 (false) or 1 (true), though all outputs are 1 for the benchmarks and tests that 
# we have. ".e" marks the end of a file. 

# The majority of three function (F = ab + ac + ba) can be described with the two ESPRESSO files in the Examples/ folder.


# Each cube can be reprsented by a bit vector, which will simplify the process dramatically. In such a bit vector, 
# a 0 can be represented by the two bits "01", a 1 can be represented by the bits "10", a dont-care (-) can be represented
# by "11", and a void can be represented by a "00". If a "00" occurs, then the entire cube is annihilated. 

# Using the above bit vector representation, the following operations can be defined: first, the intersection of two cubes
# can be accomplished with a bitwise AND between the bits in the vectors. For example, consider the following cubes:
#       A: 1 - 0                (ac')
#       B: - 1 0                (bc')
# The intersection of these two cubes would be: 
#       C = A int B = 1 1 0     (abc')
# If instead we used the bit vector representation, we get the following:
#       A: 10 11 01
#       B: 11 10 01
#       ____________
#       C: 10 10 01       <---- (110; abc')
# Where we have used bitwise AND to compute C. 

from bitarray import bitarray
import numpy as np
from itertools import chain     # yields elements from multiple iterables sequentially without having to copy or materialize them in any way. This is specificaly helpful in column_check()
import math

class Cube():
    def __init__(self, vars: str) -> None:
        self.bitarr = bitarray()    # allows Cube object to be represented in the form "101001111110" (a bit array)
        self.cube = vars            # allows Cube object to be represented in the form "110--1"       (same format as shown in ESPRESSO)

        for var in vars:
            if var not in ("1", "0", "-"):
                raise ValueError(f"Invalid symbol. Expected \"0\", \"1\", or \"-\". Got {var}.")
            if var == "1":
                self.bitarr.extend([1,0])
            elif var == "0":
                self.bitarr.extend([0,1])
            elif var == "-":
                self.bitarr.extend([1,1])

    def __repr__(self) -> str:
        # if a user wants to print a cube object, then str(self.bitarr) will be printed
        return self.cube
    
    def __getitem__(self, idx):
        """Allows cube to be subscriptable. Returns either '1', '0', or '-'."""
        return self.cube[idx]
    
    def __and__(self, other):
        """
        Allows bitwise AND with the & operator between Cube and Cube or Cube and bitarray. Other operand types are not and will not be supported.
        Bitwise AND between two cubes is equivalent to the intersection operation between two cubes.
        """
        if isinstance(other, Cube):
            return self.bitarr & other.bitarr
        elif isinstance(other, bitarray):
            return self.bitarr & other
        else:
            raise TypeError(f"Unsupported operand type for &: 'Cube' and {type(other)}.")
        
    def __or__(self, other):
        """Allows bitwise OR with the | operator between Cube and Cube or Cube and bitarray. Other operand types are not and will not be supported."""
        if isinstance(other, Cube):
            return self.bitarr | other.bitarr
        elif isinstance(other, bitarray):
            return self.bitarr | other
        else:
            raise TypeError(f"Unsupported operand type for |: 'Cube' and {type(other)}.")
        
    def is_null(self) -> bool:
        """Returns true if any of the elements in self.bitarr are '00', which indicates a null cube that will be annihilated on SCC minimization."""
        # the approach is to leverage the fact that we have bitarrays, so instead of looping I want to do bitwise operations

        return ((~self.bitarr[::2]) & ( ~self.bitarr[1::2])).any() # ((grab even indices, then invert) bitwise AND (grab odd indices, then invert )) check if resulting array has any 1s. If so, then self.bitarr has at least one '00' element

    def pop(self, idx):
        """Removes the character at idx from self.cube and returns the value that was stored there"""
        val = self.cube[idx]
        self.cube = self.cube[:idx] + self.cube[idx+1:] # constructs new cube by skipping the character at idx (strings are immutable, so we have to construct a new one and assign it back to self.cube)
        return val
        
    def contains(self, other) -> bool:
        """
        Returns true if self contains other. For example, '-10' contains '110' and '010'. In other words,
        returns true only if other is a subset of self.
        """
        return (other | self.bitarr) == self.bitarr
    
    def num_DC(self) -> int:
        """Returns number of '-' in the cube"""
        return self.cube.count("-")

    def size(self) -> int:
        """Returns the number of variables present in the cube"""
        return len(self.cube)

class Cover():
    def __init__(self, *cubes: Cube):
        # I considered the primary container being a numpy array. But the problem is that np arrays are contiguous in memory, so 
        # appending elements is expensive (requires duplicating entire array, see the documentation for numpy.ndarray). Since I 
        # expect to do many appends and deletions, a traditional python list is attractive. 
        self.cover = []
        for cube in cubes:
            self.cover.append(cube)
    def __repr__(self):
        return str(self.cover)
    
    def __getitem__(self, idx: int) -> Cube:
        """Allows the Cover object to be subscriptable. In other words, cover[0] will not throw an error."""
        return self.cover[idx]
    
    def __iter__(self):
        """
        Allows Cover objects to be iterable. Not technically necessary, since we do have the __getitem__ method, 
        but it is more efficient to iterate through a list with an iterator than with subscripting. At least I assume
        so, since the iterator does not have to check for index out of bounds errors.
        """
        return iter(self.cover)

    def add(self, cube: Cube) -> None:
        """Adds a cube to the end of the cover list"""
        if not isinstance(cube, Cube):      
            raise TypeError(f"Expected Cube, got {type(cube)}")
        self.cover.append(cube)

    def pop(self, idx):
        """Pops cube at idx from the cover"""
        return self.cover.pop(idx)

    def union(self, other):
        """Returns the union (OR) of two covers. Note that the union of two covers is not necessarily a minimal cover, since it may contain redundant cubes."""
        if not isinstance(other, Cover):
            raise TypeError(f"Unsupported operand type for union: 'Cover' and {type(other)}.")
        new_cover = Cover(*self.cover) # creates a new cover with the same cubes as self
        for cube in other.cover:
            new_cover.add(cube)
        return new_cover
    
    def intersection(self, other):
        """Returns the intersection (AND) of two covers. Obtained via pairwise AND """
        if not isinstance(other, Cover):
            raise TypeError(f"Unsupported operand type for union: 'Cover' and {type(other)}.")
        new_cover = Cover()
        for cube1 in self.cover:
            for cube2 in other:
                new_cover.add(cube1 & cube2)

    def get_columns(self):
        """
        Returns zip object containing corresponding entries in each column. For example, consider the following cover:
        F = ['1-0', '0-1']. get_columns() would return a zip object that, when unizpped, would
        return [('1','0'), ('-','-'), ('0','1')]. The get_columns() function is O(1), since zip returns an iterable. Actually
        unzipping it/iterating through it would be O(n x m), where n is the number of rows and m is the length of each row. 
        """
        return zip(*(cube.cube for cube in self.cover))
    
    def unate_columns(self): 
        """Returns a list of all unate columns (positive or negative) in the cover. """
        columns = self.get_columns()
        unate_cols = []

        for idx, column in enumerate(columns):
            column_it = iter(column)   # converts the column into an iterable
           
            for first in column_it:    # iterate through column_it until we find a non-dont-care. That value will be assigned to `first`
                if first != '-':
                    break
            else:   # this line only executes if the for loop terminates without breaking
                #print(f"Column {idx} is all dont-care's. Adding to list of unate columns")
                unate_cols.append(idx)
                continue
            
            # loop thru the remaining values in the column (note that since column_it is an iterable, this loop will not evaluate the values we've already looked at)
            for value in column_it:
                if value != first and value != '-':
                    break
            else:
                unate_cols.append(idx)

        return unate_cols

    def size(self):
        """Returns the number of cubes in the cover"""
        return len(self.cover)

    def complement(self):
        """see week 9 notes on how to implement"""



def SCC_Minimize(cover: Cover):
    """Performs SCC minimization on input Cover. Does not modify cover; instead, it builds a new, minimized one."""
    new_cover = Cover()
    rejected_cubes = set() # stores indices of cubes in cover that will not be included in the new cover
    for cube_idx in range(cover.size()):   
        curr_cube = cover[cube_idx]     

        # Do not add null cubes
        if curr_cube.is_null():
            rejected_cubes.add(cube_idx)
            continue
        
        # Reject cubes contained by other cubes
        flag = 0
        for cube2 in cover:
            if cube2.contains(curr_cube):
                rejected_cubes.add(cube_idx)
                flag = 1
                break
        if flag == 1:
            continue

        new_cover.add(curr_cube)
    
    return new_cover

def column_check(columns_zip):
    """
    Returns True if the columns_zip (expected to be a zip object from the Cover.get_columns() method) object
    has any columns with all 0s or all 1s. If a column contains '-', then it does not qualify.
    """
    
    # First, we add a check to make sure that columns_zip is not empty, since if it was empty, it would result in a False return which might 
    # mislead you into thinking there are no pure 1/0 columns in the cover when there actually are. This would occur if you accidentally
    # unpack the columns_zip object *before* passing it into this function. This did happen to me and the bug took a while to track down,
    # Hence the additional check.

    # As an example to illustrate that bug, consider this code:
    # cover = Cover()
    # cover.add(Cube("100"))
    # cover.add(Cube("001"))
    # cover.add(Cube("-0-"))
    # print(cover)
    # columns = cover.get_columns()     
    # print(list(columns))
    # print(column_check(columns))

    # We expect that code to print first the cover, then the columns, then True, since the second column is all 0s. However, this code actually returns False at the end.
    # The problem is because when we print(list(columns)), the list(columns) would exhaust the iterator, so that when we do column_check(columns), we are passing an 
    # *empty* iterator to column_check, which would skip the for loop and immediately return False. That is why it is necessary to raise an error if an empty columns_zip
    # is passed to this function. 
    
    
    it = iter(columns_zip)

    try:
        first_column = next(it)
    except StopIteration:
        raise ValueError("columns_zip is empty. Did you exhaust the columns_zip iterator already?")

    # chain(iterable, iterator) allows us to loop through the iterable first normally (in this case, a 1 element tuple containing first_column), and then through an iterator without unpacking it. Without chain, we'd
    # have to loop through something like this tuple: (first_column, *columns_zip), which would be bad because it would first unpack the entire columns_zip iterator before starting the loop, which is inefficient.
    for column in chain((first_column,),columns_zip):
        column_iterable = iter(column)  # converts column into an iterable
        first = next(column_iterable)   # assigns first to the very first value in the iterable
        if first == '-': 
            continue

        # below I am using a for...else statement, which is pretty rare. Basically, if the for loop encounters the break satement, then the else statement will not execute. If instead the for loop completes without hitting the break statement, then the else statement will execute.
        for x in column_iterable: # loops through the rest of the elements, stopping early and returning False if x == first is ever violated
            if x != first:
                break
        else: # if the loop completes without hitting a break statement, then all elements in the column are the same (and not '-'), so we return True
            return True
    return False


class NotTautology(Exception):
    """
    Creating a new exception that will be raised in unate_reduction() in the event that the corollary to general unate
    reduction results in no tautology.
    """
    pass
def unate_reduction(cover: Cover) -> Cover:
    """
    Suppose you have a cover that can be rearranged as:
    F = [[U, F1], [D, F2]], where U are unate columns and D is a matrix of '-'s. Then, F is a tautology iff 
    F2 is a tautology.

    For example, consider the cover 
    F = [
            [1,0,1,0],
            [1,0,0,-],
            [-,0,1,0],
            [-,-,1,1],
            [-,-,-,1]
        ]
    is in the correct form, where U = [[1,0],[1,0],[-,0]] (positive unate on the first column, negative unate on second column) and 
    D = [[-,-],[-,-]]. Then, F is a tautology if and only if [[1,1][-,1]] is a tautology.
    """
    # Here's what I'm thinking so far.... The matrix D can only be constructed from unate columns (positive or negative or a combination of both). So first, 
    # this algorithm should probably identify all of the unate columns in F and group them together. These columns will be used to determine the matrices U and D.
    # Next, we have to figure out which rows in those columns are all 0s. We build the maximal set of such rows to construct the matrix D. In other words, D contains
    # all of the unate columns and is also all dashes. If there are no rows of all dashes within the unate columns, we cannot construct a matrix D and unate reduction
    # fails. In other words, D is the matrix whose entries are '-' in ALL unate columns of F. Having rows that are all-dash on some proper subset of unate columns is
    # insufficient to build D. There must be simultaneous dashes across the entire set of unate columns in order for unate reduction to occur. 

    unate_columns = cover.unate_columns()       # grabbing the indices of all of the unate columns (positive or negative unate, includes columns containing all dashes, if they exist).
    unate_columns_set = set(unate_columns)      # also obtaining a set version for fast lookups

    # now we need to figure out which rows have all dashes in those unate columns
    d_rows = []         # the rows in matrix D are the rows that have all dashes in the unate_columns
    for row_num, cube in enumerate(cover):
        for i in unate_columns:
            if cube[i] != '-':
                break
        else:
            d_rows.append(row_num)

    if len(unate_columns) != 0 and len(d_rows) == 0:
        # if this is true, then by the corlollary to general unate reduction, F is not a tautology 
        raise NotTautology("No all-dash rows in unate columns means that the cover is not a tautology.")
    
    print(f"Rows in F2 are {d_rows}")
    print(f"Columns not included in F2 are {unate_columns}")

    if len(d_rows) == 0:
        return cover        # if unate reduction is impossible, return the original cover
    
    # Now we need to construct the matrix F2. This will consist of the rows in d_rows and the columns NOT IN unate_columns
    F2 = Cover()
    for row_num in d_rows:
        cube_of_interest = cover[row_num]    # a subset of this cube will be added to the new cover
        new_cube = ""                       # initializing new cube to be constructed from cube_of_interest
        for idx, val in enumerate(cube_of_interest):
            if idx in unate_columns_set:
                continue
            new_cube += val                 # we add val to new_cube as long as we are not looking in a forbidden column
            #print(f"Adding {val} from column {idx} to new_cube, since column {idx} is not a unate column")
        F2.add(Cube(new_cube))

    return F2

def most_binate_variable(cover: Cover):
    """Returns the index of the most binate variable in the cover."""
    # Binateness will be checked as follows. Let "dc" be the number of dashes in the column.
    # Then, as we iterate through the column, if we see a '1', increment the counter. If we see
    # a '0', decrement the counter. Call this counter "C". The metric for binateness will seek
    # to minimize the quantity abs(c) + dc. Thus, equal numbers of '1's and '0's is rewarded, 
    # and dashes are penalized. 
    columns = cover.get_columns()
    most_binate_column = None
    best_column_score = math.inf
    for column_idx, column in enumerate(columns):
        dc = 0
        ctr = 0
        for val in column:
            if val == '-':
                dc += 1
                if dc >= best_column_score:  # there might be a better way to prune this earlier that takes into account the number of values left in the column, but i will go with this for now
                    break  # if the number of dashes exceeds the best known column, then we already know that this column is no longer a candidate, so we stop looping
            elif val == '1':
                ctr += 1
            elif val == '0':
                ctr -= 1
        else:
            # another for...else block. This code will only run if we looped thru the entire column without breaking
            if (column_score := abs(ctr) + dc) < best_column_score:
                best_column_score = column_score
                most_binate_column = column_idx

    return most_binate_column

def cofactors(cover: Cover, variable: int) -> tuple[Cover, Cover]:
    """
    Returns a tuple of the positive and negative cofactors of cover with respect to variable.
    Positive cofactor is index 0 in return tuple, negative cofactor is index 1 in return tuple. 
    """
    # this line indexes a zip object, which is unfortunatley not O(1). However, creating the zip object is O(1), so indexing it is O(variable].
    # If instead I had a get_single_column(variable) method, it would be O(n) to get the column and O(1) to index it (where n is number of rows in the cover).
    # So indexing the zip object is still faster (unless variable is the very last column, in which case they are equally slow). 
    column = cover.get_columns()[variable]      
    pos_cofactor = Cover()
    neg_cofactor = Cover()


    for idx, val in enumerate(column):
        cube_of_interest = Cube(cover[idx].cube)  # we have to construct a new cube object in order to use the cube_remove() method, which is necessary to remove the variable column from the cofactor cubes. This is O(n) where n is the number of variables in the cover, since we have to copy the entire cube string in order to build the new cube.
        val = cube_of_interest.pop(variable)   # removes the variable column from the cube of interest and returns the value that was stored there (either '1', '0', or '-')

        #the pop method modified cube_of_interest by removing the variable column, so we can add it to the appropriate cofactor without worrying about removing the variable column later.
        if val == '1':
            pos_cofactor.add(cube_of_interest)
        elif val == '0':
            neg_cofactor.add(cube_of_interest)
        elif val == '-':
            pos_cofactor.add(cube_of_interest)
            neg_cofactor.add(cube_of_interest)
            
    return pos_cofactor, neg_cofactor

def is_tautology(cover: Cover) -> bool:
    """Checks if cover is a tautology"""

    minterms_covered = 0
    dont_cares = 0                          # number of dashes (dont care's) in the cover
    num_variables = cover[0].size()         # number of variables in a cover is equal to the number of variables in one of it's cubes
    for cube in cover:
        dashes = cube.num_DC()
        if dashes == num_variables:         # if a cube has all dashes, then it covers all minterms, so the cover is a tautology
            return True        
        dont_cares += dashes
        minterms_covered += 2**dashes       # number of minterms covered by a cube is 2**(num dashes in cube). For example, "0--" covers (000, 001, 010, 011)
    
    minterms_required = 2**num_variables    # used for special case 1 and 2

    # Special case 1 (Week 8 notes):
    # Theorem: Let F be a cover with n variables. If the total number of minterms covered by F is less than 2**n, then F is not a tautology. 
    if minterms_covered < minterms_required:
        return False
    
    # Special case 2: 
    # Let F be a cover with no "-"s. Then if the total number of minterms covered by F is exactly 2**(num variables), then F is a tautology.
    if dont_cares == 0 and minterms_covered == minterms_required:
        return True

    # Special case 3: If there is a column with all 1s or all 0s, then the cover is not a tautology.
    if column_check(cover.get_columns()):
        return False
    
    # now do unate reduction
    try:
        cover = unate_reduction(cover)
    except NotTautology:
        return False
    
    # now select the most binate variable (j)
    j = most_binate_variable(cover)

    F_j, F_j_prime = cofactors(cover, j)

    # and look at the cofactors with respect to j
    if is_tautology(F_j) == False:
        return False
    if is_tautology(F_j_prime) == False:
        return False

    return True



test_num = 1
#file_path = f"Tautology_Check_Tests/TC_T{test_num}"
file_path = f"Examples/majority-1"

with open(file_path, "r") as file:
    cover = Cover()
    for line in file:
        if line[0] == "." or line[0] == "#":
            continue
        cube_str = line.strip().split()[0] # strips leading/trailing whitespace, then splits into [inputs, output], then discards the output. Looks like "0-11-1-000-1" (or something similar)
        cover.add(Cube(cube_str))          # converts cube_str into a Cube object, then adds the Cube to cover

cover = Cover()
cover.add(Cube("10-0-1-"))
cover.add(Cube("10-1-1-"))
cover.add(Cube("1--0-0-"))
cover.add(Cube("1--1-1-"))
cover.add(Cube("---0-1-"))
cover.add(Cube("1--1-0-"))

print(cover)
print(unate_reduction(cover))
print(most_binate_variable(cover))


