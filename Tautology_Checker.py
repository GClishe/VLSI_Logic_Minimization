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

print(cover)
columns = cover.get_columns()
print(list(columns))
print(column_check(cover.get_columns()))


cover = Cover()
cover.add(Cube("100"))
cover.add(Cube("001"))
cover.add(Cube("-0-"))
print(cover)
columns = cover.get_columns()     
print(list(columns))
print(column_check(cover.get_columns()))





