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
        """Allows bitwise AND with the & operator between Cube and Cube or Cube and bitarray. Other operand types are not and will not be supported."""
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
        
    def contains(self, other) -> bool:
        """
        Returns true if self contains other. For example, '-10' contains '110' and '010'. In other words,
        returns true only if other is a subset of self.
        """
        return (other | self.bitarr) == self.bitarr
    
    def minterms(self) -> int:
        """Describes the number of minterms present in a cube. For example, --0 is 4 minterms (000, 010, 100, 110)"""
        return 2**(self.cube.count("-"))    # returns 2**n where n is the number of dashes in the cover
    
    def size(self) -> int:
        """Returns the number of variables present in the cube"""
        return len(vars)

class Cover():
    def __init__(self, *cubes: Cube):
        # I considered the primary container being a numpy array. But the problem is that np arrays are contiguous in memory, so 
        # appending elements is expensive (requires duplicating entire array). Since I expect to do many appends and deletions, 
        # a traditional python list is attractive. 
        self.cover = []
        for cube in cubes:
            self.cover.append(cube)
    def __repr__(self):
        return str(self.cover)
    
    def __getitem__(self, idx: int) -> Cube:
        """Allows the Cover object to be subscriptable. In other words, cover[0] will not throw an error."""
        return self.cover[idx]
    
    def add(self, cube: Cube) -> None:
        """Adds a cube to the end of the cover list"""
        self.cover.append(cube)

    def size(self):
        """Returns the number of cubes in the cover"""
        return len(self.cover)

    def complement(self):
        """see week 9 notes on how to implement"""

def is_tautology(cover: Cover) -> bool:
    """Checks if cover is a tautology"""
    # Special case 1 (Week 8 notes):
    # Theorem: Let F be a cover with n variables. If the total number of minterms covered by F is less than 
    # 2**n, then F is not a tautology. 
    minterms_covered = 0
    num_variables = cover[0].size()  # number of variables in a cover is equal to the number of variables in one of it's cubes
    for cube in cover:
        minterms_covered += cube.minterms()
    if minterms_covered < 2**num_variables:
        return False



test_num = 1
file_path = f"Tautology_Check_Tests/TC_T{test_num}"

with open(file_path, "r") as file:
    cover = Cover()
    for line in file:
        if line[0] == ".":
            continue
        cube_str = line.strip().split()[0] # strips leading/trailing whitespace, then splits into [inputs, output], then discards the output. Looks like "0-11-1-000-1" (or something similar)
        cover.add(Cube(cube_str))          # converts cube_str into a Cube object, then adds the Cube to cover

print(*cover[:10], sep='\n')





