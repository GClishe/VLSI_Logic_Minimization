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

class Cube():
    def __init__(self, *vars: str) -> None:
        self.bitarr = bitarray()

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
        return str(self.bitarr)
    
    def __and__(self, other):
        """Allows bitwise AND with the & operator between Cube and Cube or Cube and bitarray. Other operand types are not and will not be supported."""
        if isinstance(other, Cube):
            return self.bitarr & other.bitarr
        elif isinstance(other, bitarray):
            return self.bitarr & other
        else:
            raise TypeError(f"Unsupported operand type for &: 'Cube' and {type(other)}.")


    

    


x = Cube("1", "0", "-")
y = Cube("1", "0", "0")
z = bitarray([1,0,0,1,0,1])
w = 100101
print(x & y)
print(x & z)
print(x & w)

