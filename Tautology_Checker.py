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


