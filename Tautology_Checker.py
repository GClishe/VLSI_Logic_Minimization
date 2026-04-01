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
import time
from dataclasses import dataclass    


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
        """Removes variable idx from both representations and returns the removed value."""
        if idx < 0 or idx >= len(self.cube):
            raise IndexError(f"Index {idx} out of range for cube of size {len(self.cube)}")

        val = self.cube[idx]
        self.cube = self.cube[:idx] + self.cube[idx+1:]
        del self.bitarr[2 * idx : 2 * idx + 2]
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

def SCC_Minimize(cover: list[Cube]) -> list[Cube]:
    """Performs SCC minimization on input cover list. Does not modify cover; instead, it builds a new minimized list."""
    new_cover = []
    rejected_cubes = set() # stores indices of cubes in cover that will not be included in the new cover
    for cube_idx in range(len(cover)):
        curr_cube = cover[cube_idx]     

        # Do not add null cubes
        if curr_cube.is_null():
            rejected_cubes.add(cube_idx)
            continue
        
        # Reject cubes contained by other cubes
        flag = 0
        for idx, cube2 in enumerate(cover):
            if cube2.contains(curr_cube) and idx != cube_idx:       # every cube contains itself, so we need to exclude self-comparison
                rejected_cubes.add(cube_idx)
                flag = 1
                break
        if flag == 1:
            continue

        new_cover.append(curr_cube)
    
    return new_cover


@dataclass(frozen=True)
class CoverView:
    """
    CoverView is a lightweight object that allows us to keep track of which rows and columns of the master cover we are currently looking at, 
    without having to create new cover lists or cube objects. This is useful for the recursive calls in is_tautology(), since we can just pass
    a CoverView object that references the relevant rows and columns in the original master cover, rather than having to create entirely new 
    covers for each recursive call.
    """
    rows: tuple[int, ...]   # indices into master_cover
    cols: tuple[int, ...]   # indices of currently active variables

def make_initial_view(master_cover: list[Cube]) -> CoverView:
    """Creates an initial CoverView that references all rows and columns of the master cover."""
    if not master_cover:
        raise ValueError("Empty cover")
    return CoverView(
        rows=tuple(range(len(master_cover))),
        cols=tuple(range(master_cover[0].size()))
    )

def unate_columns_view(master_cover: list[Cube], view: CoverView) -> tuple[int, ...]:
    """
    Returns the LOCAL indices of all unate columns in the current view.

    A column is unate if it contains no '1's or no '0's (but can contain '-'s).

    Local indices refers to the indices in the current view, as opposed to the master cover. For example, if view.cols = (0, 2, 4) 
    and the first column in the view (column 0 in the view) is unate, then this function would return 0 as the local index of that 
    unate column, even though the index of that column in the master cover is 0. Similarly, if the second column in the view 
    (column 1 in the view) is unate, then this function would return 1 as the local index of that unate column, even though the 
    index of that column in the master cover is 2.
    """
    unate_local_cols = []

    for local_idx, col_idx in enumerate(view.cols):
        first_polarity = None       # first non-dash symbol encountered in this column

        for row_idx in view.rows:
            val = master_cover[row_idx][col_idx] # the value of the current column in the current row of the master cover

            if val == '-':
                continue
            if first_polarity is None:
                first_polarity = val
            elif val != first_polarity:
                break
        else:
            unate_local_cols.append(local_idx)

    return tuple(unate_local_cols)

class NotTautology(Exception):
    """
    Creating a new exception that will be raised in unate_reduction() in the event that the corollary to general unate
    reduction results in no tautology.
    """
    pass
def unate_reduction_view(master_cover: list[Cube], view: CoverView) -> CoverView:
    """
    View based unate reduction.

    Suppose the current subcover can be rearraged as [[U, F1], [D, F2]], where U consists of all unate columns
    in the current view and D is the set of rows that are '-' in every unate column.

    Then, F is tautology iff F2 is a tautology. 
    """
    unate_local_cols = unate_columns_view(master_cover, view)

    # if there are no unate columns in the current view, then no reduction is possible
    if not unate_local_cols:
        return view
    
    unate_local_set = set(unate_local_cols)     # set version for fast lookups

    d_rows = []     # rows that are '-' in every unate column
    for row_idx in view.rows:
        cube = master_cover[row_idx]    # obtaining the cube in the master cover corresponding to the current row index in the view

        for local_idx in unate_local_cols:          # iterating through the local indices of the unate columns
            original_col_idx = view.cols[local_idx] # getting the index of the unate column in the master cover corresponding to the local index in the view
            if cube[original_col_idx] != '-':
                break
        else:
            d_rows.append(row_idx)

    # Corollary: if there are unate columns but no all-dash rows in them,
    # then the cover is not a tautology.
    if len(unate_local_cols) != 0 and len(d_rows) == 0:
        raise NotTautology("No rows are '-' in all unate columns, so the cover is not a tautology.")
    
    new_cols = tuple(col_idx for local_idx, col_idx in enumerate(view.cols) if local_idx not in unate_local_set) # the new view will have the same columns as the old view, except with the unate columns removed
    
    return CoverView(rows=tuple(d_rows), cols=new_cols)

def most_binate_variable_view(master_cover: list[Cube], view: CoverView):
    """Returns the local index of the most binate variable in the view. Local index means the index of the variable in the current view, as opposed to the master cover."""

    # Binateness in a view will be checked as follows. Let "dc" be the number of dashes in the column within the view. Then, as we iterate through the column within the view, if we see a '1', increment
    # the counter. If we see a '0', decrement the counter. Call this counter "C". The metric for binateness will seek to minimize the quantity abs(c) + dc. Thus, equal numbers of '1's and '0's is 
    # rewarded, and dashes are penalized.

    best_local_idx = None
    best_score = math.inf   # the best (lowest) score for binateness that we have seen so far

    for local_idx, col_idx in enumerate(view.cols): # local_idx is the index of the variable in the current view, NOT the master cover. col_idx is the index of the variable in the master cover. We need both of these because we need to loop through the relevant rows in the master cover to compute the score for this column, but we also need to keep track of the local index of the column in the current view since that is what will be used in the recursive calls after we select the most binate variable.
        dc = 0
        ctr = 0

        for row_idx in view.rows:
            val = master_cover[row_idx][col_idx]
            if val == '-':
                dc += 1
                if dc >= best_score:        # if the number of dont cares exceeds the best known score, then we already know that this column is no longer a candidate, so we stop looping
                    break

            elif val == '1':
                ctr += 1
            elif val == '0':
                ctr -= 1
        else:   # executed only if we did not break out of the loop due to too many dont cares. If we did break out of the loop, then we skip the score calculation and move on to the next column.
            score = abs(ctr) + dc
            if score < best_score:
                best_score = score
                best_local_idx = local_idx

    return best_local_idx

def cofactors_view(master_cover: list[Cube], view: CoverView, local_var_idx: int):
    """
    Returns a tuple of the positive and negative cofactors of the cover represented by view with respect to the variable at local_var_idx in view.cols.
    """
    # local_var_idx is an index into view.cols, which is the index of the variable in the current view. 
    # We need to convert this to an index into the original master cover, since the cubes in master_cover are indexed according to the original variable ordering.
    # As an example, if view.cols = (0, 2, 4) and local_var_idx = 1, then original_var_idx = 2, since the variable at index 1 in the current view corresponds to 
    # the variable at index 2 in the original master cover
    original_var_idx = view.cols[local_var_idx] # the index of the variable in the original master cover that corresponds to the local variable index in the current view

    pos_rows = []   # rows in master cover that have a 1 in the variable of interest (original_var_idx)
    neg_rows = []   # rows in master cover that have a 0 in the variable of interest (original_var_idx)

    for row_idx in view.rows:   # each value in view.rows is an index into master_cover that corresponds to a cube that is active in the current view
        val = master_cover[row_idx][original_var_idx] # the value of the variable of interest in the current row of the master cover

        if val == '1':
            pos_rows.append(row_idx)
        elif val == '0':
            neg_rows.append(row_idx)
        elif val == '-':
            pos_rows.append(row_idx)
            neg_rows.append(row_idx)
    
    new_cols = view.cols[:local_var_idx] + view.cols[local_var_idx+1:] # the new view will have the same columns as the old view, except with the variable of interest removed

    # Instead of constructing new covers for the cofactors, we can just construct new views that reference the relevant rows and columns in the original master cover. 
    # This is more efficient because we don't have to create new cube objects or new cover lists; we can just keep track of which rows and columns are relevant for 
    # each cofactor using the CoverView object.

    # Basically the positive cofactor view contains all of the rows that have a 1 in the variable of interest along with all of the columns except the variable of interest,
    # and the negative cofactor view contains all of the rows that have a 0 in the variable of interest along with all of the columns except the variable of interest.
    return (
        CoverView(tuple(pos_rows), new_cols),
        CoverView(tuple(neg_rows), new_cols),
    )

def column_check_view(master_cover: list[Cube], view: CoverView):
    """Returns True if there is a column in the view that contains all 1s or all 0s (and no '-'s), and False otherwise."""
    for col_idx in view.cols: # loop thru all of the columns specified in the given view
        first = None    # first is assigned to the first non-dash value in the column, and is used to compare against the rest of the values in the column
        pure = True     # flag that is raised to False if we encounter a value in the column that is not the same as first (and is not a dash)

        for row_idx in view.rows: # loop thru all of the rows specified in the given view
            val = master_cover[row_idx][col_idx] # the value of the current column in the current row of the master cover
            if val == '-':
                pure = False
                break
            if first is None:
                first = val
            elif val != first:
                pure = False
                break


        if pure and first is not None:  # first would be None if all of the values in the column are dashes, in which case we do not consider the column to be pure
            return True

    return False


def is_tautology_view(master_cover: list[Cube], view: CoverView) -> bool:

    num_variables = len(view.cols)
    minterms_covered = 0
    dont_cares = 0

    for row_idx in view.rows:
        cube = master_cover[row_idx]    # cube in master cover corresponding to the current row index in the view

        dashes = 0
        all_dash = True
        for col_idx in view.cols:
            if cube[col_idx] == '-':
                dashes += 1
            else:
                all_dash = False

        if all_dash:    # if a cube has all dashes, the cover is a tautology
            return True

        dont_cares += dashes
        minterms_covered += 2 ** dashes     # number of minterms covered by a cube is 2**(num dashes in cube). For example, "0--" covers (000, 001, 010, 011)

    minterms_required = 2 ** num_variables  # number of minterms required for a tautology is 2**(number of variables in the view)

    if minterms_covered < minterms_required:
        return False

    if dont_cares == 0 and minterms_covered == minterms_required:
        return True

    # if there exists a column with all 1s or all 0s without any dashes, then the cover is not a tautology
    if column_check_view(master_cover, view):
        return False

    # Now we do unate reduction
    try:
        view = unate_reduction_view(master_cover, view)
    except NotTautology:
        return False

    # now we select the most binate variable and obtain the positive and negative cofactor views with respect to that variable.

    j = most_binate_variable_view(master_cover, view)               # j is the local index of the most binate variable (index in the current view, not the master cover)
    pos_view, neg_view = cofactors_view(master_cover, view, j)      # obtaining the positive and negative cofactor views with respect to the most binate variable

    # if either cofactor is not a tautology, then the cover is not a tautology.
    if is_tautology_view(master_cover, pos_view) == False:
        return False
    
    if is_tautology_view(master_cover, neg_view) == False:
        return False

    return True

def is_tautology(cover: list[Cube]) -> bool:
    """Wrapper for the is_tautology_view function that creates an initial view referencing the entire cover and then calls is_tautology_view on that view."""
    view = make_initial_view(cover)
    return is_tautology_view(cover, view)



test_num = 2
file_path = f"Tautology_Check_Tests/TC_T{test_num}"
#file_path = f"Examples/majority-1"

with open(file_path, "r") as file:
    cover = []
    for line in file:
        if line[0] == "." or line[0] == "#":
            continue
        cube_str = line.strip().split()[0] # strips leading/trailing whitespace, then splits into [inputs, output], then discards the output. Looks like "0-11-1-000-1" (or something similar)
        cover.append(Cube(cube_str))          # converts cube_str into a Cube object, then adds the Cube to cover

cur_time = time.perf_counter()
print(is_tautology(cover))
print(f"Time elapsed: {time.perf_counter() - cur_time}s")


