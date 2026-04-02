#To use, run the command 
#   python3 Lowry_Clishe_CompVerify.py <directory of first cover file> <directory of second cover file>
#The script simply OR (via vstack) the two covers and checks for tautology
#There are no real checks for if the variable count doesn't match or if it's the same cover so use discression

import sys
import time
import numpy as np
from Lowry_Clishe_TautCheck import is_tautology, parse_espresso, Tracker

file1 = sys.argv[1]
file2 = sys.argv[2]

cover1, ilb, _ = parse_espresso(file1)
cover2, _, _ = parse_espresso(file2)

n_vars = cover1.shape[1]

union_cover = np.vstack((cover1, cover2))
union_cover = np.array(list({tuple(cube) for cube in union_cover}), dtype=np.uint8)

print(f"File 1: {len(cover1)} cubes")
print(f"File 2: {len(cover2)} cubes")
print(f"Checking Tautology on Union ({len(union_cover)} total cubes)...")

start_time = time.perf_counter()
is_taut, witness = is_tautology(union_cover, ilb, 0, Tracker())

print(f"\n--- Verification Result ---")
if is_taut: print(f"Status                : PASS (The covers form a tautology)")
else:
    print(f"Status                : FAIL (Missing coverage)")
    if witness:
        print(f"Missing Minterm       : {''.join(str(witness.get(v, '-')) for v in ilb)}")
print(f"Time Taken            : {(time.perf_counter() - start_time):.3f} sec\n")