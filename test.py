import numpy as np
import time
import sys
from Lowry_Clishe_TautCheck import is_tautology, parse_espresso, Tracker
from Lowry_Clishe_CompGen import complement, export_espresso, ComplTracker

input_file = sys.argv[1]
cover, ilb, ob = parse_espresso(input_file)
print(len({tuple(row) for row in cover}))