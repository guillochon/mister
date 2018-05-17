"""Script to create function pickles from MIST database.

Note: The `create_pickles` functions requires a few additional dependencies
not installed with `mister`: `tqdm`
"""
from mister import Mister

import numpy as np

mr = Mister()

print(mr.radius([[0, 0.1, x] for x in np.linspace(0, 1.0, 11)]))
print(mr.radius([[0, 1.0, x] for x in np.linspace(0, 1.0, 11)]))
print(mr.radius([[0, 10.0, x] for x in np.linspace(0, 1.0, 11)]))

print(mr.radius([[x, 0.1, 0] for x in np.linspace(-4, 0.5, 3)]))
print(mr.radius([[x, 1.0, 0] for x in np.linspace(-4, 0.5, 3)]))
print(mr.radius([[x, 10.0, 0] for x in np.linspace(-4, 0.5, 3)]))

print(mr.radius([[x, 0.1, 1] for x in np.linspace(-4, 0.5, 3)]))
print(mr.radius([[x, 1.0, 1] for x in np.linspace(-4, 0.5, 3)]))
print(mr.radius([[x, 10.0, 1] for x in np.linspace(-4, 0.5, 3)]))
