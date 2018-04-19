"""Script to create function pickles from MIST database.

Note: The `create_pickles` functions requires a few additional dependencies
not installed with `mister`: `tqdm`
"""
from mister import Mister

mr = Mister()

mr.create_pickles()
