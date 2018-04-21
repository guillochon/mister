"""Script to find invertible analytical functions from MIST database.

Note: The code below requires a few additional dependencies
not installed with `mister`: `tqdm`, `dynesty`
"""
from mister import Mister

mr = Mister()

mr.create_pickles()

mr.construct_analytical_functions()
