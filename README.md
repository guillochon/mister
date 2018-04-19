# mister

[![Build Status](https://travis-ci.org/guillochon/mister.svg?branch=master)](https://travis-ci.org/guillochon/mister)

Python package that returns the basic properties of main-sequence stars taken from the MESA Isochrones & Stellar Tracks (MIST) model grid database.

Usage:

```python
from mister import Mister

mr = Mister()

# Retrieve the radius of main-sequence star with solar metallicity at half its
# main-sequence lifetime.
print(mr.radius([0.0, 1.0, 0.5]))
```

At the moment, this package provides the following functions:

*   `radius(metallicity, mass, age_fraction)`
*   `lifetime(metallicity, mass)`
