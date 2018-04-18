"""Determine fits that minimize error for MIST dataset."""
import codecs
import os
import re
import warnings
from collections import OrderedDict
from glob import glob

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from tqdm import tqdm

# from dynesty import DynamicNestedSampler
from dynesty import NestedSampler

warnings.filterwarnings("ignore")


def rad_func(b, ms, ts):
    """Use combination of power laws to fit radius."""
    scaled_ms = np.log10(ms)
    scaled_ms -= np.log10(np.min(ims))
    scaled_ms /= np.log10(np.max(ims))
    radius = b[0] * ms ** (b[1] + b[2] * (1.0 + b[3] * ts) +
                           b[4] * scaled_ms) * (
        1.0 + b[3] * ts ** (b[5] + b[6] * (1.0 + b[3] * ts) +
                            b[7] * scaled_ms))
    return radius


def rad_log_like(b):
    """Objective function for radius fitting."""
    log_like = -np.sum(((rad_func(b, mms, mts) - irs) / irs) ** 2)
    return log_like


def ptform(u):
    """Map priors to physical units."""
    x = u.copy()
    for vi, v in enumerate(free_vars):
        x[vi] = free_vars[v][0] + (free_vars[v][1] - free_vars[v][0]) * u[vi]
    return x


free_vars = OrderedDict((
    ('r0', (0.5, 1.5)),
    ('mpow', (0, 10)),
    ('mtrunning', (-5, 5)),
    ('mrunning', (-5, 5)),
    ('tnorm', (-5, 5)),
    ('tpow', (0, 10)),
    ('trunning', (-10, 10)),
    ('tmrunning', (-10, 10))
))

its = np.linspace(0, 1, 101)

irs = []
ims = []
ilifetimes = []
for metal_folder in glob('../MIST/*'):
    for mfi, mass_file in enumerate(tqdm(sorted(
            glob(os.path.join(metal_folder, '*.eep'))))):
        # if mfi > 15:
        #     break
        mass = mass_file.split('/')[-1].split('M')[0]
        mass = float(mass[:3] + '.' + mass[3:])
        ims.append(mass)
        ts = []
        rs = []
        with codecs.open(mass_file, 'r', encoding='utf-8') as mf:
            for line in mf:
                if line.startswith('#'):
                    continue
                sline = [x for x in re.split('\s+', line.strip()) if (
                    x is not None and x is not '')]
                phase = round(float(sline[-1]))
                if phase == -1:
                    continue
                if phase >= 2:
                    break
                t = float(sline[0])
                r = 10.0 ** float(sline[13])
                ts.append(t)
                rs.append(r)
        ts = np.array(ts)
        ts -= min(ts)
        ilifetimes.append(ts[-1])
        ts /= max(ts)

        rs = np.interp(its, ts, rs)

        irs.append(rs)
    break  # just 1 for now

rad_rgi = RegularGridInterpolator((ims, its), irs)

mms, mts = np.meshgrid(ims, its, indexing='ij')

ndim = len(list(free_vars.keys()))

dsampler = NestedSampler(
    rad_log_like, ptform, ndim, sample='rwalk')

# dsampler.run_nested(dlogz_init=0.01)
dsampler.run_nested(dlogz=0.01)

res = dsampler.results

bbest = res['samples'][-1]
print(res['logl'])
print(list(res.keys()))

print(bbest)
prt_ts = np.linspace(0, 1, 5)
test_masses = 10.0 ** np.arange(-1, 2)
for tm in test_masses:
    print(rad_rgi([[tm, x] for x in prt_ts]))
    print(rad_func(bbest, tm, prt_ts))
max_frac_err = np.max(np.abs(rad_func(bbest, mms, mts) - irs) / irs)

print('Maximum fractional error: {:.1%}'.format(max_frac_err))
