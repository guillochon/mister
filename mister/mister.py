"""Determine fits that minimize error for MIST dataset."""
import codecs
import os
import pickle
import re
import warnings
from collections import OrderedDict
from glob import glob

import cloudpickle
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from tqdm import tqdm

# from dynesty import DynamicNestedSampler
from dynesty import NestedSampler

warnings.filterwarnings("ignore")


def construct_analytical_functions():
    """Construct invertible functions based on interpolations."""
    def rad_func(b, lzs, ms, ts):
        """Use combination of power laws to fit radius."""
        scaled_lms = np.log10(ms)
        scaled_lms -= min_ilms
        scaled_lms /= np.max(scaled_lms)
        scaled_lzs = lzs
        scaled_lzs -= min_ilzs
        scaled_lzs /= np.max(scaled_lzs)
        scaled_lzs += 0.5
        scaled_ts = ts + 0.5
        # print(min_ilms, max_ilms, scaled_lms)
        # print(scaled_lzs)
        # print(scaled_ts)
        # raise
        radius = b[0] * ms ** (b[1] + b[2] * scaled_ts + b[3] * scaled_lms) * (
            scaled_ts ** (b[4] + b[5] * scaled_ts + b[6] * scaled_lms)) * (
            scaled_lzs ** b[7])
        return radius

    def rad_log_like(b):
        """Objective function for radius fitting."""
        log_like = -np.sum(((rad_func(b, mlzs, mms, mts) - irs) / irs) ** 2)
        return log_like

    def ptform(u):
        """Map priors to physical units."""
        x = u.copy()
        for vi, v in enumerate(free_vars):
            x[vi] = free_vars[v][0] + (
                free_vars[v][1] - free_vars[v][0]) * u[vi]
        return x

    mlzs, mms, mts = np.meshgrid(ilzs, ims, its, indexing='ij')

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
    test_masses = 10.0 ** np.linspace(min_ilms, max_ilms, 3)
    test_lzs = np.linspace(min_ilzs, max_ilzs, 3)
    for tlz in test_lzs:
        for tm in test_masses:
            print('Radii for logz = {} and m = {}'.format(tlz, tm))
            print(radius_rgi([[tlz, tm, x] for x in prt_ts]))
            print(rad_func(bbest, tlz, tm, prt_ts))
    max_frac_err = np.max(np.abs(rad_func(bbest, mlzs, mms, mts) - irs) / irs)

    print('Maximum fractional error: {:.1%}'.format(max_frac_err))


free_vars = OrderedDict((
    ('r0', (0.25, 1.5)),
    ('mpow', (-10, 10)),
    ('mtrunning', (-5, 5)),
    ('mrunning', (-5, 5)),
    ('tpow', (-10, 10)),
    ('trunning', (-10, 10)),
    ('tmrunning', (-10, 10)),
    ('zpow', (-10, 10))
))

its = np.linspace(0, 1, 101)

irs = []
ims = []
ilzs = []
ilifetimes = []

for metal_folder in sorted(glob('../MIST/*')):
    lz = float(metal_folder.split('/')[-1].split('_')[3].replace(
        'm', '-').replace('p', ''))
    ilzs.append(lz)
    ilz_order = np.argsort(ilzs)
ilzs = []

for ilz, metal_folder in enumerate(tqdm(np.array(
        sorted(glob('../MIST/*')))[ilz_order])):
    lz = float(metal_folder.split('/')[-1].split('_')[3].replace(
        'm', '-').replace('p', ''))
    ilzs.append(lz)
    irs.append([])
    ilifetimes.append([])
    for mfi, mass_file in enumerate(tqdm(sorted(
            glob(os.path.join(metal_folder, '*.eep'))))):
        # if mfi > 10:
        #     break
        mass = mass_file.split('/')[-1].split('M')[0]
        mass = float(mass[:3] + '.' + mass[3:])
        if ilz == 0:
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
        ilifetimes[ilz].append(ts[-1])
        ts /= max(ts)

        rs = np.interp(its, ts, rs)

        irs[ilz].append(rs)

min_ilms, max_ilms = np.log10(np.min(ims)), np.log10(np.max(ims))
min_ilzs, max_ilzs = np.min(ilzs), np.max(ilzs)

radius_rgi = RegularGridInterpolator((ilzs, ims, its), irs)

lifetime_rgi = RegularGridInterpolator((ilzs, ims), ilifetimes)

for (v, k) in [(v, k) for k, v in vars().items() if k.endswith('_rgi')]:
    with open(k + '.pickle', 'wb') as f:
        cloudpickle.dump(v, f)

with open('radius_rgi.pickle', 'rb') as f:
    pickled_radius_rgi = pickle.load(f)

# Testing functions
print(pickled_radius_rgi([0, np.min(ims), 0.5]))
