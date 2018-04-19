"""Determine fits that minimize error for MIST dataset."""
import os
import pickle

import numpy as np
from scipy.interpolate import RegularGridInterpolator


class Mister(object):
    """Class that computes main sequence star properties from MIST library."""

    def __init__(self, **kwargs):
        """Initialize class."""
        self._dir_path = os.path.dirname(os.path.realpath(__file__))

    def radius(self, params):
        """Return star main sequence radius."""
        try:
            self._pickled_radius_rgi
        except AttributeError:
            with open(os.path.join(self._dir_path, 'pickles',
                                   'radius_rgi.pickle'), 'rb') as f:
                self._pickled_radius_rgi = pickle.load(f)

        return self._pickled_radius_rgi(params)

    def lifetime(self, params):
        """Return star main sequence lifetime."""
        try:
            self._pickled_lifetime_rgi
        except AttributeError:
            with open(os.path.join(self._dir_path, 'pickles',
                                   'lifetime_rgi.pickle'), 'rb') as f:
                self._pickled_lifetime_rgi = pickle.load(f)

        return self._pickled_lifetime_rgi(params)

    def rad_func(self, b, lzs, ms, ts):
        """Use combination of power laws to fit radius."""
        scaled_lms = np.log10(ms)
        scaled_lms -= self._min_ilms
        scaled_lms /= np.max(scaled_lms)
        scaled_lzs = lzs
        scaled_lzs -= self._min_ilzs
        scaled_lzs /= np.max(scaled_lzs)
        scaled_lzs += 0.5
        scaled_ts = ts + 0.5
        # print(min_ilms, max_ilms, scaled_lms)
        # print(scaled_lzs)
        # print(scaled_ts)
        # raise
        radius = b[0] * ms ** (b[1] + b[2] * scaled_ts +
                               b[3] * scaled_lms) * (
            scaled_ts ** (b[4] + b[5] * scaled_ts + b[6] * scaled_lms)) * (
            scaled_lzs ** b[7])
        return radius

    def rad_log_like(self, b):
        """Objective function for radius fitting."""
        log_like = - np.sum(
            ((self.rad_func(b, self._mlzs, self._mms, self._mts) -
              self._irs) / self._irs) ** 2)
        return log_like

    def ptform(self, u):
        """Map priors to physical units."""
        x = u.copy()
        for vi, v in enumerate(self._free_vars):
            x[vi] = self._free_vars[v][0] + (
                self._free_vars[v][1] - self._free_vars[v][0]) * u[vi]
        return x

    def construct_analytical_functions(self):
        """Construct invertible functions based on interpolations."""
        import warnings
        # from dynesty import DynamicNestedSampler
        from dynesty import NestedSampler
        from collections import OrderedDict

        import numpy as np

        warnings.filterwarnings("ignore")

        self._free_vars = OrderedDict((
            ('r0', (0.25, 1.5)),
            ('mpow', (-10, 10)),
            ('mtrunning', (-5, 5)),
            ('mrunning', (-5, 5)),
            ('tpow', (-10, 10)),
            ('trunning', (-10, 10)),
            ('tmrunning', (-10, 10)),
            ('zpow', (-10, 10))
        ))

        self._min_ilms, self._max_ilms = np.log10(
            np.min(self._ims)), np.log10(np.max(self._ims))
        self._min_ilzs, self._max_ilzs = np.min(self._ilzs), np.max(self._ilzs)

        self._mlzs, self._mms, self._mts = np.meshgrid(
            self._ilzs, self._ims, self._its, indexing='ij')

        self._ndim = len(list(self._free_vars.keys()))

        dsampler = NestedSampler(
            self.rad_log_like, self.ptform, self._ndim, sample='rwalk')

        # dsampler.run_nested(dlogz_init=0.01)
        dsampler.run_nested(dlogz=0.01)

        res = dsampler.results

        bbest = res['samples'][-1]
        print(res['logl'])
        print(list(res.keys()))

        print(bbest)
        prt_ts = np.linspace(0, 1, 5)
        test_masses = 10.0 ** np.linspace(self._min_ilms, self._max_ilms, 3)
        test_lzs = np.linspace(self._min_ilzs, self._max_ilzs, 3)
        for tlz in test_lzs:
            for tm in test_masses:
                print('Radii for logz = {} and m = {}'.format(tlz, tm))
                print(self.radius_rgi([[tlz, tm, x] for x in prt_ts]))
                print(self.rad_func(bbest, tlz, tm, prt_ts))
        max_frac_err = np.max(np.abs(self.rad_func(
            bbest, self._mlzs, self._mms, self._mts
        ) - self._irs) / self._irs)

        print('Maximum fractional error: {:.1%}'.format(max_frac_err))

    def create_pickles(self, mist_path=os.path.join('..', 'MIST')):
        """Create pickled functions from MIST data."""
        import codecs
        import re

        import pickle

        import numpy as np

        from tqdm import tqdm
        from glob import glob

        self._its = np.linspace(0, 1, 101)

        self._irs = []
        self._ims = []
        self._ilzs = []
        self._ilifetimes = []

        for metal_folder in sorted(glob('../MIST/*')):
            lz = float(metal_folder.split('/')[-1].split('_')[3].replace(
                'm', '-').replace('p', ''))
            self._ilzs.append(lz)
            ilz_order = np.argsort(self._ilzs)
        self._ilzs = []

        for ilz, metal_folder in enumerate(tqdm(np.array(
                sorted(glob('../MIST/*')))[ilz_order])):
            lz = float(metal_folder.split('/')[-1].split('_')[3].replace(
                'm', '-').replace('p', ''))
            self._ilzs.append(lz)
            self._irs.append([])
            self._ilifetimes.append([])
            for mfi, mass_file in enumerate(tqdm(sorted(
                    glob(os.path.join(metal_folder, '*.eep'))))):
                # if mfi > 10:
                #     break
                mass = mass_file.split('/')[-1].split('M')[0]
                mass = float(mass[:3] + '.' + mass[3:])
                if ilz == 0:
                    self._ims.append(mass)
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
                self._ilifetimes[ilz].append(ts[-1])
                ts /= max(ts)

                rs = np.interp(self._its, ts, rs)

                self._irs[ilz].append(rs)

        radius_rgi = RegularGridInterpolator(  # noqa: F841
            (self._ilzs, self._ims, self._its), self._irs)

        lifetime_rgi = RegularGridInterpolator(  # noqa: F841
            (self._ilzs, self._ims), self._ilifetimes)

        for (v, k) in [(v, k) for k, v in vars().items() if k.endswith(
                '_rgi') and not k.startswith('_pickled_')]:
            with open(os.path.join(
                    self._dir_path, 'pickles', k + '.pickle'), 'wb') as f:
                pickle.dump(v, f)
