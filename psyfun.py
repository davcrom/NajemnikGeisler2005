import numpy as np
import pandas as pd
from tqdm import tqdm

from scipy import stats
from scipy import optimize
from scipy import integrate

from matplotlib import pyplot as plt
from matplotlib.patches import Circle

def logistic(x, a, b, g, l):
    """
    Evaluates the logistic function at x.
    """
    return g + (1 - g - l) / (1 + np.exp(-b * (x - a)))

def inv_logistic(y, a, b, g, l):
    """
    Evaluates the inverse of the logistic function at y.
    """
    c = (y - g) / (1 - g - l)
    c = (1 - c) / c
    if c > 0:
        return a - np.log(c) / b
    else:
        return -np.inf

def fit_psyfun(fun, levels, pct, bounds=None, init=None, maxfev=10000):
    """
    Fit a function to psychometric data.

    Parameters
    ----------
    levels : np.ndarray
        stimulus levels
    pct : np.ndarray
        fraction of responses at each stimulus level
    bounds : tuple of np.ndarray
        upper and lower bounds for each parameter (optional)
    init : np.ndarray
        initial guesses parameter values to assist the fitting (optional)
    maxfev : int
        maximum number of iterations during fitting (optional)
    """
    if bounds is None:
        bounds = np.array([levels.min(), 0, 0, 0]), np.array([levels.max(), np.inf, 1, 1])
    if init is None:
        init = np.array([np.median(levels), 100, 0, 0])
    (a, b, g, l), _ = optimize.curve_fit(fun, levels, pct, bounds=bounds, maxfev=maxfev)
    return a, b, g, l

class VisibilityMap():
    # DataFrame passed to init must have these columns
    DATACOLUMNS = ['grating_eccentricity', 'grating_contrast', 'correct']
    # Most pyschometric functions can be parametrized with:
    PSYFUNPARS = ['alpha', 'beta', 'gamma', 'lambda']
    PSYFUNS = {'logistic': logistic}

    def __init__(self, df, log_contrast=True, psyfun='logistic'):
        assert all([col in df.columns for col in self.DATACOLUMNS])
        self.trial_data = df.copy()
        self.log_contrast = log_contrast
        if self.log_contrast:
            self.trial_data['grating_contrast'] = np.log10(self.trial_data['grating_contrast'])
        self.psyfun = psyfun
        self.contrast_levels = np.sort(self.trial_data['grating_contrast'].unique())
        self.eccentricities = np.sort(self.trial_data['grating_eccentricity'].unique())
        self.psyfun_bounds = (
            np.array([self.contrast_levels.min(), 0, 0.25, 0]),  # lower bounds
            np.array([self.contrast_levels.max(), np.inf, 0.75, 0.25])  # upper bounds
            )

    def fit_psyfuns(self):
        # Accumulate fitted parameters for each eccetricity
        psyfuns = []
        for ecc, group in self.trial_data.groupby('grating_eccentricity'):
            (a, b, g, l), _ = optimize.curve_fit(
                self.PSYFUNS[self.psyfun],
                self.contrast_levels,
                group.groupby('grating_contrast')['correct'].mean(),
                bounds=self.psyfun_bounds,
                maxfev=10000)
            fit_dict = {key:val for key, val in zip(self.PSYFUNPARS, (a, b, g, l))}
            psyfuns.append(fit_dict)
        # Store fitted parameters in DataFrame
        self.psyfuns = pd.DataFrame(psyfuns, index=self.eccentricities)

    def plot_psyfuns(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        for (ecc, data), (ecc, fits) in zip(self.trial_data.groupby('grating_eccentricity'), self.psyfuns.iterrows()):
            p_correct = data.groupby('grating_contrast')['correct'].mean()
            ax.scatter(self.contrast_levels, p_correct, alpha=0.5)
            x = np.linspace(self.contrast_levels.min(), self.contrast_levels.max(), 1000)
            ax.plot(x, self.PSYFUNS[self.psyfun](x, *tuple(fits)), label=f'{ecc}\u00b0')
        ax.set_yticks([0, 0.5, 1])
        ax.set_ylabel('P(correct)')
        if self.log_contrast:
            xticklabels = [f'{10**c:.3f}' for c in self.contrast_levels]
        else:
            xticklabels = [f'{c:.3f}' for c in self.contrast_levels]
        ax.set_xticks(self.contrast_levels)
        ax.set_xticklabels(xticklabels)
        ax.set_xlabel('Grating contrast')
        ax.legend(title='Eccentricity', bbox_to_anchor=(1.05, 1), frameon=False)

    def interpolate_parameters(self):
        lin_fits = []
        for par in self.PSYFUNPARS:
            res = stats.linregress(self.eccentricities, self.psyfuns[par])
            fit_dict = {val:getattr(res, val) for val in ['slope', 'intercept', 'rvalue', 'pvalue']}
            lin_fits.append(fit_dict)
        self.linfits = pd.DataFrame(lin_fits, index=self.PSYFUNPARS)

    def plot_parameter(self, par, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        ax.scatter(self.eccentricities, self.psyfuns[par])
        x = np.linspace(self.eccentricities.min(), self.eccentricities.max(), 100)
        slope = self.linfits.loc[par, 'slope']
        intercept = self.linfits.loc[par, 'intercept']
        rsquared = self.linfits.loc[par, 'rvalue'] ** 2
        pvalue = self.linfits.loc[par, 'pvalue']
        ax.plot(x, slope * x + intercept, color='black', label=f'R$^2$={rsquared:.2f}\np={pvalue:.2e}')
        ax.legend(frameon=False)
        ax.set_ylabel(par)
        ax.set_xticks(self.eccentricities)
        ax.set_xlabel('Eccentricity (\u00b0)')
        return ax

    def get_psyfun_pars(self, ecc):
        par_dict = {}
        for i, par in enumerate(self.PSYFUNPARS):
            par_val = self.linfits.loc[par, 'slope'] * ecc + self.linfits.loc[par, 'intercept']
            if par_val < self.psyfun_bounds[0][i]:
                par_val = self.psyfun_bounds[0][i]
            elif par_val > self.psyfun_bounds[1][i]:
                par_val = self.psyfun_bounds[1][i]
            par_dict[par] = par_val
        return tuple(par_dict.values())

    def get_psyfun(self, ecc):
        a, b, g, l = self.get_psyfun_pars(ecc)
        # Here we cheat a little bit and ignore the asymptotes to improve numerical stability
        return lambda x: g + (1 - g - l) / (1 + np.exp(-b * (x - a)))
        # return lambda x: 0.5 + (1 - 0.5 - 0.0) / (1 + np.exp(-b * (x - a)))

    def dprime(self, contrast, ecc):
        psyfun = self.get_psyfun(ecc)
        p_correct = psyfun(contrast)
        if p_correct >= 1:
            p_correct = 1 - 10**(-6)
        return np.sqrt(2) * stats.norm.ppf(p_correct)

    def vismap(self, contrast, eccentricities):
        # Only use data from the range we actually measured
        xx = eccentricities.clip(self.eccentricities.min(), self.eccentricities.max())
        xx = eccentricities
        y = np.array([self.dprime(contrast, x) for x in xx])
        def gaussian(x, a, sig):
            return a * np.exp(-1 * x**2 / (2 * sig**2))
        (a, sig), cov = optimize.curve_fit(gaussian, xx, y)
        return gaussian(eccentricities, a, sig)

class BayesianSearcher():
    MAX_ECC = 15
    N_BINS_1D = 15
    X_BINS = Y_BINS = np.linspace(-MAX_ECC, MAX_ECC, N_BINS_1D)
    N_BINS = N_BINS_1D ** 2
    SHAPE = (N_BINS_1D, N_BINS_1D)
    LOCS = np.arange(N_BINS)
    RESPONSE = 0.5
    MAX_FIX = 30
    W_SPACE = np.linspace(-4, 4, 10)
    def __init__(self, vismap, criterion=0.99):
        self.vismap = vismap
        self.criterion = criterion
        # self.prior = 1 / self.N_BINS**2  # uniform prior over all possible locations
        self.prior = np.ones_like(self.LOCS) / len(self.LOCS)

    def _coord2ind(self, x, y):
        ix = np.digitize(x, self.X_BINS) - 1
        iy = np.digitize(y, self.Y_BINS) - 1
        return np.ravel_multi_index((iy, ix), self.SHAPE)

    def _ind2coord(self, i):
        return np.unravel_index(i, self.SHAPE)

    def _get_vismaps(self):
        # Calculate the distances between all points
        xx, yy = np.meshgrid(self.X_BINS, self.Y_BINS)
        diff_x = np.subtract.outer(xx.ravel(), xx.ravel())
        diff_y = np.subtract.outer(yy.ravel(), yy.ravel())
        # Calculate the eccentricities
        self.eccentricities = np.sqrt(diff_x**2 + diff_y**2)
        self.D = self.vismap.vismap(self.target_contrast, self.eccentricities.ravel())
        self.D = self.D.reshape((self.N_BINS, -1))
        # Calculate responses
        self.sig_t = 1 / self.D
        self.W = np.random.normal(self.u, self.sig_t)

    def initialize_trial(self, target_x, target_y, target_contrast):
        self.target_located = False
        self.fix_n = 0  #saccade number
        self.x, self.y = 0, 0  # starting location
        self.I = np.zeros(self.MAX_FIX + 1, dtype='int')
        self.I[self.fix_n] = self._coord2ind(self.x, self.y)
        self.P = np.zeros((self.MAX_FIX + 1, self.N_BINS))
        self.P_C = np.zeros((self.MAX_FIX + 1, self.N_BINS))

        # Get response
        self.target_x = target_x
        self.target_y = target_y
        self.target_i = self._coord2ind(target_x, target_y)
        self.u = -1 * self.RESPONSE * np.ones(self.N_BINS)
        self.u[self.target_i] = self.RESPONSE

        # Pre-compute visibility for all locations in the grid
        self.target_contrast = target_contrast
        self._get_vismaps()

    def _get_visibility(self, x, y):
        # TODO: check if okay to remove this function
        xx, yy = np.meshgrid(self.X_BINS, self.Y_BINS)
        dx = xx - x
        dy = yy - y
        eccentricities = np.sqrt(dx**2 + dy**2)
        vismap = self.vismap.vismap(self.target_contrast, eccentricities.ravel())
        return vismap

    def _get_P_i(self):
        accum = np.sum(self.D[self.I[:self.fix_n + 1]]**2 * self.W[self.I[:self.fix_n + 1]], axis=0)  # sum over fixations
        self.P[self.fix_n] = self.prior * np.exp(accum) / np.sum(self.prior * np.exp(accum))  # sum over spatial locations

    def _get_integrand_old(self, w):
        lr = -2 * np.log(self.P[self.fix_n] / self.P[self.fix_n, self._i])
        vis = self.D[self._k]**2 + 2 * self.D[self._k] * w + self.D[self._k, self._i]**2
        Phi_w = stats.norm.logcdf((lr + vis) / (2 * self.D[self._k]))
        Phi_w[self._i] = 0  # sum over j != i
        return stats.norm.pdf(w) * np.exp(np.sum(Phi_w))

    def _get_integrand_vectorized(self, lr, vis):
        vis = vis[:, np.newaxis] + 2 * self.D[self._k][:, np.newaxis] * self.W_SPACE
        norm = 2 * self.D[self._k][:, np.newaxis]
        cdf_args = (lr[:, np.newaxis] + vis) / norm
        cdf_args[self._i] = 0  # sum over j != i
        Phi_w = stats.norm.logcdf(cdf_args).sum(axis=0)
        return stats.norm.pdf(self.W_SPACE) * np.exp(Phi_w)

    def _get_P_C_given_ik_old(self):
        self.integrand = np.array([self._get_integrand(w) for w in self.W_SPACE])
        self.J_n[self._k, self._i] = np.trapz(self.integrand, self.W_SPACE)

    def _get_P_C_given_ik(self):
        lr = -2 * np.log(self.P[self.fix_n] / self.P[self.fix_n, self._i])
        vis = self.D[self._k]**2 + self.D[self._k, self._i]**2
        integrand = self._get_integrand_vectorized(lr, vis)
        self.J_n[self._k, self._i] = np.trapz(integrand, self.W_SPACE)

    def _get_P_C(self):
        self.J_n = np.zeros((self.N_BINS, self.N_BINS))
        # For all possible next fixation locations
        for _k in tqdm(self.LOCS):
            self._k = _k
            # For all possible target locations
            for _i in self.LOCS:
                self._i = _i
                # if self.P[self.fix_n, self._i] < 10**(-6): continue
                self._get_P_C_given_ik()
        self.P_C[self.fix_n] = np.nansum(self.J_n * self.P[self.fix_n], axis=1)

    def update_posteriors(self):
        self._get_P_i()
        print(f"max(P(T)) = {self.P[self.fix_n].max():.3e}")
        print("Assessing future fixation locations...")
        self._get_P_C()

    def make_saccade(self):
        raise NotImplementedError

    def search(self, plot=True):
        while not self.target_located:
            if self.fix_n >= self.MAX_FIX:
                break
            print(f"Fixation no.: {self.fix_n}")
            print("Updating posterior...")
            self._get_P_i()
            print(f"max(P(T)) = {self.P[self.fix_n].max():.3e}")
            if self.P[self.fix_n].max() >= self.criterion:
                self.target_located = True
                print("Target located!")
            # print("Assessing future fixation locations...")
            # self._get_P_C()
            self.make_saccade()
            if plot:
                fig, axs = plt.subplots(1, 2)
                axs[0].set_title('$P_i(T)$')
                axs[0] = self.plot(fix_n=(self.fix_n - 1), dist='P', ax=axs[0], vmin=-5, vmax=0)
                axs[1].set_title('$P_k(C | i, k(T+1))$')
                axs[1] = self.plot(fix_n=(self.fix_n - 1), dist='P_C', ax=axs[1])
                plt.show()
        ## TODO: fix final plotting (P at fix_n - 1)
        if plot:
            fig, axs = plt.subplots(1, 2)
            axs[0].set_title('$P_i(T)$')
            axs[0] = self.plot(fix_n=(self.fix_n - 1), final=True, dist='P', ax=axs[0], vmin=-5, vmax=0)
            axs[1].set_title('$P_k(C | i, k(T+1))$')
            axs[1] = self.plot(fix_n=(self.fix_n - 1), final=True, dist='P_C', ax=axs[1])
            plt.show()

    def plot(self, fix_n, final=False, dist='P', log=True, ax=None, vmin=None, vmax=None):
        matrix = getattr(self, dist)[fix_n].reshape(self.SHAPE)
        if log:
            matrix = np.log10(matrix)
        if ax is None:
            fig, ax = plt.subplots()
        mat = ax.matshow(matrix, vmin=vmin, vmax=vmax)
        # plt.colorbar(mat)
        noise_xy = (self.X_BINS.searchsorted(0), self.Y_BINS.searchsorted(0))
        noise_circle = Circle(noise_xy, self.N_BINS_1D / 2, lw=2, fc='none', ec='black')
        ax.add_patch(noise_circle)
        target_xy = (self.X_BINS.searchsorted(self.target_x) - 1, self.Y_BINS.searchsorted(self.target_y) - 1)
        ax.add_patch(Circle(target_xy, 1, lw=2, fc='none', ec='red'))
        if final:
            fix_n += 1
        fix_y, fix_x = np.row_stack([self._ind2coord(fix) for fix in self.I[:fix_n + 1]]).T
        ax.plot(fix_x, fix_y, marker='o', color='white')
        ticks = [0, (self.N_BINS_1D - 1) / 2, self.N_BINS_1D - 1]
        ax.set_xticks(ticks)
        ax.set_xticklabels([f'{self.X_BINS[int(tick)]:.0f}' for tick in ticks])
        ax.set_yticks(ticks)
        ax.set_yticklabels([f'{self.Y_BINS[int(tick)]:.0f}' for tick in ticks])
        return ax

    def convert_to_eyedata_series(self):
        assert self.target_located or (self.fix_n == self.MAX_FIX)
        yy, xx = self._ind2coord(self.I[:self.fix_n + 1])
        x_deg = self.X_BINS[xx]
        y_deg = self.Y_BINS[yy]
        eyedata_dict = {
            'fixations':np.column_stack([x_deg, y_deg]),
            'grating_eccentricity':np.sqrt(self.target_x**2 + self.target_y**2),
            'grating_angle':np.arctan2(self.target_y, self.target_x),
            'target_located':self.target_located
            }
        return pd.Series(eyedata_dict)

class BayesianOptimalSearcher(BayesianSearcher):

    def make_saccade(self):
        # Check if criterion is met
        # if self.P[self.fix_n].max() >= self.criterion:
        #     new_location = np.argmax(self.P[self.fix_n])  # saccade to target
        #     self.target_located = True
        #     print("Target located!")
        if self.target_located:
            new_location = np.argmax(self.P[self.fix_n])  # saccade to target
        else:
            print("Assessing future fixation locations...")
            self._get_P_C()
            new_location = np.argmax(self.P_C[self.fix_n])
        self.fix_n += 1
        self.I[self.fix_n] = new_location
