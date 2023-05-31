from __future__ import annotations

from typing import Union, Dict, Any
import textwrap
from functools import lru_cache

import patsy
import pandas
import numpy as np
import scipy.linalg
import scipy.stats

import tqdm

from param_pca.internals import fit_param_pca, low_rank_weights, expm_AATV

# Types that look like a pandas DataFrame for our purposes
DF_like = Union[pandas.DataFrame, Dict[str, np.ndarray], np.ndarray]

class ParamPCA:
    '''
    Represents the results of a "parametrized PCA".
    '''

    # Values used to fit
    data: np.ndarray
    metadata: DF_like

    # data matrix after regressing out the feature-specific regression
    # using the same design as the ParamPCA
    residual_data: DF_like

    # If data is a pandas.DataFrame, then this gives the column/feature names from that
    feature_names: pandas.Index | None

    # Patsy design formula of the regression performed
    # where the values are taken from the metadata field
    # Note: just right-hand side, not a full `y ~ x` equation
    # The regressor will be `data`
    formula: str
    design_matrix: patsy.DesignMatrix

    r: int # Rank of reduction
    nobs: int # number of observations
    nvars: int # number of variables measured per observation

    # Results
    # PCA of the data
    W0: DF_like
    # The circadian terms:
    params: Dict[Any, np.ndarray]

    # sum-of-squares of residuals of the
    # projection to the given subspace
    resid_sum_squares: float
    # Initial RSS from projection to the fixed
    # PCA subpsace, constant regardless of `metadata`
    PCA_resid_sum_squares: float

    # Convergence info
    niter: int # number of iterations
    RSS_history: np.ndarray # list of RSS during the fit, to check for convergence

    # Optimization parameters:
    learning_rate: float

    # Reduction to lower dimension prior to circ PCA info:
    nvars_reduced: int # number of variables reduced to by PCA before performing circ PCA
    reduction_weights: Union[DF_like, None] #Weights of the original variables used to perform the reduction

    def __init__(
            self,
            data: DF_like,
            metadata: DF_like,
            formula: str,
            r: int,
            R: Union[int, None] = None,
            standardize: bool = True,
            learning_rate: float = 0.001,
            niter: int = 1500,
            verbose: bool = False
        ):
        '''
        Compute the optimal r-dimensional subspace that captures the maximum
        variation in X, which is allowed to vary according to `times` by

        W(t) = exp(A(t)) W0, and
        A(t) ~ user-provided formula on the metadata

        where W0 is the rank r PCA estimate of weights,
        W(t) is the weights for the r-dimensional subspace given metadata x
        A(t) is a skew-symmetric, rank-r matrix given metadata x
        (specifically it is zero outside of the first r rows and columns,
        and so are parametrized as just n x r matrices)
        and exp() is the matrix exponential function.

        verbose: if True, print out during iterations
        R: number of dimensions to reduce to (via PCA) prior to performing circ PCA analysis.
            Set this to lower numbers (<100) to speed up large datasets, if they are well-captured
            by a PCA of this dimension.
            If None, then no reduction is performed
        '''

        if isinstance(data, pandas.DataFrame):
            self.feature_names = data.columns
        else:
            self.feature_names = None

        self.data = np.array(data)
        self.metadata = metadata
        self.formula = formula
        self.design_matrix = patsy.dmatrix(self.formula, self.metadata)

        self.r = r

        assert self.data.shape[0] == len(metadata), f"Expected data and metadata to have the same number of rows, instead had {data.shape[0]} and {len(metadata)}"
        assert r > 0
        N,k = self.data.shape

        self.nvars = k
        self.nobs = N
        self.learning_rate = learning_rate

        if k > 100 and R is None:
            print(f"Warning: large number of columns may make this procedure slow. Recommended to use parameter R < 100 to reduce first")

        num_terms = len(self.design_matrix.design_info.column_names)
        if num_terms * r >= N:
            raise ValueError(f"Requested dimension r={r} is too high for the provided number of observations N={N}. Must have {num_terms}r < N")
        if num_terms * r >= k:
            raise ValueError(f"Requested dimension r={r} is too high for the provided number of variables k={k}. Must have {num_terms}r < k")

        # First, we need to regress out the factors in each of the individual variables
        # so that there aren't mean-level effects
        # Solve by least squares using the usual QR formulation
        # We assume the same design matrix for both PCA and this mean-level regression
        Q, R_ = np.linalg.qr(self.design_matrix)
        regression_coeffs = scipy.linalg.solve_triangular(
            R_,
            Q.T @ self.data,
        )
        self.residual_data = self.data - self.design_matrix @ regression_coeffs

        if standardize:
            self.residual_data = scipy.stats.zscore(self.residual_data, axis=1)

        if R is not None:
            if num_terms * r >= R:
                raise ValueError(f"Requested dimension r={r} is too high for the provided number of reduced variables R={R}. Must have {num_terms}r < R")
            if R > k:
                raise ValueError(f"Requested reduction to R={R} dimensions canot be higher than the provided number of variables k={k}.")

            # Reduce to the specific number of variables first before
            # performing Param PCA
            reduction_weights = low_rank_weights(self.residual_data, R)
            reduced_data = self.residual_data @ reduction_weights
        else:
            reduction_weights = None
            reduced_data = self.residual_data
        self.reduction_weights = reduction_weights

        # Calculate the fits
        params, W0, RSS_history, niter, get_hess = fit_param_pca(
            reduced_data,
            self.design_matrix,
            r,
            learning_rate = learning_rate,
            niter = niter,
            verbose=verbose,
        )

        # Store results of fits
        self.params = params
        self.W0 = W0
        self.RSS_history = RSS_history
        self.resid_sum_squares = self.RSS_history[-1]
        self.PCA_resid_sum_squares = self.RSS_history[0]
        self.niter = niter
        self.nvars_reduced = reduced_data.shape[1]
        self._get_hessian = get_hess

    def A_at(self, metadata: DF_like):
        ''' Value of the A in exp(A)W0 for a specific value of the metadata '''
        # New design matrix modelling the provided point
        dm = patsy.build_design_matrices([self.design_matrix.design_info], metadata)[0]

        # TODO: support multiple values simultaneously from 'metadata'
        # We can only fit a single entry at a time
        assert dm.shape[0] == 1

        A = sum(self.params[term] * dm_val for dm_val, term in zip(dm[0], dm.design_info.column_names))
        return A

    def PCA_weights_at(self, metadata: DF_like):
        ''' return nvars x r matrix of PCA weights at the given value of metadata

        metadata should be shaped like a single row of the original metadata
        used in the regression.
        '''

        A = self.A_at(metadata)

        w = expm_AATV(A, self.W0)

        if self.reduction_weights is not None:
            weights = self.reduction_weights @ w
        else:
            weights = w

        if self.feature_names is not None:
            # Add col/row names if available
            return pandas.DataFrame(
                weights,
                index=self.feature_names,
                columns=self.pca_component_names(),
            )
        else:
            return weights

    @lru_cache(None)
    def hessian(self):
        return self._get_hessian()

    def PCA_scores_at(self, metadata: DF_like):
        ''' return nobs x r matrix of PCA scores at the given value of metadata

        metadata should be shaped like a single row of the original metadata
        used in the regression.
        '''
        weights = self.PCA_weights_at(metadata)
        return self.residual_data @ weights

    def pca_component_names(self):
        return [f"PCA_{i+1}" for i in range(self.r)]

    def angles(self, metadata: DF_like, from_mat=None):
        ''' Compute the angles between the r-dimensional subspace at a point `metadata`

        from_mat: matrix to obtain angles with respect to. If None (default),
            then from_mat will be the overall PCA subspace
        '''
        weights = self.weights(metadata)
        if from_mat is None:
            from_mat = self.W0
        return scipy.linalg.subspace_angles(weights, from_mat)

    def summary(self):
        ''' Summarize the results of the structure '''
        def subs(i: int):
            # Subscript unicode versions of i
            if i < 10:
                return "₀₁₂₃₄₅₆₇₈₉"[i]
            else:
                return subs(i//10) + subs(i % 10)
        formula = " + ".join(f"A{subs(i)} {term if term != 'Intercept' else ''}" for i, term in enumerate(self.design_matrix.design_info.term_names))
        terms = ','.join(str(t) for t in self.design_matrix.design_info.term_names if t != 'Intercept')
        param_norms = {term: np.linalg.norm(p, ord=float('inf'))**2
                            for term, p in self.params.items()}
        param_norm_table = "\n        ".join(f"{term: >12}: {norm:0.3e}" for term, norm in param_norms.items())
        return textwrap.dedent(f'''
        ParamPCA results:
        W({terms}) = exp({formula}) W₀

        nobs: {self.data.shape[0]}  nvars: {self.nvars} reduced to nvars: {self.nvars_reduced}
        rank r: {self.r}

        niter: {self.niter}
              PCA RSS: {self.PCA_resid_sum_squares:0.3f}
        PARAM PCA RSS: {self.resid_sum_squares:0.3f}

        L-infinity NORMS:
        {param_norm_table}
        ''').strip()

    def check_injectivity(self):
        ''' Compare size of the 'A' matrix to the injectivity radius for all observations

        The injectivity radius gives the point where the matrix exponential becomes non-injective
        (and so not one-to-one). If the magnitude of the A's is past this injectivity radius, then
        we expect that mapping is over-fit, though there may be cases where that is not true.

        If the returned value is less than 1, then all observations are within the injectivity radius
        of the fit value.
        '''
        worst_ratio = 0
        for i in range(self.nobs):
            metadata = {k: v[i] for k,v in self.metadata.items()}
            A = self.A_at(metadata)
            singular_values = np.linalg.svd(A, compute_uv = False, full_matrices = False)
            worst_ratio = max(worst_ratio, singular_values[0] / (np.pi/2))
        return worst_ratio

    def plot_RSS_history(self):
        import pylab
        fig, ax = pylab.subplots()
        ax.plot(
            self.RSS_history
        )
        ax.set_xlabel("Iteration count")
        ax.set_ylabel("Residual Sum of Squares")
        return fig

    @lru_cache(maxsize=None)
    def bootstrap(self, nbootstraps=500, seed=0):
        ''' Compute bootstrap estimates of the parameters to estimate variability '''
        if self.reduction_weights is not None:
            reduced_data = self.residual_data @ self.reduction_weights
        else:
            reduced_data = self.residual_data

        rng = np.random.default_rng(seed)

        bootstrap_params = []
        for i in tqdm.tqdm(range(nbootstraps)):
            # Select bootstrap data
            selection = rng.choice(len(reduced_data), size=len(reduced_data))
            bootstrap_data = reduced_data[selection]
            bootstrap_design = patsy.DesignMatrix(
                self.design_matrix[selection],
                self.design_matrix.design_info,
            )
            # Calculate the fit
            params, _, _, _, _ = fit_param_pca(
                bootstrap_data,
                bootstrap_design,
                self.r,
                learning_rate = self.learning_rate,
                niter = self.niter,
                verbose=False,
            )
            bootstrap_params.append(params)
        return bootstrap_params

    def test_parameters(self, nbootstraps=500, seed=0):
        ''' Estimate variability via bootstrap '''
        bootstrap_params = self.bootstrap(nbootstraps, seed)

        # Assess the results of the bootstrap
        bs_results = []
        for term, fit_value in self.params.items():
            bs_params = np.array([x[term].flatten() for x in bootstrap_params])
            bs_mean = np.mean(bs_params, axis=0)
            bs_diff = bs_params - bs_mean
            bs_variance = np.sum( bs_diff[:,:,None] @ bs_diff[:,None,:], axis=0)  /(len(bootstrap_params) - 1)

            # We can only have at most n-1 non-zero eigenvalues if we have done n bootstraps
            # so we add a bit to the variance structure (making this conservative) but allowing it to be inverted
            eigs, _ = np.linalg.eigh(bs_variance)
            last_nonzero_eig = eigs[-min(len(bootstrap_params)-1, len(eigs))]
            bs_variance += last_nonzero_eig * np.eye(len(bs_variance))

            # Compute the Mahalanobis distance to the 0 parameter value
            metric = np.linalg.inv(bs_variance)
            dist_to_zero = np.sqrt(bs_mean.T @ metric @ bs_mean)

            # Hotelling t^2 statistic test
            n = len(bootstrap_params)
            p = len(bs_mean)
            t_squared = (n-p)/(p*(n-1)) * dist_to_zero**2
            if n <= p:
                # Can't do statistics, too few bootstraps
                p_to_zero = float("NaN")
                status = "too few bootstraps"
            else:
                p_to_zero = scipy.stats.f(p, n-p).sf(t_squared)
                status = "valid"

            bs_results.append({
                "term": term,
                "p": p_to_zero,
                "status": status,
            })

        return pandas.DataFrame(bs_results)

if __name__ == '__main__':
    ## EXAMPLE DATA
    np.random.seed(0)
    N = 500 # N independent samples
    t = np.linspace(0,2 * np.pi, N) # time
    scores = np.random.normal(size=N) # Simulate one component of common variation
    data = np.concatenate([
        [np.cos(t/2)**2 * scores*5 + scores * 10], # Component largest in this at t=0
        [np.sin(t/2)**2 * scores*5 + scores * 10], # Component largest in this at t=pi (i.e., 12 hours)
        np.random.normal(size=(1,N))*5,
        np.random.normal(size=(1,N))*5,
        np.random.normal(size=(100,N)),
    ], axis=0).T
    metadata = pandas.DataFrame({"t": t})

    ## Perform the ParamPCA regression
    result = ParamPCA(data, metadata, "np.cos(t) + np.sin(t)", 3, R = 10, verbose=True)
    print(result.summary())