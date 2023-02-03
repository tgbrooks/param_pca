from __future__ import annotations

from typing import Union, Dict, Any
import textwrap

import patsy
import pandas
import numpy as np
import scipy.linalg
import scipy.stats

import jax
import jax.scipy.linalg
from jax import numpy as jnp

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
        params, W0, RSS_history, niter = fit_param_pca(reduced_data, self.design_matrix, r, verbose=verbose)

        # Store results of fits
        self.params = params
        self.W0 = W0
        self.RSS_history = RSS_history
        self.resid_sum_squares = self.RSS_history[-1]
        self.PCA_resid_sum_squares = self.RSS_history[0]
        self.niter = niter
        self.nvars_reduced = reduced_data.shape[1]

    def A_at(self, metadata: DF_like):
        ''' Value of the A in exp(A)W0 for a specific value of the metadata '''
        # New design matrix modelling the provided point
        dm = patsy.build_design_matrices([self.design_matrix.design_info], metadata)[0]

        # TODO: support multiple values simultaneously from 'metadata'
        # We can only fit a single entry at a time
        assert dm.shape[0] == 1

        A = sum(self.params[term] * dm_val for dm_val, term in zip(dm, dm.design_info.column_names))
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

def eval(params, design, W0, X):
    '''Target function for minimization

    Squared sum residual (SSR) of the projection
    '''
    def func(i, ssr):
        ''' compute SSR of just one  sample '''
        x = X[[i],:]
        sample_d = design[i]
        # Add all terms of A together with the appropriate coefficients
        # from the design matrix entry for the ith sample
        A = jnp.sum(params * sample_d[:, None, None], axis=0)
        L = expm_AATV(A, W0)
        return ssr + jnp.linalg.norm(x - x @ L @ L.T)**2
    ssr = jax.lax.fori_loop(
        0,
        len(X),
        func,
        init_val = jnp.asarray([0.]),
    )
    return ssr / len(X)

def low_rank_weights(X, r):
    ''' give best rank r weights to approximate X '''
    u,d,vt = np.linalg.svd(X, full_matrices=False)
    return vt[:r,:].T

def expm_AATV(A, V, nterms=15):
    ''' Given A, a lower triangular (rectangular) matrix and a tall matrix v
    compute exp(A - A.T) V
    where B = A - A.T is a large (sparse) square matrix with A on the left
    and A.T on the top, zeros in the bottom left block
    Assumes A and V have the same shape
    Much faster than doing np.scipy.linalg.exp(A - A.T) @ V for tall matrices
    due to not having to compute the nxn matrix exponential
    '''
    r = V.shape[1]
    res = V # first term of Taylor expansion
    BkV = V # B^k V / k!  - initializes at V for k=0
    for k in range(1,nterms):
        top = -A.T @ BkV + A[:r] @ BkV[:r]
        bottom = A[r:] @ BkV[:r]
        BkV = BkV.at[:r,:].set(top/k)
        BkV = BkV.at[r:, :].set(bottom/k)
        res += BkV
    return res

expm_AATV = jax.jit(expm_AATV, static_argnums=2)

def fit_param_pca(X, design_matrix, r, verbose=False):
    """
    Compute the optimal r-dimensional subspace that captures the maximum
    variation in X, which is allowed to vary according to `design_matrix` by

    W = exp(A@design_matrix) W0

    where W0 is the rank r PCA estimate of weights,
    W is the weightings for the r-dimensional subspace at metadata x,
    A are skew-symmetric matrices of rank r,
    (specifically they are zero outside of the first r rows and columns,
    and so are parametrized as just n x r matrices)
    and exp() is the matrix exponential function.

    returns params (dict containing terms of A), W0 (array), RSS_history (array), niter (int)
    """

    term_names = design_matrix.design_info.column_names

    def extract(mat, k, r):
        # Return the lower triangular version
        return jnp.concatenate([jnp.zeros((r,r)), mat.reshape((k-r,r))])

    n,k = X.shape
    W0 = jnp.asarray(low_rank_weights(X, r))
    X = jnp.asarray(X)
    design = jnp.asarray(design_matrix)
    N = k*r - r*r

    def f(raw_params):
        # Function to minimize
        params = jnp.array([extract(param, k, r) for param in raw_params])
        return eval(params, design, W0, X)[0]
    val_and_grad = jax.value_and_grad(f, argnums=[0])

    # Optimizer
    beta1 = 0.9
    beta2 = 0.99
    alpha = 0.001 #Learning rate
    epsilon = 1e-8
    @jax.jit
    def update(params, m, v, i):
        # Adam optimizer
        residual, (grad,) = val_and_grad(params)
        m = [beta1 * mX + (1 - beta1) * gradX for mX, gradX in zip(m, grad)]
        v = [beta2 * vX + (1 - beta2) * gradX**2 for vX, gradX in zip(v, grad)]
        mHat = [mX / (1 - beta1**i) for mX in m]
        vHat = [vX / (1 - beta2**i) for vX in v]
        new_params = jnp.array([param - alpha * mHatX / jnp.sqrt(vHatX + epsilon)
                                for param, mHatX, vHatX in zip(params, mHat, vHat)])
        return new_params, m, v, residual

    # Initialize values
    params = jnp.array([jnp.zeros(N) for term in term_names])
    m = [jnp.zeros(N) for term in term_names]
    v = [jnp.zeros(N) for term in term_names]
    resids = []

    # Perform optimization
    i = 0
    for i in range(1500):
        params, m, v, res = update(params, m ,v, i+1)
        resids.append(res)
        if (i % 100) == 0 and verbose:
            print(f"{i}, RSS = {float(res):0.4f}")

    # Extract results
    final_params = {term: extract(param, k, r) for term, param in zip(term_names, params)}
    resids.append(f(params))
    RSS_history = np.array(resids)
    niter = i + 1

    return final_params, W0, RSS_history, niter

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