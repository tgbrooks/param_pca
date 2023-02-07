import numpy as np

import jax
from jax import numpy as jnp

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
    u,d,vt = jnp.linalg.svd(X, full_matrices=False)
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

def extract(mat, k, r):
    ''' Return the lower triangular version '''
    return jnp.concatenate([jnp.zeros((r,r)), mat.reshape((k-r,r))])

def objective(raw_params, k, r, design, W0, X):
    ''' ParamPCA objective function to minimize '''
    params = jnp.array([extract(param, k, r) for param in raw_params])
    return eval(params, design, W0, X)[0]
objective_and_grad = jax.value_and_grad(objective, argnums=[0])

def update(params, m, v, i, k, r, design, W0, X, optimizer_config):
    ''' ADAM optimizer to minimize the objective function '''
    beta1 = optimizer_config['beta1']
    beta2 = optimizer_config['beta2']
    alpha = optimizer_config['alpha']
    epsilon = optimizer_config['epsilon']
    # Adam optimizer
    residual, (grad,) = objective_and_grad(params, k, r, design, W0, X)
    m = [beta1 * mX + (1 - beta1) * gradX for mX, gradX in zip(m, grad)]
    v = [beta2 * vX + (1 - beta2) * gradX**2 for vX, gradX in zip(v, grad)]
    mHat = [mX / (1 - beta1**i) for mX in m]
    vHat = [vX / (1 - beta2**i) for vX in v]
    new_params = jnp.array([param - alpha * mHatX / jnp.sqrt(vHatX + epsilon)
                            for param, mHatX, vHatX in zip(params, mHat, vHat)])
    return new_params, m, v, residual
update = jax.jit(update, static_argnums = [4,5])


def fit_param_pca(X, design_matrix, r, learning_rate = 0.001, niter = 1500, verbose=False):
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
    assert niter > 0

    term_names = design_matrix.design_info.column_names

    n,k = X.shape
    W0 = jnp.asarray(low_rank_weights(X, r))
    X = jnp.asarray(X)
    design = jnp.asarray(design_matrix)
    N = k*r - r*r

    # Optimizer
    optimizer_config = dict(
        beta1 = 0.9,
        beta2 = 0.99,
        alpha = learning_rate,
        epsilon = 1e-8,
    )

    # Initialize values
    params = jnp.array([jnp.zeros(N) for term in term_names])
    m = [jnp.zeros(N) for term in term_names]
    v = [jnp.zeros(N) for term in term_names]
    resids = []

    # Perform optimization
    i = 0
    for i in range(niter):
        params, m, v, res = update(
            params, m, v, i+1, k,
            r, design, W0, X,
            optimizer_config,
        )
        resids.append(res)
        if (i % 100) == 0 and verbose:
            print(f"{i}, RSS = {float(res):0.4f}")

    # Extract results
    final_params = {term: extract(param, k, r) for term, param in zip(term_names, params)}
    resids.append(objective(params, k, r, design, W0, X))
    RSS_history = np.array(resids)
    niter = i + 1

    return final_params, W0, RSS_history, niter