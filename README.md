# ParamPCA

ParamPCA performs regression of parametrized PCA, which allows the PCA components themselves to depend upon some auxiliary information, such as time.
We refer to these parameterizing factors as `metadata` and we perform PCA on `data`.
The metadata factors can be continuous.

ParamPCA works by non-linear optimization to minimize the squared sum residual of the data after projecting each observation in `data` on to the PCA hyperplane determined by that observations `metadata`.

Parameterizations can be specified as [patsy](https://patsy.readthedocs.io/en/latest/overview.html) formulas (similar to R).
Formulas are single-sided (like `group * age` not `y ~ group * age`) since the left-hand-side is always the provided `data`.

## Example

The following example shows using this method to perform a 'cosinor' style regression of PCA.
Cosinor regression is commonly used for identifying factors with 24 hour periods in circadian medicine and is performed by `y ~ cos(t) + sin(t)`.
This does the ParamPCA equivalent of that.

``` python
import numpy as np
import pandas
from param_pca import ParamPCA

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
```

## Explanation

For $n \times k$ data $X$ PCA to dimension $r$ minimizes the following:

$$ W = \argmin_{W} \left| X - X W W^T \right|^2 $$

where $W$ is required to be an orthogonal $k \times r$ matrix (meaning each of its $r$ columns is unit length and orthogonal to the other columns).
For ParamPCA we allow W to depend upon the metadata $t$ of a sample, $W(t)$.
If each of the $n$ observations $X_i$ in $X$ correspond to metadata $t_i$, then ParamPCA instead minimizes:

$$ W(t) = \argmin_{W(t)} \sum_{i=1}^n \left| X_i - X_i W(t_i) W(t_i)^T \right|^2 $$

where $W(t)$ is determined by the provided formula as a linear function of the metadata.
For example, the formula `t + s` means that 

$$W((t, s)) = expm(A t + B s + C) W_0$$

for some fixed matrices A, B, C corresponding and $W_0$ the weights from the overall PCA and $expm$ the matrix exponential function.
We require that A, B, C are skew-symmteric (so that $W((t,s))$ is always orthogonal) and rank at most $r$.
In particular, we require that they are zero outside of the first $r$ rows and columns which ensures that they have rank at most $r$ while still giving all possible values of $W$.