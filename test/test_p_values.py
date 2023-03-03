import numpy as np
import pandas
from param_pca import param_pca

def test_null_slow():
    ''' Slow test - of whether p-values are valid for null data '''
    rng = np.random.default_rng(0)

    bs_results_list = []
    for _ in range(50):
        ## EXAMPLE NULL DATA - NO TIME DEPENDENCE
        N = 30 # N independent samples
        data = np.concatenate([
            rng.normal(size=(1,N))*5, # Two larger factors
            rng.normal(size=(1,N))*3,
            rng.normal(size=(10,N)), # 10 independent small factors
        ], axis=0).T

        t = np.linspace(0,1, N) # time
        metadata = pandas.DataFrame({"t": t})

        ## Perform the ParamPCA regression
        result = param_pca.ParamPCA(data, metadata, "t", 2, R = 5)

        ## Perform bootstrap
        bs_results = result.test_parameters(nbootstraps=25)
        bs_results_list.append(bs_results)

    # Collect the p-values
    ps = np.array([res.p[1] for res in bs_results_list])

    # This is a rather fragile test, so not a perfect test
    frac_significant = (ps < 0.05).mean()
    assert frac_significant < 0.05