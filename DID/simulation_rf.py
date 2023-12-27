import numpy as np
import doubleml as dml
from doubleml import DoubleMLData
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from joblib import Parallel, delayed
import time
import warnings


def did_cs(n_obs, index):
    np.random.seed(123456 + index)
    warnings.filterwarnings("ignore")

    x1 = np.random.normal(0, 1, n_obs)
    x2 = np.random.normal(0, 1, n_obs)
    u = np.random.normal(0, 1, n_obs)
    x = np.transpose(np.array([x1, x2]))

    f_reg = (x1 + x2 + x1 * x2 + x1 ** 2 + x2 ** 2) / 4
    f_ps = x1 + x2 + u

    d = np.random.binomial(n=1, p=1 / (1 + np.exp(-f_ps)))
    t = np.random.binomial(n=1, p=0.5, size=n_obs)
    nu = np.random.normal(u * f_reg, 1, n_obs)

    y_t0_d0 = (f_reg + nu) + np.random.normal(0, 1, n_obs)
    y_t1_d0 = (4 * f_reg + nu) + np.random.normal(0, 1, n_obs) + 0
    y_t1_d1 = (4 * f_reg + nu) + np.random.normal(0, 1, n_obs) + 1

    y_t0 = y_t0_d0
    y_t1 = y_t1_d1 * d + y_t1_d0 * (1 - d)

    y = t * y_t1 + (1 - t) * y_t0

    dml_data = DoubleMLData.from_arrays(x=x, y=y, d=d, t=t)

    if n_obs == 500:
        ml_g_rf = RandomForestRegressor(n_estimators=18)
        ml_m_rf = RandomForestClassifier(n_estimators=18)
    elif n_obs == 1000:
        ml_g_rf = RandomForestRegressor(n_estimators=18)
        ml_m_rf = RandomForestClassifier(n_estimators=18)
    elif n_obs == 5000:
        ml_g_rf = RandomForestRegressor(n_estimators=18)
        ml_m_rf = RandomForestClassifier(n_estimators=18)
    else:
        ml_g_rf = RandomForestRegressor(n_estimators=18)
        ml_m_rf = RandomForestClassifier(n_estimators=18)

    dml_did_rf = dml.DoubleMLDIDCS(dml_data, ml_g=ml_g_rf, ml_m=ml_m_rf, score='observational', in_sample_normalization=True, n_folds=5)
    dml_did_rf.fit()

    return dml_did_rf.coef[0]

begin = time.time()

n_rep = 10000
num_obs_seq = [500, 1000, 5000, 10000]

for num_obs in num_obs_seq:
    output = Parallel(n_jobs=14)(delayed(did_cs)(num_obs, index) for index in range(n_rep))
    output_rf = np.array(output)

    print(num_obs)
    print(np.mean(output_rf))
    print(np.std(output_rf))

end = time.time()

print(end - begin)