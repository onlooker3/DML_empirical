import numpy as np
import pandas as pd
import statsmodels.api as sm
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

    y_t0_d0 = (f_reg + nu)+ np.random.normal(0, 1, n_obs)
    y_t1_d0 = (4 * f_reg + nu) + np.random.normal(0, 1, n_obs) + 0
    y_t1_d1 = (4 * f_reg + nu) + np.random.normal(0, 1, n_obs) + 1

    y_t0 = y_t0_d0
    y_t1 = y_t1_d1 * d + y_t1_d0 * (1 - d)

    y = t * y_t1 + (1 - t) * y_t0

    df_y = pd.DataFrame({'y': y})
    df_x = pd.DataFrame({'d': d, 't': t, 'dt': d * t, 'x1': x1, 'x2': x2})
    df_x = sm.add_constant(df_x)
    model = sm.OLS(df_y, df_x)

    return model.fit().params['dt']

begin = time.time()

n_rep = 10000
num_obs_seq = [500, 1000, 5000, 10000]

for num_obs in num_obs_seq:
    output = Parallel(n_jobs=14)(delayed(did_cs)(num_obs, index) for index in range(n_rep))
    output_twfe = np.array(output)

    print(num_obs)
    print(np.mean(output_twfe))
    print(np.std(output_twfe))

end = time.time()

print(end - begin)