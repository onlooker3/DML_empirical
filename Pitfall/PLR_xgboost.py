import numpy as np
import pandas as pd
import doubleml as dml
from xgboost import XGBRegressor
from xgboost import XGBClassifier
from joblib import Parallel, delayed

import warnings
warnings.filterwarnings("ignore")

def plr_xgboost(num_trees, max_depth, learning_rate, index):
    warnings.filterwarnings("ignore")
    np.random.seed(10001 + index)

    num_sample = 1000

    x1 = np.random.normal(loc=0, scale=2, size=num_sample)
    x2 = np.random.normal(loc=0, scale=2, size=num_sample)
    d = np.random.binomial(n=1, p=1 / (1 + np.exp(-(x1 + x2 + 1))))
    y = d + x1 + x2 + x1 * x2 + np.power(x1, 2) + np.power(x2, 2) + 1 + np.random.normal(loc=0, scale=1, size=num_sample)
    data = pd.DataFrame({'y': y, 'd': d, 'x1': x1, 'x2': x2})

    ml_l = XGBRegressor(n_estimators=num_trees, max_depth=max_depth, learning_rate=learning_rate)
    ml_m = XGBClassifier(n_estimators=num_trees, max_depth=max_depth, learning_rate=learning_rate)

    dml_data = dml.DoubleMLData(data, 'y', 'd')
    dml_plr = dml.DoubleMLPLR(dml_data, ml_l, ml_m)
    dml_plr.fit()

    result_l = dml_plr.evaluate_learners(learners=['ml_l'])['ml_l'][0, 0]
    result_m = dml_plr.evaluate_learners(learners=['ml_m'])['ml_m'][0, 0]
    result_coef = dml_plr.coef[0]

    return result_l, result_m, result_coef


Ntrees = [2, 5, 10, 50, 100, 500]
Max_depth = [2, 5, 7, 10, 20]
Learning_rate = [0.05, 0.1, 0.2]
Nrep = 100

result_pred = np.zeros(shape=len(Ntrees) * len(Max_depth) * len(Learning_rate))
result_est= np.zeros(shape=len(Ntrees) * len(Max_depth) * len(Learning_rate))

n = 0
for n1 in range(len(Ntrees)):
    for n2 in range(len(Max_depth)):
        for n3 in range(len(Learning_rate)):
            output = Parallel(n_jobs=14)(delayed(plr_xgboost)(Ntrees[n1], Max_depth[n2], Learning_rate[n3], index) for index in range(Nrep))
            output_l = np.array([i[0] for i in output])
            output_m = np.array([i[1] for i in output])
            output_coef = np.array([i[2] for i in output])

            prediction_error = np.mean(output_m * output_l)
            estimation_error = np.sqrt(np.mean(np.square(output_coef - 1)))

            result_pred[n] = prediction_error
            result_est[n] = estimation_error

            n = n + 1
            print(n)


print(result_pred)
print(result_est)

np.savetxt('result_xgboost.txt', np.array([result_pred, result_est]), fmt ='%f', delimiter='\t')