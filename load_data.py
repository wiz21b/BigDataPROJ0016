import csv
import urllib.request
from collections import namedtuple

import numpy as np
from matplotlib import pyplot as plt
# If you don't have this, run "pip install tabulate"
from tabulate import tabulate

from copy import copy
from geneticalgorithm import geneticalgorithm as ga


NUM_POSITIVE = 1
NUM_HOSPITALIZED = 2
NUM_CUMULATIVE_HOSPITALIZATIONS = 3
NUM_CRITICAL = 4
NUM_FATALITIES = 5

INITIAL_POP = 100

def model(values, params, days=10):

    days_data = [values]

    initial_pop = values[0]
    alpha = params[0]
    beta = params[1]
    gamma1 = params[2]
    zeta = params[3]

    for day in range(days):

        s, i, h = values

        # i = infected
        # h = hospitalised

        ds = -alpha*(s/initial_pop)*i
        di = -ds - beta*i
        dh = +beta*i - gamma1*h

        s = max(0, s+ds)
        i = max(0, i+di)
        h = max(0, h+dh)

        values = [s, i, h]

        days_data.append(values)

    return days_data




observations = []
rows = []
positive_cumulated = 0

#with urllib.request.urlopen("https://raw.githubusercontent.com/ADelau/proj0016-epidemic-data/main/data.csv") as fp:
with urllib.request.urlopen("https://raw.githubusercontent.com/ADelau/proj0016-epidemic-data/main/SRAS.csv") as fp:

    data = fp.read().decode("utf-8").split('\n')
    csvreader = csv.reader(data, delimiter=',')

    head = next(csvreader)
    DataTuple = namedtuple("DataTuple", head)

    for r in csvreader:
        if r:
            ir = [int(x) for x in r]
            t = DataTuple(*ir)
            observations.append(t)
            rows.append(ir + [positive_cumulated])

print(tabulate(observations, headers=head))


observations = np.array(rows).transpose()


# def error(values, observations):
#     i = np.log(np.array(values[1][1:]))
#     i_obs = np.log(np.array(observations[NUM_POSITIVE][1:]))
#     assert len(i) == len(i_obs)

#     h = np.log(np.array(values[2][1:]))
#     h_obs = np.log(np.array(observations[NUM_HOSPITALIZED][1:]))

#     # print(i)
#     # print(i_obs)
#     delta_i = i - i_obs
#     delta_h = h - h_obs

#     return np.sum(delta_i*delta_i) + np.sum(delta_h*delta_h)

# def gafunc( params):
#     global observations

#     i_start, h_start, alpha, beta, gamma1, gamma2 = params

#     v = np.array(model([INITIAL_POP, i_start, h_start], [alpha, beta, gamma1, gamma2], len(rows)-1)).transpose()
#     e = error(v, observations)

#     return e


# varbound=np.array([[0,5],[1,2],[0.01,0.5],[0.01,0.5],[0.01,0.5],[0.2,0.5]])

# gamodel = ga( function=gafunc, dimension=6,variable_type='real',variable_boundaries=varbound)
# gamodel.run()

# print( gamodel.output_dict)

# i_start, h_start, alpha, beta, gamma1, gamma2 = gamodel.output_dict['variable']
# best_v = np.array(model([INITIAL_POP, i_start, h_start], [alpha, beta, gamma1, gamma2], len(rows)*4)).transpose()

# def brute_force():
#     global observations

#     best_e = 9999999999
#     best_v = None

#     for h_start in np.arange(1, 10, 1):
#         for i_start in np.arange(1, 10, 1):
#             for alpha in np.arange(0.001, 1, 0.005):
#                 for beta in np.arange(0.0001, 0.2, 0.005):
#                     v = np.array(model([i_start, h_start], [alpha, beta], len(rows)-1)).transpose()
#                     e = error(v, observations)
#                     if e < best_e:
#                         print("-"*80)
#                         print(best_e)
#                         print(alpha, beta)
#                         best_e = e
#                         best_v = v

# plt.plot(best_v[1], label="Positive model")
# plt.plot(observations[NUM_POSITIVE], label="Positive real")
# plt.plot(best_v[2], label="Hospitalized model")
# plt.plot(observations[NUM_HOSPITALIZED], label="Hospitalized real")
# #plt.plot(best_v[0])
# plt.legend()
# plt.ylabel('Individuals')
# plt.xlabel('Days')
# plt.show()

# exit()


rows = np.array(rows)


plt.plot(rows[:, NUM_POSITIVE], label='num_positive')
plt.plot(rows[:, NUM_HOSPITALIZED], label='num_hospitalised')
plt.plot(rows[:, NUM_CUMULATIVE_HOSPITALIZATIONS],
         label='num_cumulative_hospitalizations')
plt.plot(rows[:, NUM_CRITICAL], label='num_critical')
plt.plot(rows[:, NUM_FATALITIES], label='num_fatalities')
plt.legend()
plt.xlabel('Days')
plt.show()
