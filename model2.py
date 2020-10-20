
import numpy as np
from matplotlib import pyplot as plt
# pip3 install tabulate
from tabulate import tabulate
# pip3 install geneticalgorithm
from geneticalgorithm import geneticalgorithm as ga

from utils import load_data

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

    for day in range(days):

        s, i, h = values

        # s = suscpetible (ie not yet infected)
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


def error(values, observations):
    i = np.log(np.array(values[1][1:]))
    i_obs = np.log(np.array(observations[NUM_POSITIVE][1:]))
    assert len(i) == len(i_obs)

    h = np.log(np.array(values[2][1:]))
    h_obs = np.log(np.array(observations[NUM_HOSPITALIZED][1:]))

    delta_i = i - i_obs
    delta_h = h - h_obs

    # FIXME Error on i is more valued than error on h !
    return np.sum(delta_i*delta_i) + np.sum(delta_h*delta_h)


def gafunc( params):
    # Plumbing function to run the GA algo
    global observations

    i_start, h_start, alpha, beta, gamma1, gamma2 = params

    v = np.array(model([INITIAL_POP, i_start, h_start], [alpha, beta, gamma1, gamma2], len(rows)-1)).transpose()
    e = error(v, observations)

    return e


headers, observations, rows = load_data()
print(tabulate(observations, headers=headers))
observations = np.array(rows).transpose()

varbound=np.array([[0,5],[1,2],[0.01,0.5],[0.01,0.5],[0.01,0.5],[0.2,0.5]])
gamodel = ga( function=gafunc, dimension=6,variable_type='real',variable_boundaries=varbound)
gamodel.run()

print( gamodel.output_dict)

i_start, h_start, alpha, beta, gamma1, gamma2 = gamodel.output_dict['variable']
best_v = np.array(model([INITIAL_POP, i_start, h_start], [alpha, beta, gamma1, gamma2], len(rows)*4)).transpose()

plt.plot(best_v[1], label="Positive model")
plt.plot(observations[NUM_POSITIVE], label="Positive real")
plt.plot(best_v[2], label="Hospitalized model")
plt.plot(observations[NUM_HOSPITALIZED], label="Hospitalized real")
#plt.plot(best_v[0])
plt.legend()
plt.ylabel('Individuals')
plt.xlabel('Days')
plt.show()
