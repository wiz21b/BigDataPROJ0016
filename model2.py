from copy import copy
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

INITIAL_POP = 200

def model(initial_conds, params, days=10):

    INITIAL_POP = initial_conds[0]

    values = copy(initial_conds)
    days_data = [values]

    alpha, beta, gamma1, _  = params

    for day in range(days-1):

        s, i, h = values

        # s = suscpetible (ie not yet infected)
        # i = infected
        # h = hospitalised

        ds = -alpha*(s/INITIAL_POP)*i
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
    global learning_set

    nb_observations = len(learning_set[0])

    i_start, h_start, alpha, beta, gamma1, gamma2 = params

    v = np.array(model([INITIAL_POP, i_start, h_start], [alpha, beta, gamma1, gamma2], nb_observations)).transpose()
    e = error(v, learning_set)

    return e


headers, observations, rows = load_data()
print(tabulate(observations, headers=headers))

# Each row is the different values of a model's value (i,r,h,...)
OBS_LENGTH=len(observations)
observations = np.array(rows).transpose()

LS_LENGTH=12
learning_set = np.array(rows[0:LS_LENGTH]).transpose()


varbound=np.array([[0,5],[1,2],[0.01,0.5],[0.01,0.5],[0.01,0.5],[0.2,0.5]])
gamodel = ga( function=gafunc, dimension=6,variable_type='real',variable_boundaries=varbound)
gamodel.run()


i_start, h_start, alpha, beta, gamma1, gamma2 = gamodel.output_dict['variable']
best_v = np.array(model([INITIAL_POP, i_start, h_start], [alpha, beta, gamma1, gamma2], len(rows)*4)).transpose()


plt.plot(learning_set[NUM_POSITIVE], color="darkorange", linestyle="--", label="Positive (training)")
plt.plot(range(LS_LENGTH-1, OBS_LENGTH), observations[NUM_POSITIVE][LS_LENGTH-1:OBS_LENGTH], label="Positive (real)", color="darkorange")
plt.plot(best_v[NUM_POSITIVE], label="Positive (model)", color="darkorange", linestyle=":")

plt.plot(learning_set[NUM_HOSPITALIZED], color="red", linestyle="--", label="Hospitalized (training)")
plt.plot(range(LS_LENGTH-1, OBS_LENGTH), observations[NUM_HOSPITALIZED][LS_LENGTH-1:OBS_LENGTH], label="Hospitalized (real)", color="red")
plt.plot(best_v[2], label="Hospitalized (model)", color="red", linestyle=":")

#plt.plot(best_v[0])
plt.legend()
plt.ylabel('Individuals')
plt.xlabel('Days')
plt.title(f'GA Simulation (population={INITIAL_POP})')
plt.show()
