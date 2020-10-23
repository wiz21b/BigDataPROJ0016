import numpy as np
import random
import pandas as pd
from scipy.integrate import odeint, solve_ivp
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from matplotlib import dates
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime
from lmfit import minimize, Parameters, Parameter, report_fit
import csv
import urllib.request
from collections import namedtuple
#import requests

observations = []
rows = []
positive_cumulated = 0

with urllib.request.urlopen("https://raw.githubusercontent.com/ADelau/proj0016-epidemic-data/main/data.csv") as fp:

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
            
rows = np.array(rows)
rows_tr = np.array(rows).transpose()

#for positive

NUM_POSITIVE = 1
NUM_HOSPITALIZED = 2
NUM_CUMULATIVE_HOSPITALIZATIONS = 3
NUM_CRITICAL = 4
NUM_FATALITIES = 5
N = 1000000

def SIHCR_model(y, t, gamma1, gamma2, gamma3, beta, tau, delta):
    S, I, H, C, R = y

    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma1 * I - tau * I
    dHdt = tau * I - gamma2 * H - delta * H
    dCdt = delta * H - gamma3 * C
    dRdt = gamma1 * I + gamma2 * H + gamma3 * C
    return [dSdt, dIdt, dHdt, dCdt, dRdt]

def ode_solver(t, IC, params):
    S0, I0, H0, C0, R0 = IC
    gamma1 = params['gamma1']
    gamma2 = params['gamma2']
    gamma3 = params['gamma3']
    beta = params['beta']
    tau = params['tau']
    delta = params['delta']

    res = odeint(SIHCR_model, [S0, I0, H0, C0, R0], t, args=(gamma1, gamma2, gamma3, beta, tau, delta))
    return res

def error(params, IC, tspan, data):
    sol = ode_solver(tspan, IC, params)
    return (sol[:, 1:4] - data).ravel()

I0 = rows[0][NUM_POSITIVE]
H0 = rows[0][NUM_HOSPITALIZED]
C0 = rows[0][NUM_CRITICAL]
R0 = 0
S0 = N - I0 - R0 - H0 - C0

initial_conditions = [S0, I0, H0, C0, R0]

gamma1 = 0.02
gamma2 = 0.02
gamma3 = 0.02
beta = 1.14
tau = 0.02
delta = 0.02

params = Parameters()
params.add('gamma1', value=gamma1, min=0, max=5)
params.add('gamma2', value=gamma2, min=0, max=5)
params.add('gamma3', value=gamma3, min=0, max=5)
params.add('beta', value = beta, min=0, max= 5)
params.add('tau', value=tau, min=0, max=5)
params.add('delta', value=delta, min=0, max=5)

days = 19
tspan = np.arange(0, days, 1)

#data = observations.loc[0:(days-1), ['num_positive', 'num_hospitalised', 'num_critical']].values
data = np.delete(rows_tr, 6, 0)
data = np.delete(data, 5, 0)
data = np.delete(data, 3, 0)
data = np.delete(data, 0, 0)

data = np.array(data).transpose()

result = minimize(error, params, args=(initial_conditions, tspan, data), method = 'leastsq')

report_fit(result)

final = data + result.residual.reshape(data.shape)
fin = np.array(final).transpose()

plt.plot(tspan, rows_tr[1],'o', c='k', label='actuel positifs')
plt.plot(tspan, fin[0], '--', linewidth = 2, c='red', label= 'best fit')

plt.plot(tspan, rows_tr[2],'+', c='k', label='actuel num_hospitalised')
plt.plot(tspan, fin[1], '--', linewidth = 2, c='red', label= 'best fit hosp')

plt.xlabel('Days')
plt.ylabel('Cas')
plt.show()