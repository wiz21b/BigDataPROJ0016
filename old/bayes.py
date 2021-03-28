from datetime import datetime
from pprint import pprint

import numpy as np
import arviz as az
import pymc3 as pm
from pymc3.ode import DifferentialEquation
from scipy.integrate import odeint
import matplotlib.pyplot as plt

from utils import ObsEnum, StateEnum, ObsFitEnum, StateFitEnum, Model, residuals_error, load_data, residual_sum_of_squares, log_residual_sum_of_squares


POP_SIZE = 1000000

def ode_model(ys, t, factor):

    S, E, A, SP, H, C, F, R, _, _ = ys[0],ys[1],ys[2],ys[3],ys[4],ys[5],ys[6],ys[7], 0, 0

    gamma1, gamma2, gamma3, gamma4, beta, tau, delta, sigma, rho, theta = factor[0],factor[1],factor[2],factor[3],factor[4],factor[5],factor[6],factor[7],factor[8],factor[9]

    norm = S/POP_SIZE
    new_exposed = beta * (A+SP) * norm

    dEdt = new_exposed - rho * E

    dSdt = - new_exposed
    #dAdt = rho * E - sigma*E - gamma4 * A
    dAdt = rho * E - sigma * A - gamma4 * A
    #dSPdt = sigma * E - tau * SP - gamma1 * SP
    dSPdt = sigma * A - tau * SP - gamma1 * SP
    dHdt = tau * SP - delta * H - gamma2 * H

    dCdt = delta * H - theta * C - gamma3 * C
    dFdt = theta * C
    dRdt = gamma1 * SP + gamma2 * H + gamma3 * C + gamma4 * A

    R_out_HC = gamma2 * H + gamma3 * C - theta * F

    return [dSdt, dEdt, dAdt, dSPdt, dHdt, dCdt, dFdt, dRdt, 0, R_out_HC]


head, observations, rows = load_data()
rows = np.array(rows)
observations = rows.T
days = len(observations)
STEPS = len(observations[0])
INITIAL_CONDITIONS = [POP_SIZE,1,1,1, 1,1,1,1, 0, 0]

SUMMARY ="""
mean     sd  hdi_3%  hdi_97%  mcse_mean  mcse_sd  ess_mean  ess_sd  ess_bulk  ess_tail  r_hat
gamma1        0.131  0.047   0.044    0.217      0.002    0.002     403.0   387.0     398.0     462.0    NaN
gamma2        0.871  0.075   0.721    0.970      0.003    0.002     497.0   497.0     510.0     473.0    NaN
gamma3        0.898  0.045   0.812    0.967      0.002    0.001     579.0   572.0     524.0     331.0    NaN
gamma4        0.498  0.188   0.182    0.843      0.009    0.006     443.0   443.0     437.0     471.0    NaN
beta          1.072  0.476   0.208    1.866      0.022    0.015     478.0   478.0     478.0     448.0    NaN
tau           0.500  0.164   0.194    0.774      0.007    0.005     559.0   559.0     561.0     513.0    NaN
delta         0.189  0.062   0.106    0.308      0.003    0.002     490.0   477.0     512.0     472.0    NaN
sigma         0.150  0.040   0.083    0.218      0.002    0.001     544.0   544.0     556.0     474.0    NaN
rho           0.569  0.214   0.214    0.912      0.010    0.007     462.0   462.0     495.0     478.0    NaN
theta         0.182  0.034   0.127    0.239      0.002    0.001     387.0   387.0     381.0     394.0    NaN
error_sigma1  9.522  2.570   5.230   14.173      0.117    0.083     484.0   484.0     459.0     420.0    NaN
error_sigma2  6.662  1.620   3.583    9.691      0.074    0.053     479.0   473.0     490.0     472.0    NaN
error_sigma3  9.367  2.484   5.160   13.844      0.122    0.086     416.0   416.0     418.0     380.0    NaN"""


SUMMARY="""
mean     sd  hdi_3%  hdi_97%  mcse_mean  mcse_sd  ess_mean  ess_sd  ess_bulk  ess_tail  r_hat
gamma1        0.220  0.013   0.196    0.241      0.001    0.000     551.0   551.0     556.0     474.0    NaN
gamma2        0.533  0.122   0.320    0.767      0.006    0.004     490.0   485.0     495.0     478.0    NaN
gamma3        0.540  0.127   0.297    0.757      0.005    0.004     560.0   558.0     561.0     513.0    NaN
gamma4        0.220  0.013   0.196    0.240      0.001    0.000     436.0   436.0     459.0     420.0    NaN
beta          0.552  0.139   0.321    0.817      0.006    0.004     502.0   491.0     512.0     472.0    NaN
tau           0.283  0.139   0.089    0.581      0.007    0.005     438.0   429.0     437.0     471.0    NaN
delta         0.531  0.118   0.312    0.728      0.005    0.004     511.0   511.0     510.0     473.0    NaN
sigma         0.095  0.025   0.055    0.144      0.001    0.001     475.0   472.0     490.0     472.0    NaN
rho           0.355  0.068   0.245    0.485      0.004    0.003     379.0   368.0     398.0     462.0    NaN
theta         0.528  0.082   0.361    0.671      0.004    0.003     478.0   478.0     478.0     448.0    NaN
error_sigma1  4.808  1.986   2.018    8.493      0.095    0.069     438.0   411.0     524.0     331.0    NaN
error_sigma2  4.435  1.753   1.754    7.620      0.086    0.061     416.0   416.0     418.0     380.0    NaN
error_sigma3  3.933  1.799   1.223    6.935      0.091    0.065     389.0   389.0     381.0     394.0    NaN"""

lines = iter(SUMMARY.split("\n"))
next(lines)
next(lines)

params = []

print("Variable        Mean  STD   HDI3% HDI97%")
for line in lines:
    if "error_" in line:
        continue

    #print(line)
    ls = line.split()[:5]
    name = ls[0]

    # HDI = Highest Density Interval
    mean, sd, hdi3, hdi97 = [float(x) for x in ls[1:]]
    print(f"{name:15s} {mean:.3f} {sd:.3f} {hdi3:.3f} {hdi97:.3f}")

    params.append(mean)

def array(z):
    return z


MAP = {'beta': array(0.31682162),
'beta_interval__': array(-1.70217264),
'delta': array(0.09),
'delta_interval__': array(-46.89510637),
'error_sigma1': array(14.40658814),
'error_sigma1_log__': array(2.66768561),
'error_sigma2': array(418.0336681),
'error_sigma2_log__': array(6.03556197),
'error_sigma3': array(4.23003013),
'error_sigma3_log__': array(1.44220912),
'error_sigma4': array(4.25093969),
'error_sigma4_log__': array(1.44714006),
'error_sigma5': array(7.99459884),
'error_sigma5_log__': array(2.07876617),
'gamma1': array(0.10053355),
'gamma1_interval__': array(-5.63527248),
'gamma2': array(0.02995032),
'gamma2_interval__': array(-3.09624881),
'gamma3': array(0.25),
'gamma3_interval__': array(25.51182749),
'gamma4': array(0.13017952),
'gamma4_interval__': array(-1.37883115),
'rho': array(1.),
'rho_interval__': array(20.34276008),
'sigma': array(0.25),
'sigma_interval__': array(39.46377817),
'tau': array(0.05),
'tau_interval__': array(-24.19792853),
'theta': array(0.1),
'theta_interval__': array(-135.74228083)}

params = [ MAP['gamma1'],
           MAP['gamma2'],
           MAP['gamma3'],
           MAP['gamma4'],
           MAP['beta'],
           MAP['tau'],
           MAP['delta'],
           MAP['sigma'],
           MAP['rho'],
           MAP['theta'] ]



ode_solution = odeint(ode_model, y0=INITIAL_CONDITIONS, t=range(STEPS*2), args=(params,)).T
# ode_solution = odeint(ode_model, y0=INITIAL_CONDITIONS, t=range(STEPS*100000),
#                       args=([0.0,0.0,0.0,0.0,
#                              1.1,0.0,0.0,0.0,0.0,0.0],)).T

Y_NAMES = ["S", "E", "A", "SP", "H", "C", "F", "R", "/", "R_out_HC"]


for i,name in enumerate(Y_NAMES):
    # if name not in ("S","R","R_out_HC"):
    if name in ("C","H","F"):
        plt.plot( ode_solution[i][:STEPS], label=name)

plt.plot( observations[ObsEnum.NUM_HOSPITALIZED.value], label="H data")
plt.plot( observations[ObsEnum.NUM_CRITICAL.value], label="C data")
plt.plot( observations[ObsEnum.NUM_FATALITIES.value], label="F data")
plt.legend()


plt.figure(2)
for i,name in enumerate(Y_NAMES):
    # if name not in ("S","R","R_out_HC"):
    #if name in ("C","H","F"):
    plt.plot( ode_solution[i], label=name)
plt.legend()

# plt.show()
# exit()





survivor = ([0] + observations[ObsEnum.RSURVIVOR.value]) - (observations[ObsEnum.RSURVIVOR.value] + [0])

print(observations[ObsEnum.RSURVIVOR.value])

print("Making the model...")

# This is a DiffEq definition for pymc3.
pm_ode_model = DifferentialEquation(
    func=ode_model,
    times=[x for x in range(STEPS)],
    n_states=10,  # dimension of the return value of 'func'
    n_theta=10,  # dimension of p (additional parameters)
    t0=0)

# Now we use some declarations to se up the random
# variables and how they are tied together
basic_model = pm.Model()
with basic_model:
    # See : https://docs.pymc.io/notebooks/getting_started.html

    # Prior belief (because it's not based on data).

    # Bound = put bounds around the parameter we're modelling
    # Normal = the parameter is ditributed around some normal distribution

    gamma1 = pm.Bound(pm.Normal, lower=1/10, upper=1/4)("gamma1", mu=(1/10 + 1/4)/2, sigma=0.3)
    gamma2 = pm.Bound(pm.Normal, lower=0.02, upper=1/4)("gamma2", mu=0.74, sigma=0.5)
    gamma3 = pm.Bound(pm.Normal, lower=0.02, upper=1/4)("gamma3", mu=0.88, sigma=0.5)
    gamma4 = pm.Bound(pm.Normal, lower=1/10, upper=1/4)("gamma4", mu=(1/10 + 1/4)/2, sigma=0.5)
    beta   = pm.Bound(pm.Normal, lower=0.01, upper=2)("beta", mu=1, sigma=0.5)
    tau    = pm.Bound(pm.Normal, lower=0.05, upper=1)("tau", mu=1, sigma=0.5)
    delta  = pm.Bound(pm.Normal, lower=0.09, upper=1)("delta", mu=0.19, sigma=0.5)
    sigma  = pm.Bound(pm.Normal, lower=0.05, upper=0.25)("sigma", mu=0.25, sigma=0.5)
    rho    = pm.Bound(pm.Normal, lower=0.20, upper=1)("rho", mu=1, sigma=0.5)
    theta  = pm.Bound(pm.Normal, lower=0.10, upper=1)("theta", mu=0.10, sigma=0.5)

    # We "connect" b to the ODE model
    ode_solution = pm_ode_model(
        y0=INITIAL_CONDITIONS,
        theta=[gamma1,gamma2,gamma3,gamma4,beta,tau,delta,sigma,rho,theta])

    # We want to know how close our model (with b)
    # is to the actual observed data
    # HalfNormal means positive normals (because sigma
    # can't be negative when used in the resultX normals)
    error_sigma1 = pm.HalfNormal("error_sigma1", sigma=10)
    error_sigma2 = pm.HalfNormal("error_sigma2", sigma=10)
    error_sigma3 = pm.HalfNormal("error_sigma3", sigma=10)
    error_sigma4 = pm.HalfNormal("error_sigma4", sigma=10)
    error_sigma5 = pm.HalfNormal("error_sigma5", sigma=10)

    # This is where the magic happens.
    # Some call this "likelihood" but I fail to see why.
    # I interpret it as having a normal distribution
    # centered (mu) at each pointsof the observed.
    # And within a distance sigma of each point.

    # (Probabilty of having d as a prediction by h)

    # Using the ".T" was seen here : https://docs.pymc.io/notebooks/ODE_API_shapes_and_benchmarking.html

    result1 = pm.Normal(
        'CurrentHospitalized',
        mu=ode_solution.T[StateEnum.HOSPITALIZED.value], sigma=error_sigma1,
        observed=observations[ObsEnum.NUM_HOSPITALIZED.value])

    result2 = pm.Normal(
        'SymptomaticTestedPositive',
        mu=ode_solution.T[StateEnum.SYMPTOMATIQUE.value], sigma=error_sigma2,
        observed=observations[ObsEnum.CUMULATIVE_TESTED_POSITIVE.value]/0.8)

    result3 = pm.Normal(
        'CurrentCritical',
        mu=ode_solution.T[StateEnum.CRITICAL.value],
        sigma=error_sigma3,
        observed=observations[ObsEnum.NUM_CRITICAL.value]/0.8)

    result5 = pm.Normal(
        'Survivor', mu=ode_solution.T[StateEnum.RSURVIVOR.value], sigma=error_sigma5,
        observed=observations[ObsEnum.RSURVIVOR.value])

    result4 = pm.Normal(
        'Fatalities',
        mu=ode_solution.T[StateEnum.FATALITIES.value],
        sigma=error_sigma4,
        observed=observations[ObsEnum.NUM_FATALITIES.value])



print(datetime.now())

# Compute "posteriors"

# Fitting

with basic_model:
    approx = pm.fit(model=basic_model)
    trace = approx.sample()
    print(trace)
    print(pm.summary(trace))


# MAP
# map_estimate = pm.find_MAP(model=basic_model)
# pprint(map_estimate)

print(datetime.now())

exit()


# # Show how good the fit is
# fit = odeint(ode_model, y0=1, t=range(STEPS), args=([map_estimate['b']],))
# plt.plot(observations)
# plt.plot(fit)


# Ask pymc3 to do sampling. Sampling is another way
# to figure out the posterior. The advantage is that
# it tells information about confidence.
# But it's super slow

with basic_model:
    trace = pm.sample(return_inferencedata=False, cores=2)
    print(pm.summary(trace))

    # az.plot_trace(trace)
    # data = az.from_pymc3(trace=trace)
    # az.plot_posterior(data, round_to=2, credible_interval=0.95)


plt.show()
