from datetime import datetime
from pprint import pprint

import numpy as np
import arviz as az
import pymc3 as pm
from pymc3.ode import DifferentialEquation
from scipy.integrate import odeint
import matplotlib.pyplot as plt

from utils import ObsEnum, StateEnum, ObsFitEnum, StateFitEnum, Model, residuals_error, load_data, residual_sum_of_squares, log_residual_sum_of_squares


def ode_model(ys, t, factor):

    S, E, A, SP, H, C, F, R, _, _ = ys[0],ys[1],ys[2],ys[3],ys[4],ys[5],ys[6],ys[7], 0, 0

    gamma1, gamma2, gamma3, gamma4, beta, tau, delta, sigma, rho, theta = factor[0],factor[1],factor[2],factor[3],factor[4],factor[5],factor[6],factor[7],factor[8],factor[9]

    N = 1000000

    dSdt = -beta * S * (A+SP) / N
    dEdt = beta * S * (A+SP) / N - rho * E
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
print(observations[ObsEnum.RSURVIVOR.value])
days = len(observations)
STEPS = len(observations[0])

# Integrate differential equations to provide ground truth
#observations = odeint(ode_model, y0=1, t=range(STEPS), args=([0.2],))

# Add some randomness (this will also create impossible data)
#observations += np.random.random((STEPS,1))*10

#plt.plot(observations)


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

    gamma1 = pm.Bound(pm.Normal, lower=0.02, upper=0.25)("gamma1", mu=0.02, sigma=0.5)
    gamma2 = pm.Bound(pm.Normal, lower=0.02, upper=1)("gamma2", mu=0.74, sigma=0.5)
    gamma3 = pm.Bound(pm.Normal, lower=0.02, upper=1)("gamma3", mu=0.88, sigma=0.5)
    gamma4 = pm.Bound(pm.Normal, lower=0.07, upper=1)("gamma4", mu=0.07, sigma=0.5)
    beta   = pm.Bound(pm.Normal, lower=0.01, upper=3)("beta", mu=1, sigma=0.5)
    tau    = pm.Bound(pm.Normal, lower=0.05, upper=1)("tau", mu=1, sigma=0.5)
    delta  = pm.Bound(pm.Normal, lower=0.09, upper=1)("delta", mu=0.19, sigma=0.5)
    sigma  = pm.Bound(pm.Normal, lower=0.05, upper=0.25)("sigma", mu=0.25, sigma=0.5)
    rho    = pm.Bound(pm.Normal, lower=0.20, upper=1)("rho", mu=1, sigma=0.5)
    theta  = pm.Bound(pm.Normal, lower=0.10, upper=1)("theta", mu=0.10, sigma=0.5)

    # We "connect" b to the ODE model
    ode_solution = pm_ode_model(
        y0=[1,1,1,1, 1,1,1,1, 0, 0],
        theta=[gamma1,gamma2,gamma3,gamma4,beta,tau,delta,sigma,rho,theta])

    # We want to know how close our model (with b)
    # is to the actual observed data
    # HalfNormal means positive normals (because sigma
    # can't be negative when used in the resultX normals)
    error_sigma1 = pm.HalfNormal("error_sigma1", sigma=1)
    error_sigma2 = pm.HalfNormal("error_sigma2", sigma=1)
    error_sigma3 = pm.HalfNormal("error_sigma3", sigma=1)

    # This is where the magic happens.
    # Some call this "likelihood" but I fail to see why.
    # I interpret it as having a normal distribution
    # centered (mu) at each pointsof the observed.
    # And within a distance sigma of each point.

    # (Probabilty of having d as a prediction by h)

    # Using the ".T" was seen here : https://docs.pymc.io/notebooks/ODE_API_shapes_and_benchmarking.html

    result1 = pm.Normal(
        'Hospi',
        mu=ode_solution.T[StateFitEnum.HOSPITALIZED.value], sigma=error_sigma1,
        observed=observations[ObsFitEnum.HOSPITALIZED.value])

    result2 = pm.Normal(
        'Critical',
        mu=ode_solution.T[StateFitEnum.CRITICAL.value], sigma=error_sigma2,
        observed=observations[ObsFitEnum.CRITICAL.value])

    result3 = pm.Normal(
        'Survivor', mu=ode_solution.T[StateFitEnum.RSURVIVOR.value], sigma=error_sigma3,
        observed=observations[ObsFitEnum.RSURVIVOR.value])



# Compute "posteriors"
print(datetime.now())
with basic_model:
    approx = pm.fit(model=basic_model)
    trace = approx.sample()
    print(trace)
    print(pm.summary(trace))




#map_estimate = pm.find_MAP(model=basic_model)
#pprint(map_estimate)
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
