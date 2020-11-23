from datetime import datetime


from pprint import pprint

import arviz as az
import numpy as np
import pymc3 as pm
from pymc3.ode import DifferentialEquation

import matplotlib.pyplot as plt

from utils import ObsEnum, StateEnum, ObsFitEnum, StateFitEnum, Model, residuals_error, load_data, residual_sum_of_squares, log_residual_sum_of_squares

from scipy.integrate import odeint




#TOTAL_POP = 100

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


    # # Just a toy equation (the suspect equation of SIR)

    # # pymc3 is tricky : pop[0], factor[0] all is important

    # dpop = pop[0] * factor[0] * (TOTAL_POP - pop[0])/TOTAL_POP

    # beta = factor[0]


    # dSdt = -beta * S * (A+SP) / N
    # dEdt = beta * S * (A+SP) / N - rho * E


    return dpop



head, observations, rows = load_data()
rows = np.array(rows)
observations = rows.T
print(observations[ObsEnum.RSURVIVOR.value])
days = len(observations)


STEPS = 50

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

    # b is the parameter governing our model.
    # that's the one we want to guess.
    # It's a prior (because it's not based on data).
    gamma1 = pm.Normal("gamma1", mu=0.02, sigma=0.5)
    gamma2 = pm.Normal("gamma2", mu=0.74, sigma=0.5)
    gamma3 = pm.Normal("gamma3", mu=0.88, sigma=0.5)
    gamma4 = pm.Normal("gamma4", mu=0.07, sigma=0.5)
    beta = pm.Normal("beta", mu=1, sigma=0.5)
    tau = pm.Normal("tau", mu=1, sigma=0.5)
    delta = pm.Normal("delta", mu=0.19, sigma=0.5)
    sigma = pm.Normal("sigma", mu=0.25, sigma=0.5)
    rho = pm.Normal("rho", mu=1, sigma=0.5)
    theta = pm.Normal("theta", mu=0.10, sigma=0.5)

    # We "connect" b to the ODE model
    ode_solution = pm_ode_model(
        y0=[1,1,1,1, 1,1,1,1, 0, 0],
        theta=[gamma1,gamma2,gamma3,gamma4,beta,tau,delta,sigma,rho,theta])

    # We want to know how close our model (with b)
    # is to the actual observed data
    error_sigma1 = pm.HalfNormal("error_sigma1", sigma=1)
    error_sigma2 = pm.HalfNormal("error_sigma2", sigma=1)
    error_sigma3 = pm.HalfNormal("error_sigma3", sigma=1)

    # This is where the magic happens.
    # Some call this "likelihood" but I fail to see why.
    # I interpret it as having a normal distribution
    # centered (mu) at each pointsof the observed.
    # And within a distance sigma of each point.

    # (Probabilty of having d as a prediction by h)

    #     S, E, A, SP, H, C, F, R

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
map_estimate = pm.find_MAP(model=basic_model)
pprint(map_estimate)
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
