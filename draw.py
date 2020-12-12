import numpy as np
import math
from lmfit import minimize, Parameters, report_fit
#from geneticalgorithm import geneticalgorithm as ga
from scipy.optimize import minimize as scipy_minimize
from scipy.stats import binom, scoreatpercentile
from scipy.special import binom as binomial_coeff
import random
import statistics as st

import matplotlib.pyplot as plt
from utils import ObsEnum, StateEnum, ObsFitEnum, StateFitEnum, Model, residuals_error, load_data, residual_sum_of_squares, log_residual_sum_of_squares

SKIP_OBS=10

def _params_array_to_dict(params):
    return dict(
        zip(['gamma1', 'gamma2', 'gamma3', 'gamma4', 'beta', 'tau', 'delta', 'sigma','rho','theta','mu','eta'],
            params))

def draw_conf(days, params):
    head, observations, rows = load_data()

    N = 1000000
    rows = np.array(rows)
    nb_observations = rows.shape[0]

    #observations = np.array(rows)

    perc50, perc5, perc95 = confidence(nb_observations, days, params)

    dday = np.linspace(0, (days-1), num=days)

    plt.figure()
    plt.title('HOSPITALIZED')
    t = StateEnum.HOSPITALIZED
    plt.plot(perc50[:, t.value], label=f"{t} (model)")

    u = ObsEnum.NUM_HOSPITALIZED
    plt.plot(rows[:, u.value], label=f"{u} (real)")
        
    plt.fill_between(dday, perc5[:, t.value], perc95[:, t.value], alpha=0.2)
    plt.xlabel('Days')
    plt.ylabel('Individuals')
    plt.xlim(0, days)
    plt.ylim(0, 250)
    plt.legend()
    plt.savefig('Conf_hosp.pdf')
    plt.show()

    plt.figure()
    plt.title('HOSPITALIZED')
    t = StateEnum.CRITICAL
    plt.plot(perc50[:, t.value], label=f"{t} (model)")

    u = ObsEnum.NUM_CRITICAL
    plt.plot(rows[:, u.value], label=f"{u} (real)")
        
    plt.fill_between(dday, perc5[:, t.value], perc95[:, t.value], alpha=0.2)
    plt.xlabel('Days')
    plt.ylabel('Individuals')
    plt.xlim(0, days)
    plt.ylim(0, 250)
    plt.legend()
    plt.savefig('Conf_critical.pdf')
    plt.show()

    plt.figure()
    plt.title('HOSPITALIZED')
    t = StateEnum.FATALITIES
    plt.plot(perc50[:, t.value], label=f"{t} (model)")

    u = ObsEnum.NUM_FATALITIES
    plt.plot(rows[:, u.value], label=f"{u} (real)")
        
    plt.fill_between(dday, perc5[:, t.value], perc95[:, t.value], alpha=0.2)
    plt.xlabel('Days')
    plt.ylabel('Individuals')
    plt.xlim(0, days)
    plt.ylim(0, 150)
    plt.legend()
    plt.savefig('Conf_fatalities.pdf')
    plt.show()
    return

def binomial_dist(param, population):
    if ((population < 0) or (param < 0)):
        # Fix for some edge cases
        return 0

    average = param * population

    r = np.random.binomial( round(2*average), 0.5)

    return r

def confidence(nb_observations, days, params):
    N = 1000000
    E0 = 15 * 3
    A0 = 10 * 3
    SP0 = 5 * 3
    H0 = 1
    C0 = 0
    R0 = 0
    F0 = 0
    S0 = N - E0 - A0 - SP0 - R0 - H0 - C0

    initial_conditions = [S0, E0, A0, SP0, H0, C0, F0, R0]

    nExperiments = 100

    gamma1 = params['gamma1']
    gamma2 = params['gamma2']
    gamma3 = params['gamma3']
    gamma4 = params['gamma4']
    beta = params['beta']
    tau = params['tau']
    delta = params['delta']
    sigma = params['sigma']
    rho = params['rho']
    theta = params['theta']
    mu = params['mu']
    eta = params['eta']

    values = initial_conditions

    S, E, A, SP, H, C, F, R = values
    cumulI = A+ SP
    cumulH = H

    values = initial_conditions
    perc50 = [values]
    perc5 = [values]
    perc95 = [values]

    #Day t-1
    S_prev = np.full((1, nExperiments), S).ravel()
    E_prev = np.full((1, nExperiments), E).ravel()
    A_prev = np.full((1, nExperiments), A).ravel()
    SP_prev = np.full((1, nExperiments), SP).ravel()
    H_prev = np.full((1, nExperiments), H).ravel()
    C_prev = np.full((1, nExperiments), C).ravel()
    F_prev = np.full((1, nExperiments), F).ravel()
    R_prev = np.full((1, nExperiments), R).ravel()

    #Day t
    S_next = np.full((1, nExperiments), 0).ravel()
    E_next = np.full((1, nExperiments), 0).ravel()
    A_next = np.full((1, nExperiments), 0).ravel()
    SP_next = np.full((1, nExperiments), 0).ravel()
    H_next = np.full((1, nExperiments), 0).ravel()
    C_next = np.full((1, nExperiments), 0).ravel()
    F_next = np.full((1, nExperiments), 0).ravel()
    R_next = np.full((1, nExperiments), 0).ravel()

    for day in range(days - 1):
        for exp in range(nExperiments):
            S = S_prev[exp]
            E = E_prev[exp]
            A = A_prev[exp]
            SP = SP_prev[exp]
            H = H_prev[exp]
            C = C_prev[exp]
            F = F_prev[exp]
            R = R_prev[exp]

            beta_S = binomial_dist((beta*S/N), A+SP)
            rho_E = binomial_dist(rho, E)
            sigma_A = binomial_dist(sigma, A)
            gamma4_A = binomial_dist(gamma4, A)
            tau_SP = binomial_dist(tau, SP)
            gamma1_SP = binomial_dist(tau, SP)
            delta_H = binomial_dist(delta, H)
            gamma2_H = binomial_dist(gamma2, H)
            theta_C = binomial_dist(theta, C)
            gamma3_C = binomial_dist(gamma3, C)

            dSdt = -beta_S
            dEdt = beta_S - rho_E
            dAdt = rho_E - sigma_A - gamma4_A
            dSPdt = sigma_A - tau_SP - gamma1_SP
            dHdt = tau_SP - delta_H - gamma2_H
            dCdt = delta_H - theta_C - gamma3_C
            dFdt = theta_C
            dRdt = gamma1_SP + gamma2_H + gamma3_C + gamma4_A

            S_next[exp] = S + dSdt
            E_next[exp] = E + dEdt
            A_next[exp] = A + dAdt
            SP_next[exp] = SP + dSPdt
            H_next[exp] = H + dHdt
            C_next[exp] = C + dCdt
            F_next[exp] = F + dFdt
            R_next[exp] = R + dRdt

        S_prev = S_next
        E_prev = E_next
        A_prev = A_next
        SP_prev = SP_next
        H_prev = H_next
        C_prev = C_next
        F_prev = F_next
        R_prev = R_next

        S50 = scoreatpercentile(S_next, 50)
        S5 = scoreatpercentile(S_next, 5)
        S95 = scoreatpercentile(S_next, 95)
        E50 = scoreatpercentile(E_next, 50)
        E5 = scoreatpercentile(E_next, 5)
        E95 = scoreatpercentile(E_next, 95)
        A50 = scoreatpercentile(A_next, 50)
        A5 = scoreatpercentile(A_next, 5)
        A95 = scoreatpercentile(A_next, 95)
        SP50 = scoreatpercentile(SP_next, 50)
        SP5 = scoreatpercentile(SP_next, 5)
        SP95 = scoreatpercentile(SP_next, 95)
        H50 = scoreatpercentile(H_next, 50)
        H5 = scoreatpercentile(H_next, 5)
        H95 = scoreatpercentile(H_next, 95)
        C50 = scoreatpercentile(C_next, 50)
        C5 = scoreatpercentile(C_next, 5)
        C95 = scoreatpercentile(C_next, 95)
        F50 = scoreatpercentile(F_next, 50)
        F5 = scoreatpercentile(F_next, 5)
        F95 = scoreatpercentile(F_next, 95)
        R50 = scoreatpercentile(R_next, 50)
        R5 = scoreatpercentile(R_next, 5)
        R95 = scoreatpercentile(R_next, 95)

        val50 = [S50, E50, A50, SP50, H50, C50, F50, R50]
        val5 = [S5, E5, A5, SP5, H5, C5, F5, R5]
        val95 = [S95, E95, A95, SP95, H95, C95, F95, R95]

        perc50.append(val50)
        perc5.append(val5)
        perc95.append(val95)
        
    return (np.array(perc50), np.array(perc5), np.array(perc95))

if __name__ == "__main__":
    gamma1 = 0.02
    gamma2 = 0.06
    gamma3 = 0.2
    gamma4 = 0.5333
    beta = 0.3714
    tau = 0.05
    delta = 0.07
    sigma = 0.25
    rho = 0.89
    theta = 0.10
    mu = 0
    eta = 0

    params=[gamma1, gamma2, gamma3, gamma4, beta, tau, delta, sigma, rho, theta, mu, eta]
    params = _params_array_to_dict(params)

    draw_conf(100, params)