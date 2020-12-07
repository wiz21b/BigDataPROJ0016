
import numpy as np
import math
from lmfit import minimize, Parameters, report_fit
#from geneticalgorithm import geneticalgorithm as ga
from scipy.optimize import minimize as scipy_minimize
from scipy.stats import binom
from scipy.special import binom as binomial_coeff
import random
import statistics as st
from sklearn.neighbors import KernelDensity

import matplotlib.pyplot as plt
from utils import ObsEnum, StateEnum, ObsFitEnum, StateFitEnum, Model, residuals_error, load_data, residual_sum_of_squares, log_residual_sum_of_squares

# For repeatiblity (but less random numbers)

# import random
# random.seed(1000)
# np.random.seed(1000)

SKIP_OBS=10

class SarahStat(Model):
    def __init__(self, observations, N):

        self._N = N
        nb_observations = observations.shape[0]
        self._nb_observations = observations.shape[0]

        self._observations = np.array(observations)
        self._fittingObservations = observations[np.ix_(range(nb_observations),
                                                        list(map(int, ObsFitEnum)))]
        self._nExperiments = 200

        self._preprocess = True
        self._evaluation = 0

        E0 = 15
        A0 = 10
        SP0 = 5
        H0 = self._observations[0][ObsEnum.NUM_HOSPITALIZED.value]
        C0 = self._observations[0][ObsEnum.NUM_CRITICAL.value]
        R0 = 0
        F0 = 0
        S0 = self._N - E0 - A0 - SP0 - R0 - H0 - C0
        infected_per_day = 0
        R_out_HC = 0
        cumulI = A0 + SP0

        self._initial_conditions = [S0, E0, A0, SP0, H0, C0, F0, R0, infected_per_day,R_out_HC,cumulI]

        print("initial conditions: ", self._initial_conditions)
        self._fit_params = None

    def get_initial_parameters(self):
        """
        # Return:
        # the initial parameters of the model with their bounds to delimit
        # the ranges of possible values
        """
        min_incubation_time = 1
        max_incubation_time = 5
        min_symptomatic_time = 4
        max_symptomatic_time = 10

        error_margin = 0.2
        cumulative_positives = np.cumsum(self._observations[:, ObsEnum.NUM_POSITIVE.value])
        cumulative_hospitalizations = self._observations[:, ObsEnum.CUMULATIVE_HOSPITALIZATIONS.value]

        tau_0 = cumulative_hospitalizations[-1] / cumulative_positives[-2]
        tau_min = tau_0 * (1 - error_margin)
        tau_max = max(1, tau_0 * (1 + error_margin))
        #tau_bounds = [tau_0 * (1 - error_margin), tau_0, max(1, tau_0 * (1 + error_margin))]

        gamma1_max = 1/ min_symptomatic_time
        gamma1_min = 0.02
        gamma1_0 = 1/(max_symptomatic_time + min_symptomatic_time)
        #gamma1_bounds = [gamma1_min, gamma1_0, gamma1_max]

        gamma2_min = 0.02
        gamma2_max = 0.999
        gamma2_0 = 0.5 # arbitrary choice

        gamma3_min = 0.02
        gamma3_max = 0.999
        gamma3_0 = 0.5 # arbitrary choice
        #gamma2_bounds = [0.02, gamma2_0, 1]
        #gamma3_bounds = [0.02, gamma3_0, 1]

        gamma4_max = 1 / min_incubation_time
        gamma4_min = 1 / (max_incubation_time + max_symptomatic_time)
        gamma4_0 = (gamma4_max + gamma4_min) / 2
        #gamma4_bounds = [gamma4_min, gamma4_0, gamma4_max]

        beta_0 = 0.5
        beta_min = 0.01
        beta_max = 2
        #beta_bounds = [beta_min, beta_0, beta_max]

        cumulative_criticals_max = np.sum(self._observations[:, ObsEnum.NUM_CRITICAL.value])
        cumulative_criticals_min = self._observations[-1, ObsEnum.NUM_CRITICAL.value]

        delta_max = cumulative_criticals_max / cumulative_hospitalizations[-2]
        if delta_max < 1:
            tmp = delta_max
            if tmp > 0:
                delta_max = tmp
            else:
                delta_max = 0
        else:
            delta_max = 1
        #delta_max = max(0, min(delta_max, 1))
        delta_min = cumulative_criticals_min / cumulative_hospitalizations[-2]
        #delta_min = min(max(delta_min, 0), 1)
        if delta_min > 0:
            tmp = delta_min
            if tmp < 1:
                delta_min = tmp
            else:
                delta_min = 0
        else:
            delta_min = 0

        delta_0 = delta_min * 0.7 + delta_max * 0.3
        #delta_bounds = [delta_min, delta_0, delta_max]

        # For the period of incubation
        rho_max = 1
        rho_0 = 1/3
        rho_min = 1/5
        #rho_bounds = [rho_min,rho_0,rho_max]

        #For the death...
        theta_min = 0.1
        theta_max = 1
        theta_0 = 0.5
        #theta_bounds = [theta_min,theta_0,theta_max]


        sigma_max = 1/4
        sigma_min = 1/20
        sigma_0 = (sigma_max + sigma_min) / 2
        #sigma_bounds = [sigma_min, sigma_0, sigma_max]

        #Best first estim so far
        b_gamma1 = 0.02712409
        b_gamma2 = 0.50743732
        b_gamma3 = 0.3725912
        b_gamma4 = 0.4588356
        b_beta = 0.77522
        b_tau = 0.660597
        b_delta = 0.153682
        b_sigma = 0.19352783
        b_rho = 0.545133
        b_theta = 0.31228517

        best_preprocess = [b_gamma1, b_gamma2, b_gamma3, b_gamma4, b_beta, b_tau, b_delta, b_sigma, b_rho, b_theta]
        best_preprocess = self._params_array_to_dict(best_preprocess)
        mini = 0

        print(best_preprocess)
        self._preprocess = False

        gamma1_bounds = [gamma1_min, best_preprocess['gamma1'], gamma1_max]
        gamma2_bounds = [gamma2_min, best_preprocess['gamma2'], gamma2_max]
        gamma3_bounds = [gamma3_min, best_preprocess['gamma3'], gamma3_max]
        gamma4_bounds = [gamma4_min, best_preprocess['gamma4'], gamma4_max]
        beta_bounds = [beta_min, best_preprocess['beta'], beta_max]
        tau_bounds = [tau_min, best_preprocess['tau'], tau_max]
        delta_bounds = [delta_min, best_preprocess['delta'], delta_max]
        sigma_bounds = [sigma_min, best_preprocess['sigma'], sigma_max]
        rho_bounds = [rho_min, best_preprocess['rho'], rho_max]
        theta_bounds = [theta_min, best_preprocess['theta'], theta_max]

        bounds = [gamma1_bounds, gamma2_bounds, gamma3_bounds, gamma4_bounds, beta_bounds, tau_bounds, delta_bounds, sigma_bounds,rho_bounds,theta_bounds]
        param_names = ['gamma1', 'gamma2', 'gamma3', 'gamma4', 'beta', 'tau', 'delta', 'sigma','rho','theta']
        params = Parameters()

        for param_str, param_bounds in zip(param_names, bounds):
            params.add(param_str, value=param_bounds[1], min=param_bounds[0], max=param_bounds[2])

        return params


    def _plumb_scipy(self, params, days, error_func=None):

        days = len(self._observations)

        # Sarah's function prefers params as a dictionary
        # so we convert.
        params_as_dict = self._params_array_to_dict(params)

        res = self._predict(self._initial_conditions, days,
                            params_as_dict)

        # data.append([S, E, A, SP, H, C, F, R,
        #             infected_per_day, R_survivor, cumulI,
        #              cumulH, R_out_HC])

        rselect = np.ix_(range(res.shape[0]),
                         [StateEnum.HOSPITALIZED.value,
                          StateEnum.CRITICAL.value,
                          StateEnum.FATALITIES.value,
                          StateEnum.RSURVIVOR.value])

        oselect = np.ix_(range(self._nb_observations),
                         [ObsEnum.NUM_HOSPITALIZED.value,
                          ObsEnum.NUM_CRITICAL.value,
                          ObsEnum.NUM_FATALITIES.value,
                          ObsEnum.RSURVIVOR.value])

        short_obs = self._observations[oselect][10:]
        short_res = res[rselect][10:]

        rel = short_obs
        rel[rel == 0] = 1
        rel = 1/rel

        residuals = (short_res - short_obs)*rel
        least_squares = np.sum(residuals*residuals)
        return least_squares


    def fit_parameters(self, error_func):
        # L-BFGS-B accepts bounds

        np.seterr(all='raise')

        params = self.get_initial_parameters()
        bounds = np.array([(p.min, p.max) for p_name, p in params.items()])

        #self._error_func = error_func

        for p_name, p in params.items():
            print( "{:10s} [{:.2f} - {:.2f}]".format(p_name,p.min, p.max))

        x0 = [ p.value for p_name, p in params.items() ]
        print( "initial guess for params: {}".format(x0))

        #exit()
        res = scipy_minimize(self._plumb_scipy,
                             x0=x0,
                             method='L-BFGS-B',
                             bounds=bounds, args=(error_func,) )

        print(res)

        self._fit_params = self._params_array_to_dict(res.x)

        for p_name, p in params.items():
            print( "{:10s} [{:.2f} - {:.2f}] : {:.2f}".format(p_name,p.min, p.max,self._fit_params[p_name]))

    def predict(self, days):
        res = self._predict(self._initial_conditions, days, self._fit_params)
        return res

    def _predict(self, initial_conditions, days, params):
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

        S, E, A, SP, H, C, F, R, infected_per_day, R_out_HC, cumulI = initial_conditions
        cumulH = 0

        data = []

        for d in range(days):
            ys = [S, E, A, SP, H, C, F, R]
            dSdt, dEdt, dAdt, dSPdt, dHdt, dCdt, dFdt, dRdt, dHOutdt = self._model(ys, gamma1, gamma2, gamma3, gamma4, beta, tau, delta, sigma, rho, theta)

            S += dSdt
            E += dEdt
            A += dAdt
            SP += dSPdt
            H += dHdt
            C += dCdt
            F += dFdt
            R += dRdt

            cumulH += dHOutdt
            R_survivor = cumulH - H - C - F

            data.append([S, E, A, SP, H, C, F, R,
                         infected_per_day, R_survivor, cumulI,
                         cumulH, R_out_HC])

        return np.array(data)


    def _model(self, ys, gamma1, gamma2, gamma3, gamma4, beta, tau, delta, sigma, rho, theta):
        S, E, A, SP, H, C, F, R = ys

        N = self._N

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

        dHOutdt = tau*SP

        return [dSdt, dEdt, dAdt, dSPdt, dHdt, dCdt, dFdt, dRdt, dHOutdt]


    def _params_array_to_dict(self, params):
        return dict(
            zip(['gamma1', 'gamma2', 'gamma3', 'gamma4', 'beta', 'tau', 'delta', 'sigma','rho','theta'],
                params))

if __name__ == "__main__":
    head, observations, rows = load_data()
    rows = np.array(rows)
    days = len(observations)

    ms = SarahStat(rows, 1000000)
    ms.fit_parameters(residual_sum_of_squares)
    sres = ms.predict(250)

    plt.figure()
    plt.title('LM fit')
    for t in [StateEnum.RSURVIVOR, StateEnum.HOSPITALIZED, StateEnum.CRITICAL, StateEnum.FATALITIES]:
        plt.plot(sres[:, t.value], label=f"{t} (model)")

    for u in [ObsEnum.RSURVIVOR, ObsEnum.NUM_HOSPITALIZED, ObsEnum.NUM_CRITICAL,ObsEnum.NUM_FATALITIES]:
        plt.plot(rows[:, u.value], label=f"{u} (real)")


    plt.title('Curve Fitting')
    plt.xlabel('Days')
    plt.ylabel('Individuals')
    prediction_days = 10 # prediction at prediction_days
    plt.xlim(0, days + prediction_days)
    plt.ylim(0, 1000)
    plt.legend()
    plt.savefig('data_fit.pdf')
    plt.savefig(f'data_fit_{days}_days.pdf')
    plt.show()

    plt.figure()
    for t in [StateEnum.EXPOSED, StateEnum.ASYMPTOMATIQUE, StateEnum.SYMPTOMATIQUE ,StateEnum.HOSPITALIZED, StateEnum.CRITICAL, StateEnum.FATALITIES]:
        plt.plot(sres[:, t.value], label=f"{t} (model)")

    plt.title('Exposed - Infectious - Hospitalized - Critical')
    plt.xlabel('Days')
    plt.ylabel('Individuals')
    plt.legend()
    plt.savefig('projection_zoom.pdf')
    plt.savefig(f'projection_zoom_{days}_days.pdf')
    plt.show()

    plt.figure()
    for t in [StateEnum.SUCEPTIBLE, StateEnum.RECOVERED, StateEnum.CUMULI, StateEnum.FATALITIES]:
        plt.plot(sres[:, t.value], label=f"{t} (model)")

    plt.title('States')
    plt.xlabel('Days')
    plt.ylabel('Individuals')
    plt.legend()
    plt.savefig('projection_global.pdf')
    plt.savefig(f'projection_global_{days}_days.pdf')
    plt.show()
