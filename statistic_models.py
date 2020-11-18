# coding=utf-8
import numpy as np
import math
from lmfit import minimize, Parameters, report_fit
from geneticalgorithm import geneticalgorithm as ga
from scipy.optimize import minimize as scipy_minimize
import random

import matplotlib.pyplot as plt
from utils import ObsEnum, StateEnum, ObsFitEnum, StateFitEnum, Model, residuals_error, load_data, residual_sum_of_squares, log_residual_sum_of_squares

SKIP_OBS=10


class Sarah1(Model):

    def __init__(self, observations, N):

        self._N = N
        nb_observations = observations.shape[0]

        self._observations = np.array(observations)
        self._fittingObservations = observations[np.ix_(range(nb_observations),
                                                        list(map(int, ObsFitEnum)))]

        # Je n'ai pas trouvé de conditions initiales pour E0 et I0 qui m'évite le message:
        # Warning: uncertainties could not be estimated

        # print(self._observations[0:SKIP_OBS, ObsEnum.HOSPITALIZED.value])
        # print(np.cumsum( self._observations[0:SKIP_OBS, ObsEnum.HOSPITALIZED.value]))

        E0 = 8
        A0 = 3
        SP0 = 1
        H0 = self._observations[0][ObsEnum.NUM_HOSPITALIZED.value]
        C0 = self._observations[0][ObsEnum.NUM_CRITICAL.value]
        R0 = 0
        F0 = 0
        S0 = self._N - E0 - A0 - SP0 - R0 - H0 - C0

        self._initial_conditions = [S0, E0, A0, SP0, H0, C0,F0, R0]

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

        error_margin = 0.2 # 20% of error margin (arbitrary value)

        # ----------------------------------------------------
        # Tau = the rate at which infected people leave the
        # I state to go to the H state.

        # Compute tau = cumulative_hospitalizations / cumulative_positives
        cumulative_positives = np.cumsum(self._observations[:, ObsEnum.NUM_POSITIVE.value])
        cumulative_hospitalizations = self._observations[:, ObsEnum.CUMULATIVE_HOSPITALIZATIONS.value]
        # There is a difference of 1 between the indexes because to have the possibility
        # to go from one state to another, at least 1 day must pass. We can not go from
        # infected to hospitalized the same day.
        tau_0 = cumulative_hospitalizations[-1] / cumulative_positives[-2]
        tau_bounds = [tau_0 * (1 - error_margin), tau_0, max(1, tau_0 * (1 + error_margin))]

        # "best-case": if people recover all in min_symptomatic_time
        gamma1_max = 1/ min_symptomatic_time
        # "worst-case": if people do not recover and go to the H state
        gamma1_min = 0.02 # chosen arbitrarily

        gamma1_0 = 1/(max_symptomatic_time + min_symptomatic_time)

        gamma1_bounds = [gamma1_min, gamma1_0, gamma1_max]


        gamma2_0 = 0.2 # arbitrary choice
        gamma3_0 = 0.2 # arbitrary choice
        gamma2_bounds = [0.02, gamma2_0, 1]
        gamma3_bounds = [0.02, gamma3_0, 1]

        # ------------------------------------------------------------
        # gamma4 : the rate at which people leave the E state to go to the R state
        # "best-case": if people recover all directly after min_incubation_time
        gamma4_max = 1 / min_incubation_time
        # "worst-case": if even after max_incubation_time, people do not recover because they are
        # asymptomatic for a long time, corresponding exactly to the time a symptomatic who is never hospitalised
        # would take to recover (max_incubation_time + max_symptomatic_time).
        gamma4_min = 1 / (max_incubation_time + max_symptomatic_time)
        # "avg-case":
        gamma4_0 = (gamma4_max + gamma4_min) / 2
        gamma4_bounds = [gamma4_min, gamma4_0, gamma4_max]



        beta_0 = 0.5  # on average each exposed person in contact with a susceptible person
        # will infect him with a probability 1/2
        beta_min = 0.01  # on average each exposed person in contact with a susceptible person
        # will infect him with a probability 1/100
        beta_max = 3  # on average each exposed person in contact with a susceptible person will infect him

        beta_bounds = [beta_min, beta_0, beta_max]

        cumulative_criticals_max = np.sum(self._observations[:, ObsEnum.NUM_CRITICAL.value])

        cumulative_criticals_min = self._observations[-1, ObsEnum.NUM_CRITICAL.value]

        delta_max = cumulative_criticals_max / cumulative_hospitalizations[-2]
        delta_max = max(0, min(delta_max, 1))
        delta_min = cumulative_criticals_min / cumulative_hospitalizations[-2]
        delta_min = min(max(delta_min, 0), 1)

        delta_0 = delta_min * 0.7 + delta_max * 0.3
        delta_bounds = [delta_min, delta_0, delta_max]

        # For the period of incubation
        rho_max = 1
        rho_0 = 1/3
        rho_min = 1/5
        rho_bounds = [rho_min,rho_0,rho_max]

        #For the death...
        theta_min = 0.1
        theta_max = 1
        theta_0 = 0.2
        theta_bounds = [theta_min,theta_0,theta_max]


        sigma_max = 1/4
        sigma_min = 1/20 # = 1 / 100
        # "avg-case":
        sigma_0 = (sigma_max + sigma_min) / 2
        sigma_bounds = [sigma_min, sigma_0, sigma_max]

        bounds = [gamma1_bounds, gamma2_bounds, gamma3_bounds, gamma4_bounds, beta_bounds, tau_bounds, delta_bounds, sigma_bounds,rho_bounds,theta_bounds]
        param_names = ['gamma1', 'gamma2', 'gamma3', 'gamma4', 'beta', 'tau', 'delta', 'sigma','rho','theta']
        params = Parameters()

        for param_str, param_bounds in zip(param_names, bounds):
            params.add(param_str, value=param_bounds[1], min=param_bounds[0], max=param_bounds[2])

        return params


    def fit_parameters(self, error_func) :
        # Fit parameters using lmfit package

        params = self.get_initial_parameters()
        result = minimize(self._plumb_lmfit,
                          params,
                          args=(len(self._observations), error_func),
                          method='leastsq')

        report_fit(result)

        self._fit_params = result.params


    def predict(self, days):
        res = self._predict(self._initial_conditions, days, self._fit_params)

        return res

    def stat_predict(self, days):
        res = self._stat_predict(self._initial_conditions, days, self._fit_params)

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

        values = initial_conditions
        states_over_days = [values + [0,0,0]]
        #S0, E0, I0, H0, C0, R0 = initial_conditions
        cumulI = 0

        #print("_predict : params={}".format(params))

        # days - 1 because day 0 is the initial conditions
        for day in range(days - 1):
            m = self._model(values, gamma1, gamma2, gamma3, gamma4, beta, tau, delta, sigma,rho,theta)
            #print("_predict : values={}".format(values))
            #print("_predict : \t\tdeltas={}".format(m))
            dSdt, dEdt, dAdt, dSPdt , dHdt, dCdt, dFdt, dRdt = m

            S, E, A, SP, H, C,F, R = values
            infected_per_day = sigma * E
            R_out_HC = gamma2 * H + gamma3 * C - theta * F
            cumulI += rho * E
            S = S+dSdt
            E = E+dEdt
            A = A+dAdt
            SP = SP+dSPdt
            H = H+dHdt
            C = C+dCdt
            F = F+dFdt
            R = R+dRdt

            values = [S, E, A, SP, H, C,F, R]
            states_over_days.append(values + [infected_per_day,R_out_HC,cumulI])
        return np.array(states_over_days)

    def _stat_predict(self, initial_conditions, days, params):
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

        values = initial_conditions
        states_over_days = [values + [0,0,0]]
        #S0, E0, I0, H0, C0, R0 = initial_conditions
        cumulI = 0
        cumulH = 1
        N = self._N

        #print("_predict : params={}".format(params))

        # days - 1 because day 0 is the initial conditions
        for day in range(days - 1):
            #m = self._model(values, gamma1, gamma2, gamma3, gamma4, beta, tau, delta, sigma,rho,theta)
            #print("_predict : values={}".format(values))
            #print("_predict : \t\tdeltas={}".format(m))
            #dSdt, dEdt, dAdt, dSPdt , dHdt, dCdt, dFdt, dRdt = m

            S, E, A, SP, H, C,F, R = values

            beta_S = self.binomial_dist(beta, 0, 1, S)
            rho_E = self.binomial_dist(rho, 0, 1, E)
            sigma_A = self.binomial_dist(sigma, 0, 1, A)
            gamma4_A = self.binomial_dist(gamma4, 0, 1, A)
            tau_SP = self.binomial_dist(tau, 0, 1, SP)
            gamma1_SP = self.binomial_dist(tau, 0, 1, SP)
            delta_H = self.binomial_dist(delta, 0, 1, H)
            gamma2_H = self.binomial_dist(gamma2, 0, 1, H)
            theta_C = self.binomial_dist(theta, 0, 1, C)
            gamma3_C = self.binomial_dist(gamma3, 0, 1, C)
            #print("C: {}".format(C))
            #print("Theta C: {}".format(theta_C))
            #print("Gamma3 C: {}".format(gamma3_C))


            dSdt = -beta_S * (A+SP) / N
            dEdt = int(np.ceil(beta_S * (A+SP) / N)) - rho_E
            dAdt = rho_E - sigma_A - gamma4_A
            dSPdt = sigma_A - tau_SP - gamma1_SP
            dHdt = tau_SP - delta_H - gamma2_H
            dCdt = delta_H - theta_C - gamma3_C
            dFdt = theta_C
            dRdt = gamma1_SP + gamma2_H + gamma3_C + gamma4_A

            S = S+dSdt
            E = E+dEdt
            A = A+dAdt
            SP = SP+dSPdt
            H = H+dHdt
            C = C+dCdt
            F = F+dFdt
            R = R+dRdt
            cumulH += tau_SP

            infected_per_day = sigma * E
            R_out_HC = cumulH - H - C - F
            cumulI += rho_E

            values = [S, E, A, SP, H, C,F, R]
            states_over_days.append(values + [infected_per_day,R_out_HC,cumulI])
        return np.array(states_over_days)

    def binomial_dist(self, value, bound_min, bound_max, population):

        if population > 0:
            rnd = np.random.uniform(bound_min,bound_max,int(population))
            return rnd[ rnd < value ].shape[0]
        else:

            moving_population = 0
            #print("population {}".format(population))
            for i in range(int(population)):
                proba = random.uniform(bound_min,bound_max)
                if proba < value:
                    moving_population += 1

            return moving_population


    def _stat_model(self, ys, gamma1, gamma2, gamma3, gamma4, beta, tau, delta, sigma, rho, theta):
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

        return [dSdt, dEdt, dAdt, dSPdt, dHdt, dCdt, dFdt, dRdt]

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

        return [dSdt, dEdt, dAdt, dSPdt, dHdt, dCdt, dFdt, dRdt]


    def _plumb_lmfit(self, params, days, error_func):
        assert error_func == residuals_error, "lmfit requires residuals errors"

        res = self._predict(self._initial_conditions, days, params)

        rselect = np.ix_(range(res.shape[0]),
                         list(map(int, StateFitEnum)))

        return error_func(res[rselect], self._fittingObservations).ravel()


    def fit_parameters_bfgs(self, error_func):
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



    def _plumb_scipy(self, params, error_func):

        days = len(self._observations)

        # _predict function prefers params as a dictionary
        # so we convert.
        params_as_dict = self._params_array_to_dict(params)

        # print("\n_plumb_scipy : params: {}".format(params))
        # print("_plumb_scipy : init_cond: {}".format(self._initial_conditions))
        res = self._predict(self._initial_conditions,
                            days, params_as_dict)
        # print("back from predict")

        # INFECTED_PER_DAY = StateEnum.INFECTED_PER_DAY.value
        # HOSPITALIZED = StateEnum.HOSPITALIZED.value
        # CRITICAL = StateEnum.CRITICAL.value

        rselect = np.ix_(range(res.shape[0]),
                         list(map(int, StateFitEnum)))

        return error_func(res[rselect][SKIP_OBS:,:],
                          self._fittingObservations[SKIP_OBS:,:])

    def fit_parameters_ga(self, error_func):
        # Fit parameters using a genetic algorithm

        params = self.get_initial_parameters()

        bounds = np.array([(p.min, p.max) for p_name, p in params.items()])
        self._error_func = error_func

        gamodel = ga(function=self._plumb_ga,
                     dimension=len(bounds),
                     variable_type='real',
                     variable_boundaries=bounds)
        gamodel.run()

        self._fit_params = self._params_array_to_dict(
            gamodel.output_dict['variable'])

        for p_name, p in params.items():
            print( "{:10s} [{:.2f} - {:.2f}] : {:.2f}".format(p_name,p.min, p.max,self._fit_params[p_name]))

    def _params_array_to_dict(self, params):
        return dict(
            zip(['gamma1', 'gamma2', 'gamma3', 'gamma4', 'beta', 'tau', 'delta', 'sigma','rho','theta'],
                params))

    def _plumb_ga(self, params):

        days = len(self._observations)

        # _predict function prefers params as a dictionary
        # so we convert.
        params_as_dict = self._params_array_to_dict(params)

        res = self._predict(self._initial_conditions,
                            days, params_as_dict)

        rselect = np.ix_(range(res.shape[0]),
                         list(map(int, StateFitEnum)))

        # The genetic algorithm uses an error represented
        # as a single float.
        return self._error_func(res[rselect], self._fittingObservations)


if __name__ == "__main__":
    head, observations, rows = load_data()
    rows = np.array(rows)
    days = len(observations)

    ms = Sarah1(rows, 1000000)
    #ms.fit_parameters(residuals_error)
    #ms.fit_parameters_ga(log_residual_sum_of_squares)
    ms.fit_parameters_bfgs(residual_sum_of_squares)

    print("Making prediction")
    sres = ms.stat_predict(250)

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
    plt.ylim(0, 150)
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
