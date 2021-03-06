import random
import statistics as st
import numpy as np
import math

from lmfit import Parameters
#from geneticalgorithm import geneticalgorithm as ga
from scipy.optimize import minimize as scipy_minimize
from scipy.optimize import basinhopping
from scipy.interpolate import interp1d
from sklearn.neighbors import KernelDensity

from utils import ObsEnum, StateEnum, ObsFitEnum, StateFitEnum, Model, residuals_error, load_data, residual_sum_of_squares, log_residual_sum_of_squares, COLORS_DICT

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
from scipy.stats import binom


# For repeatiblity (but less random numbers)

import random
random.seed(1000)
np.random.seed(1000)

SKIP_OBS=10

def population_leave(param, population):
    # param : the proportion of population that should
    # leave on average

    if population < 0:
        # Fix for some edge cases
        return 0

    # Part of the population that leaves on average
    average = param * population

    # Binomial centered on the population part

    # The rounding is important because binomial
    # is for integer number. By using a round we favor
    # sometimes the high limit sometimes the loaw
    # limit => on average we center. I think np
    # will use "int" instead which always favour
    # the low limit => the distribution is skewed.
    r = np.random.binomial( round(2*average), 0.5)

    return r


class SarahStat(Model):
    def __init__(self, observations, N):

        self._track = []

        self._N = N
        nb_observations = observations.shape[0]
        self._nb_observations = observations.shape[0]

        self._observations = np.array(observations)
        self._fittingObservations = observations[np.ix_(range(nb_observations),
                                                        list(map(int, ObsFitEnum)))]
        self._nExperiments = 200

        self._preprocess = True
        self._evaluation = 0
        self._iterations = 0

        E0 = 30
        A0 = 20
        SP0 = 10
        H0 = observations[0][ObsEnum.NUM_HOSPITALIZED.value]
        C0 = observations[0][ObsEnum.NUM_CRITICAL.value]
        R0 = 0
        F0 = observations[0][ObsEnum.NUM_FATALITIES.value]
        S0 = N - E0 - A0 - SP0 - H0 - C0 - R0 - F0
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
        print(ms._observations[:, ObsEnum.NUM_POSITIVE.value])
        cumulative_positives = np.cumsum(ms._observations[:, ObsEnum.NUM_POSITIVE.value])
        cumulative_hospitalizations = ms._observations[:, ObsEnum.CUMULATIVE_HOSPITALIZATIONS.value]

        # ----------------------------------
        # Tau (SP -> H)
        cumulative_positives = np.cumsum(ms._observations[:, ObsEnum.NUM_POSITIVE.value])
        cumulative_hospitalizations = ms._observations[:, ObsEnum.CUMULATIVE_HOSPITALIZATIONS.value]
        tau_0 = 0.01
        error_margin = 0.2
        tau_min = 0.001
        tau_max = 0.1
        tau_bounds = [tau_min, tau_0, min(1, tau_max)]

        # ------------------------------------------------------------
        # gamma4 : the rate at which people leave the E state to go to the R state
        # "best-case": if people recover all directly after min_incubation_time
        gamma4_max = 1 / min_incubation_time
        # "worst-case": if even after max_incubation_time, people do not recover because they are
        # asymptomatic for a long time, corresponding exactly to the time a symptomatic who is never hospitalised
        # would take to recover (max_incubation_time + max_symptomatic_time).
        gamma4_min = 1 / (max_incubation_time + max_symptomatic_time)
        # "avg-case":
        gamma4_0 = 0.12
        gamma4_bounds = [gamma4_min, gamma4_0, gamma4_max]

        # ----------------------------------
        # Gamma1 (SP -> R)
        # "best-case": if people recover all in min_symptomatic_time
        gamma1_max = 1 / min_symptomatic_time
        # "worst-case": if people do not recover and go to the H state
        gamma1_min = 1/max_symptomatic_time
        gamma1_0 = 0.23 #2 / (max_symptomatic_time + min_symptomatic_time)
        gamma1_bounds = [gamma1_min, gamma1_0, gamma1_max]

        # ----------------------------------
        # Gamma2 (H -> R) & Gamma3 (C -> R)
        gamma2_min = 1/15
        gamma2_0 = 1/13 #0.2  # arbitrary choice
        gamma2_max = 0.5
        gamma2_bounds = [gamma2_min, gamma2_0, gamma2_max]

        gamma3_min = 1/20
        gamma3_0 = 1/19
        gamma3_max = 0.5
        gamma3_bounds = [gamma3_min, gamma3_0, gamma3_max]

        # gamma2_min = 1/10 # on peut rester en moyenne une bonne semaine donc disons 10 jours
        # gamma2_max = 0.4 # quand on va a l'hopital on reste pas 1 jour, au moins 2-3
        # gamma2_0 = 0.3
        # gamma2_bounds = [gamma2_min, gamma2_0, gamma2_max] # quand on va a l'hopital

        # gamma3_min = 1/20 # on peut rester 15 jours-3semaines
        # gamma3_max = 0.3 # encore une fois on va pas en critique pour 1-2 jours mais 3-4 minimum
        # gamma3_0 = 1/10
        # gamma3_bounds = [gamma3_min, gamma3_0, gamma3_max]

        # ------------------------------------------------------------
        # The reproduction number R0 is the number of people
        # each infected person will infect during the time he is infectious
        # R0 = beta * infectious_time
        # We can make the assumption that once an individual is infectious during the time
        # corresponding to max_symptomatic_time even if he he is asymptomatic.
        # Once being in hospital, he will not infect anyone else anymore.
        # Under this assumption, infectious_time is
        # included in [min_symptomatic_time, max_symptomatic_time]
        # R0 = 0.1 # on average, for 10 infected person, only one susceptible will become infected
        R0_min = 1  # or else the virus is not growing exponentially
        R0_max = 2.8 * 1.5  # the most virulent influenza pandemic
        # and we were told that covid-20 looked similar to influenza
        # We multiply by 2 (arbitrary choice) because covid-20 might
        # become more virulent than the most virulent influenza pandemic
        # (which is the case for covid-19 with a R0 that got to 3-4 at peak period)
        R0_avg = (R0_min + R0_max) / 2
        infectious_time = (min_symptomatic_time + max_symptomatic_time) / 2
        beta_0 = R0_avg / infectious_time  # 0.39
        beta_min = R0_min / max_symptomatic_time
        beta_max = R0_max / min_symptomatic_time

        beta_0 = 0.25
        beta_min = 0.2
        beta_max = 0.55
        beta_bounds = [0.30, 0.34, 0.55]



        # ------------------------------------------------------------
        delta_min = 1 / 100 # 1/10
        delta_max = 57/1297
        delta_0 = 0.025
        delta_bounds = [delta_min, delta_0, delta_max]

        # ------------------------------------------------------------
        # E-> A
        # For the period of incubation
        rho_max = 1
        rho_0 = 0.89 #2 / (min_incubation_time + max_incubation_time)
        rho_min = 1 / max_incubation_time
        rho_bounds = [rho_min, rho_0, rho_max]

        # ------------------------------------------------------------
        # C-> F
        # For the death...
        theta_min = 0.01
        theta_max = 0.2
        theta_0 = 0.04
        theta_bounds = [theta_min, theta_0, theta_max]

        # ------------------------------------------------------------
        sigma_max = 0.7
        sigma_min = 0.5 # = 1 / 100
        sigma_0 = 0.6 #(sigma_max + sigma_min) / 2
        sigma_bounds = [sigma_min, sigma_0, sigma_max]

        mu_max = 0.90
        mu_min = 0.5
        mu_0 = 0.67
        mu_bounds = [mu_min,mu_0,mu_max]
        # nombre teste, teste entre 30 et 70% des gens sembent pas fou

        eta_max = 0.85
        eta_min = 0.7
        eta_0 = 0.8
        eta_bounds = [eta_min,eta_0,eta_max]

        if False:
            mini = 99999999
            for pre_test in range(1000):
                print("3 - Pre test of the parameters: {} of 1000".format(pre_test + 1))

                gamma1 = random.uniform(gamma1_min, gamma1_max)
                gamma2 = random.uniform(gamma2_min, gamma2_max)
                gamma3 = random.uniform(gamma3_min, gamma3_max)
                gamma4 = random.uniform(gamma4_min, gamma4_max)
                beta = random.uniform(beta_min, beta_max)
                tau = random.uniform(tau_min, tau_max)
                delta = random.uniform(delta_min, delta_max)
                sigma = random.uniform(sigma_min, sigma_max)
                rho = random.uniform(rho_min, rho_max)
                theta = random.uniform(theta_min, theta_max)
                mu = random.uniform(mu_min, mu_max)
                eta = random.uniform(eta_min, eta_max)

                param_values = [gamma1, gamma2, gamma3, gamma4, beta, tau, delta, sigma, rho, theta, mu, eta]
                #params = self._params_array_to_dict(param_values)
                #print(params)

                neg_likelihood = self._plumb_scipy_stocha(param_values)

                if neg_likelihood < mini:
                    mini = neg_likelihood
                    print("Min preprocess: {}".format(mini))
                    best_preprocess = param_values
                    print("Corresponding params:")
                    print(best_preprocess)

            best_preprocess = self._params_array_to_dict(best_preprocess)
            print(best_preprocess)

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
            mu_bounds = [mu_min, best_preprocess['mu'], mu_max]
            eta_bounds = [eta_min, best_preprocess['eta'], eta_max]

        bounds = [gamma1_bounds, gamma2_bounds, gamma3_bounds, gamma4_bounds, beta_bounds, tau_bounds, delta_bounds, sigma_bounds,rho_bounds,theta_bounds, mu_bounds, eta_bounds]
        param_names = ['gamma1', 'gamma2', 'gamma3', 'gamma4', 'beta', 'tau', 'delta', 'sigma','rho','theta', 'mu', 'eta']
        params = Parameters()

        for param_str, param_bounds in zip(param_names, bounds):
            params.add(param_str, value=param_bounds[1], min=param_bounds[0], max=param_bounds[2])

        return params


    def compute_log_likelihoods(self, all_exp, obs_rows):
        lhs = dict()

        days = obs_rows.shape[0]

        for state, obs, param in [(StateEnum.SYMPTOMATIQUE, ObsEnum.DHDT,self._fit_params['tau'])]:

            log_likelihood = 0
            for day_ndx in np.arange(5,days):
                # Take all the values of experiments on a given day day_ndx
                # for a given measurement (state.value)
                d = all_exp[:, day_ndx, state.value] # binomial
                observation = obs_rows[day_ndx, obs.value] # observation
                print(str(state) + " d = "+ str(np.ceil(np.mean(d))) + "-----------" + " obs = " + str(observation) + "\n")
                #valeur la probabilite d'obtenir observation sachant que d suit cette distribution, avec parametre de tau
                try :
                    x = binom.pmf(observation,max(observation,np.ceil(np.mean(d))),param)
                except FloatingPointError as exception:
                    x = 0.001
                log_bin = np.log(x)
                print(" log_bin = " + str(log_bin) + "---------------------------------------------------")
                log_likelihood += log_bin

            lhs[obs] = log_likelihood

            #print(f"likelihood {state} over {days} days: log_likelihood:{log_likelihood}")

        return lhs


    def _plumb_scipy(self, params, days, error_func=None):
        lhs = dict()

        days = len(self._observations)

        # print("_plumb_scipy " + " ".join([f"{p:.4}" for p in params]))

        # Sarah's function prefers params as a dictionary
        # so we convert.
        params_as_dict = self._params_array_to_dict(params)

        res = self._predict(self._initial_conditions, days,
                            params_as_dict)

        # data.append([S, E, A, SP, H, C, F, R,
        #             infected_per_day, R_survivor, cumulI,
        #              cumulH, R_out_HC])

        for state, obs, param in [(StateEnum.SYMPTOMATIQUE, ObsEnum.DHDT,params_as_dict['tau']),(StateEnum.CRITICAL, ObsEnum.DFDT,params_as_dict['theta']),
                                (StateEnum.DSPDT, ObsEnum.NUM_TESTED,params_as_dict['mu']),(StateEnum.DTESTEDDT, ObsEnum.NUM_POSITIVE,params_as_dict['eta'])]:
        # donc 1) depuis le nombre predit de personne SymPtomatique et le parametre tau, je regarde si l'observations dhdt est probable
        #      2) depuis le nombre predit de personne Critical et le parametre theta, je regarde si l'observations dfdt est probable
        #      3) sur la transition entre Asymptomatique et Symptomatique ( sigma*A -> dSPdt) avec le parmetre de test(mu), je regarde si l'observation num_tested est probable
        #      4) sur la transition entre Asymptomatique et Symptomatique ( sigma*A -> dSPdt), je regarde la proportion de test realisees ( mu*sigma*A) avec le parmetre de sensibilite (eta), je regarde si l'observation num_positive est probable
            log_likelihood = 0
            for day_ndx in range(days):
                # Take all the values of experiments on a given day day_ndx
                # for a given measurement (state.value)

                observation = max(1,self._observations[day_ndx][obs.value])
                prediction = res[day_ndx][state.value]
                #print(str(state) + " d = "+ str(np.ceil(prediction)) + "-----------" + " obs = " + str(observation) + "\n")
                try :
                    x = binom.pmf(observation,np.ceil(np.mean(prediction)),param) + 0.0000000001
                except FloatingPointError as exception:
                    x = 0.0000000001
                log_bin = np.log(x)
                #print(" log_bin = " + str(log_bin) + "---------------------------------------------------")
                log_likelihood += log_bin

            lhs[obs] = log_likelihood

            #print(f"likelihood {state} over {days} days: log_likelihood:{log_likelihood}")

        #print("likelihood: {}".format(-sum(lhs.values())))
        return -sum(lhs.values())


        # return least_squares




    def _plumb_scipy_hopin(self, params):
        lhs = dict()

        days = len(self._observations)

        #print("_plumb_scipy " + " ".join([f"{p:.4}" for p in params]))

        # Sarah's function prefers params as a dictionary
        # so we convert.
        params_as_dict = self._params_array_to_dict(params)

        res = self._predict(self._initial_conditions, days,
                            params_as_dict)

        ndx = np.ix_(
            range(days),
            list(map(int, ObsFitEnum)))


        res = residual_sum_of_squares(
            res[ndx], self._fittingObservations)
        #print("hopin predit : " + str(res))

        p = np.append(params, res)
        self._track.append(p)

        return res




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

        class MyBounds(object):
            def __init__(self, params):
                #bounds = np.array([(p.min, p.max) for p_name, p in params.items()])

                self.xmin = np.array([p.min for p_name, p in params.items()])
                self.xmax = np.array([p.max for p_name, p in params.items()])
                # print("MyBounds")
                # print(self.xmin)
                # print(self.xmax)

            def __call__(self, **kwargs):
                x = kwargs["x_new"]
                tmax = bool(np.all(x <= self.xmax))
                tmin = bool(np.all(x >= self.xmin))
                return tmax and tmin

        def callbackF(Xi):
            print(Xi)
            return False

        hopin_bounds = MyBounds(params)
        minimizer_kwargs = { "method": "L-BFGS-B",
                             "bounds": bounds }
        res = basinhopping(self._plumb_scipy_hopin, x0, minimizer_kwargs=minimizer_kwargs, stepsize=0.1, accept_test=hopin_bounds)

        # res = scipy_minimize(self._plumb_scipy,
        #                      x0=x0,
        #                      method='L-BFGS-B',
        #                      bounds=bounds,
        #                      args=(error_func,),
        #                      callback=callbackF)

        print(res)

        self._fit_params = self._params_array_to_dict(res.x)

        for p_name, p in params.items():
            print( "{:10s} [{:.2f} - {:.2f}] : {:.2f}".format(p_name,p.min, p.max,self._fit_params[p_name]))

    def predict(self, days):
        res = self._predict(self._initial_conditions, days, self._fit_params)
        return res

    def predict_stochastic(self, days):
        res = self._predict(self._initial_conditions, days, self._fit_params,stochastic = True)
        return res

    def _predict_stochastic(self, initial_conditions, days, params):
        return self._predict(initial_conditions, days, params, stochastic = True)

    def _predict(self, initial_conditions, days, params, stochastic = False):
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

        S, E, A, SP, H, C, F, R, infected_per_day, R_out_HC, cumulI = initial_conditions
        cumulH = 0

        data = []

        for d in range(days):
            ys = [S, E, A, SP, H, C, F, R]

            if stochastic:
                dSdt, dEdt, dAdt, dSPdt, dHdt, dCdt, dFdt, dRdt, dHIndt,dFIndt,dSPIndt,DTESTEDDT,DTESTEDPOSDT= self._model_stochastic(ys, gamma1, gamma2, gamma3, gamma4, beta, tau, delta, sigma, rho, theta,mu,eta)
            else:
                dSdt, dEdt, dAdt, dSPdt, dHdt, dCdt, dFdt, dRdt, dHIndt,dFIndt,dSPIndt,DTESTEDDT,DTESTEDPOSDT = self._model(ys, gamma1, gamma2, gamma3, gamma4, beta, tau, delta, sigma, rho, theta,mu,eta)

            S += dSdt
            E += dEdt
            A += dAdt
            SP += dSPdt
            H += dHdt
            C += dCdt
            F += dFdt
            R += dRdt

            data.append([S, E, A, SP, H, C, F, R,dHIndt,dFIndt,dSPIndt,DTESTEDDT,DTESTEDPOSDT])

        return np.array(data)


    def _model(self, ys, gamma1, gamma2, gamma3, gamma4, beta, tau, delta, sigma, rho, theta,mu,eta):
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

        dHIndt = tau*SP
        dFIndt = theta *C
        dSPIndt = sigma * A
        DTESTEDDT = dSPIndt*mu
        DTESTEDPOSDT = DTESTEDDT * eta # eta = sensibilite


        return [dSdt, dEdt, dAdt, dSPdt, dHdt, dCdt, dFdt, dRdt, dHIndt,dFIndt,dSPIndt,DTESTEDDT,DTESTEDPOSDT]



    def _model_stochastic(self, ys, gamma1, gamma2, gamma3, gamma4, beta, tau, delta, sigma, rho, theta,mu,eta):
        S, E, A, SP, H, C, F, R = ys

        N = self._N

        # betaS = beta * S * (A+SP) / N
        # rhoE = rho * E
        # sigmaA = sigma * A
        # gamma4A = gamma4 * A
        # tauSP = tau * SP
        # gamma1SP = gamma1 * SP
        # deltaH = delta * H
        # gamma2H = gamma2 * H
        # thetaC = theta * C
        # gamma3C = gamma3 * C

        betaS = population_leave(beta, S * (A+SP) / N)
        rhoE = population_leave(rho, E)
        sigmaA = population_leave(sigma, A)
        gamma4A = population_leave(gamma4, A)
        tauSP = population_leave(tau, SP)
        gamma1SP = population_leave(gamma1, SP)
        deltaH = population_leave(delta, H)
        gamma2H = population_leave(gamma2, H)
        thetaC = population_leave(theta, C)
        gamma3C = population_leave(gamma3, C)
        muSP = population_leave(mu,sigmaA)# pas sur qu'il faut pas la moyenne
        etaSP = population_leave(eta,muSP)


        dSdt = -betaS
        dEdt =  betaS - rhoE
        #dAdt = rho * E - sigma*E - gamma4 * A
        dAdt = rhoE - sigmaA - gamma4A
        #dSPdt = sigma * E - tau * SP - gamma1 * SP
        dSPdt = sigmaA - tauSP - gamma1SP
        dHdt = tauSP - deltaH - gamma2H
        dCdt = deltaH - thetaC - gamma3C
        dFdt = thetaC
        dRdt = gamma1SP + gamma2H + gamma3C + gamma4A

        dHIndt = tauSP
        dFIndt = thetaC
        dSPIndt = sigmaA
        DTESTEDDT = muSP
        DTESTEDPOSDT = etaSP

        return [dSdt, dEdt, dAdt, dSPdt, dHdt, dCdt, dFdt, dRdt, dHIndt,dFIndt,dSPIndt,DTESTEDDT,DTESTEDPOSDT]


    def _params_array_to_dict(self, params):
        return dict(
            zip(['gamma1', 'gamma2', 'gamma3', 'gamma4', 'beta', 'tau', 'delta', 'sigma','rho','theta','mu','eta'],
                params))


    def _plumb_scipy_stocha(self, params):

        days = len(self._observations)

        # Sarah's function prefers params as a dictionary
        # so we convert.
        params_as_dict = self._params_array_to_dict(params)

        NB_EXPERIMENTS = 100
        PREDICTED_DAYS = 80

        # print(f"Running {NB_EXPERIMENTS} experiments")

        experiments = [] # dims : [experiment #][day][value]

        for i in range(NB_EXPERIMENTS):
            sres = ms._predict_stochastic(
                self._initial_conditions, PREDICTED_DAYS,
                params_as_dict)
            experiments.append(sres)
        # print("... done running experiments")

        all_exp = np.stack(experiments)
        lhs = dict()

        for state, obs, param in [(StateEnum.SYMPTOMATIQUE, ObsEnum.DHDT,params_as_dict['tau']),
                                (StateEnum.DSPDT, ObsEnum.NUM_TESTED,params_as_dict['mu']),(StateEnum.DTESTEDDT, ObsEnum.NUM_POSITIVE,params_as_dict['eta'])]:
        # donc 1) depuis le nombre predit de personne SymPtomatique et le parametre tau, je regarde si l'observations dhdt est probable
        #      2) depuis le nombre predit de personne Critical et le parametre theta, je regarde si l'observations dfdt est probable
        #      3) sur la transition entre Asymptomatique et Symptomatique ( sigma*A -> dSPdt) avec le parmetre de test(mu), je regarde si l'observation num_tested est probable
        #      4) sur la transition entre Asymptomatique et Symptomatique ( sigma*A -> dSPdt), je regarde la proportion de test realisees ( mu*sigma*A) avec le parmetre de sensibilite (eta), je regarde si l'observation num_positive est probable
            log_likelihood = 0
            for day_ndx in np.arange(10,days-8):
                # Take all the values of experiments on a given day day_ndx
                # for a given measurement (state.value)

                observation = max(1,self._observations[day_ndx][obs.value])
                d = all_exp[:, day_ndx, state.value] # binomial
                prediction = np.mean(d)
                #print(str(state) + " d = "+ str(np.ceil(prediction)) + "-----------" + " obs = " + str(observation) + "\n")
                try :
                    x = binom.pmf(observation,np.ceil(np.mean(prediction)),param)
                    log_bin = np.log(x)
                except FloatingPointError as exception:
                    log_bin = -999
                #print(" log_bin = " + str(x) + "---------------------------------------------------")
                log_likelihood += log_bin

            lhs[obs] = log_likelihood
        return -sum(lhs.values())


    def stocha_fit_parameters(self):
        # L-BFGS-B accepts bounds

        np.seterr(all='raise')

        # Find first set of parameters
        params = self.get_initial_parameters()
        bounds = np.array([(p.min, p.max) for p_name, p in params.items()])

        # Group parameters
        for p_name, p in params.items():
            print( "{:10s} [{:.2f} - {:.2f}]".format(p_name,p.min, p.max))

        x0 = [ p.value for p_name, p in params.items() ]
        print( "initial guess for params: {}".format(x0))


        res = scipy_minimize(self._plumb_scipy_stocha,
                             x0=x0,
                             method='L-BFGS-B',
                             bounds=bounds)

        print(res)

        self._fit_params = self._params_array_to_dict(res.x)

        for p_name, p in params.items():
            print( "{:10s} [{:.2f} - {:.2f}] : {:.2f}".format(p_name,p.min, p.max,self._fit_params[p_name]))




if __name__ == "__main__":
    head, observations, rows = load_data()
    rows = np.array(rows)
    days = len(observations)

    ms = SarahStat(rows, 1000000)


    ms.fit_parameters(residual_sum_of_squares)

    track = np.array(ms._track)
    np.save("tracks.npy", track)

    exit()
    #ms.stocha_fit_parameters()

    sres = ms.predict(80)

    version = 3

    plt.figure()
    plt.title('HOSPITALIZED / PER DAY fit')
    t = StateEnum.DHDT
    plt.plot(sres[:, t.value], label=str(t) +" (model)")
    u = ObsEnum.DHDT
    plt.plot(rows[:, u.value], "--", label=str(u) +" (real)")
    plt.savefig('img/v{}-dhdt.pdf'.format(version))

    plt.figure()
    plt.title('Hospitalized')
    t = StateEnum.HOSPITALIZED
    plt.plot(sres[:, t.value], label=str(t) +" (model)")
    u = ObsEnum.NUM_HOSPITALIZED
    plt.plot(rows[:, u.value], "--", label=str(u) +" (real)")
    plt.savefig('img/v{}-hospitalized.pdf'.format(version))

    plt.figure()
    plt.title('Critical')
    t = StateEnum.CRITICAL
    plt.plot(sres[:, t.value], label=str(t) +" (model)")
    u = ObsEnum.NUM_CRITICAL
    plt.plot(rows[:, u.value], "--", label=str(u) +" (real)")
    plt.savefig('img/v{}-critical.pdf'.format(version))

    plt.figure()
    plt.title('FATALITIES')
    t = StateEnum.FATALITIES
    plt.plot(sres[:, t.value], label=str(t) +" (model)")
    u = ObsEnum.NUM_FATALITIES
    plt.plot(rows[:, u.value], "--", label=str(u) +" (real)")
    plt.savefig('img/v{}-FATALITIES.pdf'.format(version))

    plt.figure()
    plt.title('FATALITIES / PER DAY fit')
    t = StateEnum.DFDT
    plt.plot(sres[:, t.value], label=str(t) +" (model)")
    u = ObsEnum.DFDT
    plt.plot(rows[:, u.value], "--", label=str(u) +" (real)")
    plt.savefig('img/v{}-dftf.pdf'.format(version))

    plt.figure()
    plt.title('NUM_tested / PER DAY fit')
    t = StateEnum.DTESTEDDT
    plt.plot(sres[:, t.value], label=str(t) +" (model)")
    u = ObsEnum.NUM_TESTED
    plt.plot(rows[:, u.value], "--", label=str(u) +" (real)")
    plt.savefig('img/v{}-dtesteddt.pdf'.format(version))

    plt.figure()
    plt.title('NUM_Positive / PER DAY fit')
    t = StateEnum.DTESTEDPOSDT
    plt.plot(sres[:, t.value], label=str(t) +" (model)")
    u = ObsEnum.NUM_POSITIVE
    plt.plot(rows[:, u.value], "--", label=str(u) +" (real)")
    plt.savefig('img/v{}-dtestedposdt.pdf'.format(version))
    exit()
    #
    # plt.figure()
    # plt.title('LM fit')
    #
    #
    # NB_EXPERIMENTS = 1000
    # PREDICTED_DAYS = 80
    #
    # print(f"Running {NB_EXPERIMENTS} experiments")
    # experiments = [] # dims : [experiment #][day][value]
    #
    # for i in range(NB_EXPERIMENTS):
    #     sres = ms.predict_stochastic(PREDICTED_DAYS)
    #     experiments.append(sres)
    # print("... done running experiments")
    #
    # all_exp = np.stack(experiments)
    # log_likelihoods = ms.compute_log_likelihoods(all_exp, rows)
    # log_likelihoods_all_params = sum(log_likelihoods.values())
    #
    #
    # # NB_BINS = NB_EXPERIMENTS//20
    #
    # # for state, obs in [(StateEnum.HOSPITALIZED, ObsEnum.NUM_HOSPITALIZED),
    # #                    (StateEnum.CRITICAL, ObsEnum.NUM_CRITICAL)]:
    #
    # #     likelihood = 1
    # #     log_likelihood = 0
    # #     for day_ndx in range(days):
    # #         d = all_exp[:, day_ndx, state.value]
    # #         histo = np.histogram(d,bins=NB_BINS)
    #
    # #         # From histogram counts to probabilities
    # #         proba = histo[0] * (1/np.sum(histo[0]))
    #
    # #         # In which bin fits the observation ?
    # #         observation = rows[day_ndx, obs.value]
    # #         tbin = np.digitize(observation, histo[1]) - 1
    #
    # #         if tbin < 0:
    # #             tbin = 0
    # #         elif tbin >= NB_BINS:
    # #             tbin = NB_BINS - 1
    #
    #
    # #         prob = proba[tbin]
    #
    # #         if True or prob == 0:
    # #             # print(histo)
    # #             # print(observation)
    #
    # #             # x = np.arange(len(proba))
    # #             # idx = np.nonzero(proba)
    # #             # interp = interp1d(x[idx], proba[idx])
    # #             # print(interp(tbin))
    #
    # #             model = KernelDensity(bandwidth=max(1, (np.amax(d) / 10)), kernel='gaussian')
    # #             model.fit(d.reshape(-1, 1))
    #
    #
    # #             ls = np.linspace(observation-0.5,observation+0.5,2)[:,np.newaxis]
    # #             prob = np.sum( np.exp( model.score_samples(ls))) / 2
    #
    # #             prob2 = model.score_samples([[observation]])
    #
    #
    #
    # #             # # Debug
    # #             # print(f"obs:{observation} prob:{prob} prob2:{prob2}")
    # #             # plt.figure()
    #
    # #             # density, bins = np.histogram(d, bins=NB_BINS, density=True)
    # #             # unity_density = density / density.sum()
    # #             # widths = bins[:-1] - bins[1:]
    # #             # #plt.bar(bins[1:], unity_density, width=widths)
    #
    # #             # plt.hist(d,bins=NB_BINS,density=True)
    #
    # #             # log_dens = model.score_samples(histo[1].reshape( -1, 1))
    # #             # plt.plot(histo[1], np.exp(log_dens), c='red')
    # #             # plt.title(f"Day:{day_ndx}")
    # #             # plt.show()
    #
    # #         likelihood *= prob
    # #         log_likelihood += prob2
    #
    # #     print(f"likelihood {state}: {likelihood}, log_likelihood:{log_likelihood}")
    #
    # # exit()
    #
    # # for sres in experiments:
    # #     for t in [StateEnum.RSURVIVOR]: #, StateEnum.HOSPITALIZED, StateEnum.CRITICAL, StateEnum.FATALITIES]:
    # #         plt.plot(sres[:, t.value], c=COLORS_DICT[t], alpha=0.05, linewidth=0.4)
    #
    #
    # sres = ms.predict(250)
    #
    # plt.figure()
    # t = StateEnum.DHDT
    # plt.plot(sres[:, t.value], c=COLORS_DICT[t], label=f"{t} (model)")
    # u = ObsEnum.DHDT
    # plt.plot(rows[:, u.value], "--", c=COLORS_DICT[u], label=f"{u} (real)")
    # plt.savefig('dhdt.pdf')
    #
    # plt.figure()
    # t = StateEnum.DFDT
    # plt.plot(sres[:, t.value], c=COLORS_DICT[t], label=f"{t} (model)")
    # u = ObsEnum.DFDT
    # plt.plot(rows[:, u.value], "--", c=COLORS_DICT[u], label=f"{u} (real)")
    # plt.savefig('dftf.pdf')
    #
    # exit()
    #
    # # for t in [, StateEnum.DFDT]:
    #
    # # for u in [ObsEnum.NUM_HOSPITALIZED, ObsEnum.NUM_CRITICAL,ObsEnum.NUM_FATALITIES]:
    # #     plt.plot(rows[:, u.value], "--", c=COLORS_DICT[u], label=f"{u} (real)")
    #
    #
    # plt.title('Curve Fitting')
    # plt.xlabel('Days')
    # plt.ylabel('Individuals')
    # prediction_days = 10 # prediction at prediction_days
    # plt.xlim(0, days + prediction_days)
    # plt.ylim(0, 1000)
    # plt.legend()
    # plt.savefig('data_fit.pdf')
    # plt.savefig(f'data_fit_{days}_days.pdf')
    # plt.show()
    #
    # plt.figure()
    # jet = plt.get_cmap('jet')
    # days_max = min(200, PREDICTED_DAYS)
    # cNorm  = colors.Normalize(vmin=20, vmax=days_max)
    # scalarMap = cm.ScalarMappable(norm=cNorm, cmap=jet)
    # for day in range(60,days_max):
    #     d = [experiments[exp_i][day, StateEnum.HOSPITALIZED.value] for exp_i in range(NB_EXPERIMENTS)]
    #     hd = np.histogram(d,bins=50)[0]
    #     plt.plot(hd, c=scalarMap.to_rgba(day), alpha=0.2)
    # plt.title("Normalized histograms of 1000/day hospitalized values (color = day)")
    # plt.xticks([])
    # plt.colorbar(scalarMap)
    #
    # plt.figure()
    # for t in [StateEnum.EXPOSED, StateEnum.ASYMPTOMATIQUE, StateEnum.SYMPTOMATIQUE ,StateEnum.HOSPITALIZED, StateEnum.CRITICAL, StateEnum.FATALITIES]:
    #     plt.plot(sres[:, t.value], label=f"{t} (model)")
    #
    # plt.title('Exposed - Infectious - Hospitalized - Critical')
    # plt.xlabel('Days')
    # plt.ylabel('Individuals')
    # plt.legend()
    # plt.savefig('projection_zoom.pdf')
    # plt.savefig(f'projection_zoom_{days}_days.pdf')
    # plt.show()
    #
    # plt.figure()
    # for t in [StateEnum.SUCEPTIBLE, StateEnum.RECOVERED, StateEnum.CUMULI, StateEnum.FATALITIES]:
    #     plt.plot(sres[:, t.value], label=f"{t} (model)")
    #
    # plt.title('States')
    # plt.xlabel('Days')
    # plt.ylabel('Individuals')
    # plt.legend()
    # plt.savefig('projection_global.pdf')
    # plt.savefig(f'projection_global_{days}_days.pdf')
    # plt.show()
