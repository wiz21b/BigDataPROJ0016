# coding=utf-8
import random
import numpy as np
import math
import argparse
import pandas as pd

from datetime import date

from scipy.optimize import minimize as scipy_minimize
from scipy.optimize import differential_evolution

from utils import Model, ObsEnum, StateEnum, ObsFitEnum, StateFitEnum, load_model_data, load_vaccination_data, residual_sum_of_squares, periods_in_days, plot_periods, residuals_error

import matplotlib.pyplot as plt

from scipy.stats import binom, lognorm, norm


random.seed(1001)
np.random.seed(1001)

class SEIR_HCD(Model):
    """ 'stocha' -> modèle stochastique ou pas
        'immunity' -> Les gens développent une immunité ou pas
        'errorFct' à fournir si pas stochastique
        'nbExpériments' pour le fit
    """
    def __init__ (self, stocha = True, immunity = True, errorFct = None, nbExperiments = 100, constantParamNames = [], constantParams = {}):
        super().__init__(stocha = stocha, errorFct = errorFct, nbExperiments = nbExperiments)
        self._immunity = immunity # ne sert pas à l'instant xar posait un problème
                                  # si set à False car alors on avait un paramètre
                                  # dont les bounds étaient 0
        self._fittingPeriod = None

        self._compartmentNames = ['Susceptibles',
                                  'Exposed',
                                  'Asymptomatic',
                                  'Symptomatic',
                                  'Hospitalized',
                                  'Criticals',
                                  'Death',
                                  'Recover']

        self._paramNames = ['Beta',
                            'Rho',
                            'Sigma',
                            'Tau',
                            'Delta',
                            'Theta1',
                            'Theta2',
                            'Gamma1',
                            'Gamma2',
                            'Gamma3',
                            'Gamma4',
                            'Mu1',
                            'Mu2',
                            'Eta']

        self._constantParamNames = constantParamNames
        self._constantParams = constantParams
        if not(immunity):
            self._paramNames += ['Alpha']

        self._param_recorder = []

    def dump_recorded_params(self, fname):
        import pickle
        with open(fname,"wb") as f:
            pickle.dump(self._param_recorder, f)
        self._param_recorder = []

    def set_vaccination(self, vaccination_data, one_dose_efficacy = 0, two_dose_efficacy = 0):
        self.vaccinated_once = vaccination_data["VACCINATED_ONCE"].to_numpy()
        self.vaccinated_twice = vaccination_data["VACCINATED_TWICE"].to_numpy()
        self.one_dose_efficacy = one_dose_efficacy
        self.two_dose_efficacy = two_dose_efficacy

    def set_IC(self, conditions):
        assert len(conditions) == len(self._compartmentNames), \
            "Number of initial conditions given not matching with the model."

        self.initialConditionsAsArray = conditions
        self._initialConditions = dict(zip(self._compartmentNames, conditions))
        self._currentState = dict(zip(self._compartmentNames, conditions))
        self._population = sum(conditions)
        self._ICInitialized = True

    # J'ai mis ça là mais je ne sais pas encore si je l'utiliserai
    def set_param(self, parameters):
        requiredParameters = len(self._paramNames)

        assert len(parameters) == len(self._paramNames),\
            "Number of parameters given not matching with the model."

        self._params = dict(zip(self._paramNames, parameters))
        self._paramInitialized = True

        return

    # J'ai mis ça là mais je ne sais pas encore si je l'utiliserai
    def set_state(self, compartments):
        assert len(compartments) == len(self._compartmentNames),\
            "Number of initial conditions given not matching with the model."

        self._currentState = dict(zip(self._compartmentNames, compartments))
        return

    """ - Seulement la méthode 'LBFGSB' est implémentée pour l'instant mais
          j'ai laissé la possibliité au cas où.
        - RandomPick = True permet de faire un pre-processing des paramètres
          pour trouver un premier jeu correct
        - Fera un fit sur end - start days sur les données entre le jour 'start'
          et le jour 'end'
        - params permets de définir le valeur initiale des paramètres lors du fit.
    """
    def fit_parameters(self, data = None, optimizer = 'LBFGSB',
                       randomPick = False,
                       picks = 1000,
                       start = 0,
                       end = None,
                       params = None,
                       params_random_noise = 0,
                       is_global_optimisation = False):

        assert self._ICInitialized, 'ERROR: Inital conditions not initialized.'

        if isinstance(data, np.ndarray):
            self._data = data
            self._dataLength = data.shape[0]
        else:
            print("ERROR: Data required")
            return

        if not(end):
            self._fittingPeriod = [start, len(self._data)]
        else:
            self._fittingPeriod = [start, end]

        #print("fitting period: {}".format(self._fittingPeriod))


        # L-BFGS-B accepts bounds
        # np.seterr(all = 'raise')
        nonConstantParamNames = []
        constantParams = {}
        if self._constantParams:
            nonConstantParamNames = [pName for pName in self._paramNames if pName not in self._constantParams]
            constantParams = self._constantParams
        else:
            nonConstantParamNames = [pName for pName in self._paramNames if pName not in self._constantParamNames]
            constantParams, _ = self.get_initial_parameters(paramNames = self._constantParamNames,
                                                            randomPick = randomPick, picks = picks)
        # Find first set of parameters
        initialParams, bounds = self.get_initial_parameters(paramNames = nonConstantParamNames, randomPick = randomPick, picks = picks)
        bounds = [bound for bound in bounds.values()]

        if params:
            x0 = [pValue for pName, pValue in params.items() if pName in nonConstantParamNames]
        else:
            x0 = [p for p in initialParams.values()]

        #print(f"Initial guess for the parameters:\n{x0}")
        #for pName, (pMin, pMax) in zip(nonConstantParamNames, bounds):
            #print("{:10s} [{:.4f} - {:.4f}] : {:.4f}".format(pName, pMin, pMax, initialParams[pName]))

        if optimizer == 'LBFGSB':
            #print(constantParams)
            if is_global_optimisation:
                x0 = differential_evolution(self.plumb,
                                             bounds = bounds,
                                             args = (constantParams, False),
                                             popsize = 30,
                                             mutation = (1, 1.9),
                                             recombination = 0.3,
                                             seed = 1)
                print('Status : %s' % x0['message'])
                print('Total Evaluations: %d' % x0['nfev'])
                solution = x0['x']
                print(f'Solution:\n{solution}')
                x0 = solution

            x0 = np.array(x0)
            x0 = x0 + ((np.random.random() * 2) - 1) * x0 * params_random_noise

            res = scipy_minimize(self.plumb,
                                 x0 = x0,
                                 args = (constantParams, True),
                                 method = 'L-BFGS-B',
                                 bounds = bounds,
                                 options = {'eps': 1e-06}) # forcing the steps of the optimizer to be bigger (default value: eps = 1e-8)
            print(res.x)
            parameters = res.x
            for paramName, i in zip(self._paramNames, range(len(parameters) + len(constantParams))):
                if paramName in constantParams:
                    parameters = np.insert(parameters, i, constantParams[paramName])
            self._optimalParams = dict(zip(self._paramNames, parameters))
            self._fitted = True

            print('Optimal parameters after the fitting:')
            for pName, (pMin, pMax) in zip(nonConstantParamNames, bounds):
                print("{:10s} [{:.4f} - {:.4f}] : {:.4f}".format(pName, pMin, pMax,
                                                                 self._optimalParams[pName]))
            print([self._optimalParams[pName] for pName in self._paramNames])
        else:
            print("Other method to implement")

        return self._optimalParams

    """ Fonction à nettoyer ! """

    def get_initial_parameters(self, paramNames = None, randomPick = False, picks = 1000):
        observations = load_model_data()
        min_incubation_time = 5
        max_incubation_time = 6

        min_presymptomatic_time = 1
        max_presymptomatic_time = 3

        min_symptomatic_time = 5
        max_symptomatic_time = 10

        min_fraction_of_asymptomatic = 0.17
        max_fraction_of_asymptomatic = 0.25

        mortality_rate_in_ICU = 0.279
        mortality_rate_in_simple_hospital_beds = 0.168

        avg_stay_in_ICU_in_case_of_death = 19.3
        avg_stay_in_simple_hospital_beds_in_case_of_death = 6.1

        avg_stay_in_ICU_in_case_of_recovery = 9.9
        avg_stay_in_hospital_simple_beds_in_case_of_recovery = 8

        fraction_of_hospitalized_not_transfering_to_ICU = 0.753
        fraction_of_hospitalized_transfering_to_ICU = 1 - fraction_of_hospitalized_not_transfering_to_ICU

        # ----------------------------------
        # Tau (SP -> H) # -> will probably not be constant over time
        avg_time_for_transfer_from_SP_to_H = 5.7
        tau_0 = 0.01 / avg_time_for_transfer_from_SP_to_H  # blind hypothesis: 1 symptomatic out of 100 goes to the hospital
        tau_min = 0.0001 / avg_time_for_transfer_from_SP_to_H  # blind hypothesis: 1 symptomatic out of 10000 goes to the hospital
        tau_max = 0.1 / avg_time_for_transfer_from_SP_to_H  # blind hypothesis: 1 symptomatic out of 10 goes to the hospital

        # ----------------------------------
        # Gamma 4 (A -> R) # -> probably constant over time
        gamma4_max = max_fraction_of_asymptomatic / min_incubation_time
        gamma4_min = min_fraction_of_asymptomatic / (max_incubation_time + max_symptomatic_time)
        gamma4_0 = (gamma4_max + gamma4_min) / 2

        # ----------------------------------
        # Gamma1 (SP -> R) # -> probably constant over time
        gamma1_max = (1 - tau_min) / min_symptomatic_time
        gamma1_min = (1 - tau_max) / max_symptomatic_time
        gamma1_0 = (gamma1_max + gamma1_min) / 2

        # Discuter du bazard en dessous
        # ----------------------------------
        # Beta (S -> E) # -> will vary a lot over time
        R0_min = 0.1  # should be set < 1 if we want to permit a fall after a peak
        R0_max = 4
        R0_avg = (R0_min + R0_max) / 2
        min_infectious_time = min_symptomatic_time + min_presymptomatic_time
        max_infectious_time = max_symptomatic_time + max_presymptomatic_time
        avg_infectious_time = (min_infectious_time + max_infectious_time) / 2
        beta_0 = R0_avg / avg_infectious_time
        beta_min = R0_min / max_infectious_time
        beta_max = R0_max / min_infectious_time

        # ----------------------------------
        # Delta (H -> C) # -> should vary with the influence of the British variant
        fraction_of_hospitalized_transfering_to_ICU_in_case_of_eventual_recovery = fraction_of_hospitalized_transfering_to_ICU / \
                                                                                   avg_stay_in_ICU_in_case_of_recovery
        fraction_of_hospitalized_transfering_to_ICU_in_case_of_eventual_death = fraction_of_hospitalized_transfering_to_ICU / \
                                                                                avg_stay_in_ICU_in_case_of_death
        fraction_of_hospitalized_not_transfering_to_ICU_in_case_of_eventual_recovery = fraction_of_hospitalized_not_transfering_to_ICU / \
                                                                                       avg_stay_in_hospital_simple_beds_in_case_of_recovery
        fraction_of_hospitalized_not_transfering_to_ICU_in_case_of_eventual_death = fraction_of_hospitalized_not_transfering_to_ICU / \
                                                                                    avg_stay_in_simple_hospital_beds_in_case_of_death

        delta_0 = (1 - mortality_rate_in_ICU) * fraction_of_hospitalized_transfering_to_ICU_in_case_of_eventual_recovery + \
                  mortality_rate_in_ICU * fraction_of_hospitalized_transfering_to_ICU_in_case_of_eventual_death
        delta_max = 3 * fraction_of_hospitalized_transfering_to_ICU_in_case_of_eventual_recovery
        # hypothesis: all people eventually recover after they transfer to ICU, they have thus on average a shorter stay in ICU, and thus a higher delta
        delta_min = fraction_of_hospitalized_transfering_to_ICU_in_case_of_eventual_death
        # hypothesis: all people eventually die after they transfer to ICU, they have thus on average a longer stay in ICU, and thus a lower delta

        # delta_min = 0.01  # blind hypothesis
        # delta_max = 0.06  # blind hypothesis
        # delta_0 = (1 - fraction_of_hospitalized_not_transfering_to_ICU) / \
        # ((avg_stay_in_hospital_simple_beds_in_case_of_recovery + avg_stay_in_ICU_in_case_of_death) / 2)  # semi-blind hyptohesis

        # ----------------------------------
        # Theta1 (H -> F) # -> should vary with the influence of the British variant
        # Hypothesis: stay and mortality in simple hospital beds lower bounds the corresponding numbers in ICU
        theta1_min = 0.7 * mortality_rate_in_simple_hospital_beds / avg_stay_in_ICU_in_case_of_death
        theta1_max = 1.3 * mortality_rate_in_ICU / avg_stay_in_simple_hospital_beds_in_case_of_death
        theta1_0 = mortality_rate_in_simple_hospital_beds * fraction_of_hospitalized_not_transfering_to_ICU_in_case_of_eventual_death

        # ----------------------------------
        # Theta2 (C -> F) # -> should vary with the influence of the British variant
        # Hypothesis: stay and mortality in simple hospital beds lower bounds the corresponding numbers in ICU
        theta2_min = 0.7 * mortality_rate_in_simple_hospital_beds / avg_stay_in_ICU_in_case_of_death  # semi-blind hypothesis
        theta2_max = 1.3 * mortality_rate_in_ICU / avg_stay_in_simple_hospital_beds_in_case_of_death  # semi-blind hypothesis
        theta2_0 = mortality_rate_in_ICU / avg_stay_in_ICU_in_case_of_death

        # ----------------------------------
        # Gamma2 (H -> R) # -> probably constant over time
        gamma2_min = 0.001  # blind hypothesis
        gamma2_0 = (1 - mortality_rate_in_simple_hospital_beds) * fraction_of_hospitalized_not_transfering_to_ICU_in_case_of_eventual_recovery
        # (1 - mortality_rate_in_simple_hospital_beds) / avg_stay_in_hospital_simple_beds_in_case_of_recovery
        gamma2_max = fraction_of_hospitalized_not_transfering_to_ICU_in_case_of_eventual_recovery  # blind hypothesis

        # ----------------------------------
        # Gamma3 (C -> R) # -> probably constant over time
        gamma3_min = 0.001  # blind hypothesis
        gamma3_0 = (1 - mortality_rate_in_ICU) / avg_stay_in_ICU_in_case_of_recovery
        gamma3_max = 1 / avg_stay_in_ICU_in_case_of_recovery  # blind hypothesis: everyone eventually recover in ICU

        # ----------------------------------
        # Rho (E -> A) # -> probably constant over time
        rho_max = 1 / min_incubation_time
        rho_0 = 2 / (min_incubation_time + max_incubation_time)
        rho_min = 1 / max_incubation_time

        # ----------------------------------
        # Sigma (A -> SP) # -> probably constant over time
        sigma_max = (1 - min_fraction_of_asymptomatic) / min_presymptomatic_time
        sigma_min = (1 - max_fraction_of_asymptomatic) / max_presymptomatic_time
        sigma_0 = (sigma_max + sigma_min) / 2

        # ----------------------------------
        # Mu (A -> T) # -> will vary over time with the test capacity and the testing rules
        mu1_max = 0.7  # blind hypothesis
        mu1_min = 0  # 0.4  # blind hypothesis
        mu1_0 = (mu1_min + mu1_max) / 2  # blind hypothesis

        # ----------------------------------
        # Mu (SP -> T) # -> will vary over time with the test capacity and the testing rules
        mu2_max = 0.9  # blind hypothesis
        mu2_min = 0.1 # 0.4  # blind hypothesis
        mu2_0 = (mu1_min + mu1_max) / 2  # blind hypothesis

        # ----------------------------------
        # Eta (T -> TP) # -> will vary a lot over time with the peak of contamination
        positivity_rates = observations.NUM_POSITIVE / observations.NUM_TESTED
        eta_max = np.max(positivity_rates)  # 0.3288 # max 32.8 % of positive tests
        eta_min = np.min(positivity_rates)  # 0.009 # min 0.9 % of positive tests
        eta_0 = np.median(positivity_rates)  # 0.08 # on average, 8% of positive tests

        # ----------------------------------
        # Alpha
        #alpha_min = 0.001
        #alpha_max = 0.999
        #alpha_0 = 0.01
        alpha_min = 0
        alpha_max = 0
        alpha_0 = 0
        #alpha_bounds = [0.001, 0.01, 0.95]

        # ----------------------------------
        gamma1_bounds = (gamma1_min, gamma1_max)
        gamma2_bounds = (gamma2_min, gamma2_max)
        gamma3_bounds = (gamma3_min, gamma3_max)
        gamma4_bounds = (gamma4_min, gamma4_max)
        beta_bounds = (beta_min, beta_max)
        tau_bounds = (tau_min, tau_max)
        delta_bounds = (delta_min, delta_max)
        sigma_bounds = (sigma_min, sigma_max)
        rho_bounds = (rho_min, rho_max)
        theta1_bounds = (theta1_min, theta1_max)
        theta2_bounds = (theta2_min, theta2_max)
        mu1_bounds = (mu1_min, mu1_max)
        mu2_bounds = (mu2_min, mu2_max)
        eta_bounds = (eta_min, eta_max)

        bounds = [beta_bounds, rho_bounds, sigma_bounds, tau_bounds, delta_bounds,
                  theta1_bounds, theta2_bounds, gamma1_bounds, gamma2_bounds, gamma3_bounds,
                  gamma4_bounds, mu1_bounds, mu2_bounds, eta_bounds]

        if not (self._immunity):
            # alpha_bounds = [alpha_min, bestParams['Alpha'], alpha_max]
            alpha_bounds = (alpha_min, alpha_max)
            bounds += [alpha_bounds]


        bestParams = [beta_0, rho_0, sigma_0, tau_0, delta_0, theta1_0, theta2_0, gamma1_0, gamma2_0,
                      gamma3_0, gamma4_0, mu1_0, mu2_0, eta_0]

        if not(self._immunity):
            bestParams += [alpha_0]

        if randomPick:
            best = float("inf")
            for test in range(picks):
                if (test % (picks/10) == 0):
                    print("Pre test of the parameters: {} of {}".format(test, picks))

                gamma1 = random.uniform(gamma1_min, gamma1_max)
                gamma2 = random.uniform(gamma2_min, gamma2_max)
                gamma3 = random.uniform(gamma3_min, gamma3_max)
                gamma4 = random.uniform(gamma4_min, gamma4_max)
                beta = random.uniform(beta_min, beta_max)
                tau = random.uniform(tau_min, tau_max)
                delta = random.uniform(delta_min, delta_max)
                sigma = random.uniform(sigma_min, sigma_max)
                rho = random.uniform(rho_min, rho_max)
                theta1 = random.uniform(theta1_min, theta1_max)
                theta2 = random.uniform(theta2_min, theta2_max)
                mu1 = random.uniform(mu1_min, mu1_max)
                mu2 = random.uniform(mu2_min, mu2_max)
                eta = random.uniform(eta_min, eta_max)

                paramValues = [beta, rho, sigma, tau, delta, theta1, theta2, gamma1, gamma2,
                               gamma3, gamma4, mu1, mu2, eta]
                if not(self._immunity):
                    alpha = random.uniform(alpha_min, alpha_max)
                    paramValues += [alpha]


                # Pas en dict ici car ça poserait un problème dans fit_parameters()
                score = self.plumb(paramValues, isMLE = False)
                if score < best:
                    best = score
                    print("Score preprocessing parameters: {}".format(score))
                    bestParams = paramValues

            print('Best preprocessing parameters: {}'.format(dict(zip(self._paramNames, bestParams))))
        bestParams = dict(zip(self._paramNames, bestParams))
        bounds = dict(zip(self._paramNames, bounds))
        bestParams = dict((k, bestParams[k]) for k in paramNames)
        bounds = dict((k, bounds[k]) for k in paramNames)
        return bestParams, bounds

    """
    constantParams is a dictionary (key = paramName, value = paramValue) of parameters that should not be altered
    (used for fitting while keeping some parameters constant across multiple periods)
    """
    def plumb(self, parameters, constantParams = [], isMLE = True):
        for paramName, i in zip(self._paramNames, range(len(parameters) + len(constantParams))):
            if paramName in constantParams:
                parameters = np.insert(parameters, i, constantParams[paramName])
        if isMLE:
            cost = self._plumb_mle(parameters)
        else:
            cost = self._plumb_deterministic(parameters)

        self._param_recorder.append(list(parameters) + list(constantParams.values()) + [cost])

        return cost

    def _plumb_deterministic(self, parameters):
        days = self._fittingPeriod[1]-self._fittingPeriod[0]
        params = dict(zip(self._paramNames, parameters))

        res = self.predict(end = days, parameters = params)

        fittingSelect = [ObsEnum.DHDT.value,
                         #ObsEnum.NUM_TESTED.value,
                         ObsEnum.NUM_POSITIVE.value]#,
                         #ObsEnum.DFDT.value]
        fittingObservations = self._data[self._fittingPeriod[0]:self._fittingPeriod[1], fittingSelect]
        fittingObservations = np.concatenate((fittingObservations, self._data[self._fittingPeriod[0]:self._fittingPeriod[1],
                                                                              [ObsEnum.NUM_HOSPITALIZED.value,
                                                                               ObsEnum.NUM_CRITICAL.value,
                                                                               ObsEnum.NUM_FATALITIES.value#,
                                                                               #ObsEnum.RSURVIVOR.value
                                                                               ]]), axis=1)
        rselect = [StateEnum.SYMPTOMATIQUE.value,
                   #StateEnum.DSPDT.value,
                   StateEnum.DTESTEDDT.value]#,
                   #StateEnum.CRITICAL.value]
        statesToFit = np.array([params['Tau'], params['Eta']]) * res[:,rselect]#np.array([params['Tau'], params['Mu'], params['Eta']]) * res[:,rselect]#np.array([params['Tau'], params['Mu'], params['Eta'], params['Theta']]) * res[:,rselect]
        statesToFit = np.concatenate((statesToFit, res[:, [StateEnum.HOSPITALIZED.value,
                                                           StateEnum.CRITICAL.value,
                                                           StateEnum.FATALITIES.value]]), axis=1)
        # fittingObservations = self._data[self._fittingPeriod[0]:self._fittingPeriod[1], [#ObsEnum.NUM_TESTED.value,
        #                                                                                  #ObsEnum.NUM_POSITIVE.value,
        #                                                                                  ObsEnum.NUM_HOSPITALIZED.value,
        #                                                                                  ObsEnum.NUM_CRITICAL.value]]#,
        #                                                                                  #ObsEnum.DFDT.value]]
        # statesToFit = res[:, [#StateEnum.DTESTEDDT.value,
        #                       #StateEnum.DTESTEDPOSDT.value,
        #                       StateEnum.HOSPITALIZED.value,
        #                       StateEnum.CRITICAL.value]]#,
        #                       #StateEnum.DFDT.value]]
        return np.sum(np.abs(residuals_error(statesToFit, fittingObservations)))


    def _plumb_mle(self, parameters):
        days = self._fittingPeriod[1]-self._fittingPeriod[0]
        params = dict(zip(self._paramNames, parameters))

        if self._stochastic:
            # Stochastic : on fait plusieurs experimentations
            # et chaque expérimentation a un peu de random dedans.

            # et on prend la moyenne
            experiments = []  # dims : [experiment #][day][value]

            for i in range(self._nbExperiments):
                res = self.predict(end = days, parameters = params)
                experiments.append(res)
            # print("... done running experiments")

            experiments = np.stack(experiments)

        else:
            res = self.predict(end = days, parameters = params)


        lhs = dict()
        for state, obs, param in [(StateEnum.SYMPTOMATIQUE, ObsEnum.DHDT, params['Tau']),
                                  #(StateEnum.DSPDT, ObsEnum.NUM_TESTED, params['Mu']),
                                  (StateEnum.DTESTEDDT, ObsEnum.NUM_POSITIVE, params['Eta'])]:#,
                                  #(StateEnum.CRITICAL, ObsEnum.DFDT, params['Theta2'])]:
            # donc 1) depuis le nombre predit de personne SymPtomatique et le parametre tau, je regarde si l'observations dhdt est probable
            #      2) depuis le nombre predit de personne Critical et le parametre theta, je regarde si l'observations dfdt est probable
            #      3) sur la transition entre Asymptomatique et Symptomatique ( sigma*A -> dSPdt) avec le parmetre de test(mu), je regarde si l'observation num_tested est probable
            log_likelihood = 0
            for day in np.arange(0, days):
                # Take all the values of experiments on a given day day_ndx
                # for a given measurement (state.value)

                observation = max(1, self._data[day + self._fittingPeriod[0]][obs.value])
                prediction = None
                if self._stochastic:
                    values = experiments[:, day, state.value]  # binomial
                    prediction = np.mean(values)
                else:
                    prediction = res[day, state.value]

                try:
                    log_bin = binom.logpmf(np.round(observation), np.round(prediction), param)
                    if prediction == 0:
                        log_bin = 0
                except FloatingPointError as exception:
                    log_bin = -999
                log_likelihood += log_bin
                #if log_likelihood == float("-inf"):
                    #print("Error likelihood")

            lhs[str(obs) + "Transition"] = log_likelihood

        std = 2
        for state, obs, param in [(StateEnum.HOSPITALIZED, ObsEnum.NUM_HOSPITALIZED, std),
                                  (StateEnum.CRITICAL, ObsEnum.NUM_CRITICAL, std),
                                  (StateEnum.FATALITIES, ObsEnum.NUM_FATALITIES, std)]:
            log_likelihood = 0
            for day in np.arange(0, days):
                # Take all the values of experiments on a given day day_ndx
                # for a given measurement (state.value)

                observation = max(1, self._data[day + self._fittingPeriod[0]][obs.value])
                prediction = None
                if self._stochastic:
                    values = experiments[:, day, state.value]  # binomial
                    prediction = np.mean(values)
                else:
                    prediction = res[day, state.value]

                try:
                    log_bin = norm.logpdf(observation, prediction, param)
                    if prediction == 0:
                        log_bin = 0
                except FloatingPointError as exception:
                    log_bin = -999
                log_likelihood += log_bin
                # if log_likelihood == float("-inf"):
                # print("Error likelihood")

            lhs[str(obs) + "State"] = log_likelihood

        return -sum(lhs.values())


    """ - Va simuler 'end' days mais ne retournera que ceux après 'start'
        - Si on ne fournit pas 'parameters' on utilise les paramètres trouvés
          par le fit.
    """
    def predict(self, start = 0, end = None, parameters = None):

        # print("PREDICT PARAMS", start, end, parameters)
        # print("PREDICT IC", self._initialConditions)

        if not(end):
            end = len(self._data)
        params = parameters
        if not parameters:
            if self._fitted:
                params = self._optimalParams
            else:
                raise Exception('ERROR: Finding optimal parameters is required!')
                return
        IC = [self._initialConditions[state] for state in self._compartmentNames]
        S, E, A, SP, H, C, F, R = IC
        data = []

        for d in range(end):
            ys = [S, E, A, SP, H, C, F, R]

            vaccinated_once = self.vaccinated_once[d]
            vaccinated_twice = self.vaccinated_twice[d]
            dSdt, dEdt, dAdt, dSPdt, dHdt, dCdt, dFdt, dRdt, dHIndt, dFIndt, dSPIndt, DTESTEDDT, DTESTEDPOSDT = self.model(ys, params, vaccinated_once, vaccinated_twice)

            S += dSdt
            E += dEdt
            A += dAdt
            SP += dSPdt
            H += dHdt
            C += dCdt
            F += dFdt
            R += dRdt

            # On a peut être plus besoin de tout ça mais je le laisse en attendant car sinon faut aussi tout changer
            # dans utils.
            if ( d >= start ):
                data.append([S, E, A, SP, H, C, F, R, dHIndt, dFIndt, dSPIndt, DTESTEDDT, DTESTEDPOSDT])

        return np.array(data)

    def model(self, state, parameters, vaccinated_once, vaccinated_twice):
        # ATTENTION! Ajouter l'équation pour alpha si on veut l'utiliser
        S, E, A, SP, H, C, F, R = state
        N = self._population
        gamma1 = parameters['Gamma1']
        gamma2 = parameters['Gamma2']
        gamma3 = parameters['Gamma3']
        gamma4 = parameters['Gamma4']
        beta = parameters['Beta']
        tau = parameters['Tau']
        delta = parameters['Delta']
        sigma = parameters['Sigma']
        rho = parameters['Rho']
        theta1 = parameters['Theta1']
        theta2 = parameters['Theta2']
        mu1 = parameters['Mu1']
        mu2 = parameters['Mu2']
        eta = parameters['Eta']
        alpha = 0
        if not(self._immunity):
            alpha = parameters['Alpha']

        if self._stochastic:
            betaS = self.population_leave(beta * (A + SP) / N, S)#(beta, S * (A + SP) / N)
            rhoE = self.population_leave(rho, E)
            sigmaA = self.population_leave(sigma, A)
            gamma4A = self.population_leave(gamma4, A)
            tauSP = self.population_leave(tau, SP)
            gamma1SP = self.population_leave(gamma1, SP)
            deltaH = self.population_leave(delta, H)
            gamma2H = self.population_leave(gamma2, H)
            thetaC = self.population_leave(theta, C)
            gamma3C = self.population_leave(gamma3, C)
            muSP = self.population_leave(mu, sigmaA)
            etaSP = self.population_leave(eta, muSP)
            alphaR = 0
            if not(self._immunity):
                alphaR = self.population_leave(alpha, R)

            dSdt = -betaS + alphaR
            dEdt = betaS - rhoE
            dAdt = rhoE - sigmaA - gamma4A
            dSPdt = sigmaA - tauSP - gamma1SP
            dHdt = tauSP - deltaH - gamma2H
            dCdt = deltaH - thetaC - gamma3C
            dFdt = thetaC
            dRdt = gamma1SP + gamma2H + gamma3C + gamma4A - alphaR

            dHIndt = tauSP
            dFIndt = thetaC
            dSPIndt = sigmaA
            DTESTEDDT = muSP
            DTESTEDPOSDT = etaSP
        else:
            alphaR = 0
            if not(self._immunity):
                alphaR = alpha * R

            beta_not_vaccinated = beta * (1 - vaccinated_once / (N - F))
            beta_vaccinated_only_once = beta * ((vaccinated_once - vaccinated_twice) / (N - F)) * (1 - self.one_dose_efficacy)
            beta_vaccinated_twice = beta * (vaccinated_twice / (N - F)) * (1 - self.two_dose_efficacy)
            beta = beta_not_vaccinated + beta_vaccinated_only_once + beta_vaccinated_twice

            dSdt = -beta * S * (A + SP) / N + alphaR
            dEdt = beta * S * (A + SP) / N - rho * E
            dAdt = rho * E - sigma * A - gamma4 * A
            dSPdt = sigma * A - tau * SP - gamma1 * SP
            dHdt = tau * SP - delta * H - gamma2 * H - theta1 * H
            dCdt = delta * H - theta2 * C - gamma3 * C
            dFdt = theta1 * H + theta2 * C
            dRdt = gamma1 * SP + gamma2 * H + gamma3 * C + gamma4 * A - alphaR

            dHIndt = tau * SP
            dFIndt = theta1 * H + theta2 * C
            dSPIndt = sigma * A
            DTESTEDDT = mu1 * A + mu2 * SP #dSPIndt * mu
            DTESTEDPOSDT = eta * DTESTEDDT

        return [dSdt, dEdt, dAdt, dSPdt, dHdt, dCdt, dFdt, dRdt, dHIndt, dFIndt, dSPIndt, DTESTEDDT, DTESTEDPOSDT]


def study_minimum(model, data, initial_conditions, parameters, fraction=0.3):
    # Set NON_PREDICTED_PERIODS to 3 to have a good example.

    from tqdm import tqdm

    SAMPLES = 50
    MLE = True # False = dterministic

    model.set_IC(initial_conditions)
    # Optimizer set to X to allow fit_paramteers to only initialize
    # stuff, not actually do fitting.
    model.fit_parameters(data = data,
                         params = dict(zip(model._paramNames, parameters)),
                         optimizer = 'X')

    minimum = model.plumb(parameters, constantParams = dict(), isMLE=MLE)

    fig, axarr = plt.subplots(4, 4, figsize=(10,12))
    #fig.suptitle("Plot") # Tight layout screws it, and you need tight layout to put more space between subplots

    for p_ndx in tqdm(range(len(parameters))): # len(parameters)
        params = [p for p in parameters]  # copy

        all_values = []
        all_preds = []
        new_min = minimum, parameters[p_ndx]

        for i in range(SAMPLES):
            params[p_ndx] = parameters[p_ndx] * (1 + fraction*2*(i/SAMPLES - 0.5))

            cost = model.plumb(params, constantParams = dict(), isMLE=MLE)
            if cost < new_min[0]:
                new_min = cost, params[p_ndx]

            all_preds.append(cost)
            all_values.append(params[p_ndx])

        if p_ndx % 4 != 0:
            axarr.flat[p_ndx].get_yaxis().set_visible(False)

        axarr.flat[p_ndx].set_xlim(parameters[p_ndx] * (1 + fraction*2*(0 - 0.5)),
                                   parameters[p_ndx] * (1 + fraction*2*(1 - 0.5)))
        axarr.flat[p_ndx].set_ylim(0, minimum*2)
        axarr.flat[p_ndx].set_ylabel("Cost")
        axarr.flat[p_ndx].title.set_text(model._paramNames[p_ndx])
        #axarr.flat[p_ndx].axhline(y=minimum)
        axarr.flat[p_ndx].plot(all_values, all_preds, c="black")

        has_better_minium = abs((new_min[0] - minimum)/minimum) > 0.01
        if has_better_minium:
            axarr.flat[p_ndx].axvline(x=new_min[1],label=f"{new_min[1]:.3f}",color="red")

        axarr.flat[p_ndx].axvline(x=parameters[p_ndx],c="blue")

        if has_better_minium:
            axarr.flat[p_ndx].legend()


    axarr.flat[14].axis('off')
    axarr.flat[15].axis('off')

    fig.tight_layout()
    plt.savefig("minimum.png")
    plt.show()

def stability(model, initial_conditions, days_to_predict, parameters, stab_type, fraction):

    fig, axarr = plt.subplots(3, 3, figsize=(10,10))

    for v_ndx, t in enumerate(list(StateEnum)[:8]):

        all_preds = []
        for i in range(600):
            if stab_type == 1:
                rparams = parameters
                # Randomize initial condition
                ric = np.array(initial_conditions)
                ric *= 1+fraction*(np.random.rand(ric.shape[0]) - 0.5)/0.5
                ric = ric.tolist()

                S0, E0, A0, SP0, H0, C0, F0, R0 = ric
                S0 = N - E0 - A0 - SP0 - H0 - C0 - R0 - F0
                ric = [S0, E0, A0, SP0, H0, C0, F0, R0]

            elif stab_type == 2:
                ric = initial_conditions
                # Randomize parameters
                rparams = np.array(parameters)
                rparams *= 1+fraction*(np.random.rand(rparams.shape[0]) - 0.5)/0.5
                rparams = rparams.tolist()
            else:
                raise Exception("Unsupported")

            model.set_IC(ric)
            sres_temp = model.predict(end = days_to_predict, parameters = dict(zip(model._paramNames, rparams)))
            all_preds.append(sres_temp[:,t.value])

        #plt.figure()
        #plt.title(str(t))
        axarr.flat[v_ndx].title.set_text(str(t))

        for pred in all_preds:#all_preds.shape[0]):
            axarr.flat[v_ndx].plot(pred, c="black", alpha=0.01)

        axarr.flat[v_ndx].title.set_text(str(t))
        model.set_IC(initial_conditions)
        sres_temp = model.predict(end = days_to_predict, parameters = dict(zip(model._paramNames, parameters)))
        # print(sres_temp[:,t.value])
        axarr.flat[v_ndx].plot(range(days_to_predict), sres_temp[:,t.value], '--', c='black')

        axarr.flat[v_ndx].set_ylim(bottom=0) # Must be set after plot

    axarr.flat[8].axis('off')
    # if stab_type == 1:
    #     fig.suptitle(f"Stability of initial conditions, k={fraction:.1f}")
    # else:
    #     fig.suptitle(f"Stability of parameters, k={fraction:.1f}")

    fig.tight_layout()
    if stab_type == 1:
        plt.savefig("stability_IC.png")
    else:
        plt.savefig("stability_params.png")

def graph_synthesis(parameters, periods_in_days, dates, rows):
    # Print dates in LaTex format

    MOTIVES = [
        "First lockdown",
        "Lockdown leave phase 1,2",
        "Lockdown leave phase 3",
        "Masks and social distancing",
        "5 persons bubbles",
        "Beginning of second wave and light lockdown",
        "Top of second wave, full lockdown",
        "Lockdown continues, shops re-open",
        "Lockdown continues, tighter border controls",
        "Easing lockdown",
        "Lockdown, school tightened, shopd closed",
        "Leaving lockdown, schools, shops reopen"
    ]

    year = None
    for i in range(len(dates)-1):
        if dates[i].year != year:
            year = dates[i].year
            d = dates[i].strftime("{%-d}/{%-m}/%Y")
            d2 = dates[i+1].strftime("{%-d}/{%-m}")
        else:
            d = dates[i].strftime("{%-d}/{%-m}")
            d2 = dates[i+1].strftime("{%-d}/{%-m}")

        print(f"{i+1} & {d} & {d2} & {MOTIVES[i]} \\\\")

    P_NAMES = ["Beta (S -> E)", "Rho (E -> A)", "Sigma (A -> SP)", "Tau (SP -> H)", 'Delta (H -> C)',
               'Theta1 (H -> F)',
               'Theta2 (C -> F)', 'Gamma1', 'Gamma2', 'Gamma3',
               'Gamma4', 'Mu1', 'Mu2', 'Eta']

    print(parameters.shape)

    RED = '#FFA0A0'
    periods_colors = [RED, # 13/3/2020
                      None, # 4/5/2020 leaving lockdown
                      None, # 8/6 leaving lockdown
                      None, # 25/7 Masks and social distancing
                      None, # 24/9 new coronavirus restrictions were announced by the government. BIZARRE !!!
                      None, # 6/10 : Beginning of second wave and light lockdown
                      RED, # 2/11 : Top of second wave, full lockdown. :
                      RED, # 1/12 ???
                      None, # 27/1/2021 of January 2021: ???
                      None, # 1/3 : soft lockdown
                      RED, # 27/3 : lockdown
                      None # 11/5/2021 : Leaving lockdown
                      ]

    # for i in range(len(periods_in_days)):
    #     print(i, dates[i+1], periods_in_days[i])

    period_starts = [(p[0] + p[1])//2 for p in periods_in_days]

    fig, ax1 = plt.subplots(figsize=(10,7))

    for ndx_p, p in enumerate(periods_in_days):
        if periods_colors[ndx_p]:
            ax1.axvspan(p[0], p[1]-1, color=periods_colors[ndx_p], zorder=0) # , alpha=0.5)

    """
    https://fr.wikipedia.org/wiki/Pand%C3%A9mie_de_Covid-19_en_Belgique

    14 mars 2020 : Fermeture des écoles, discothèques, cafés et restaurants et l'annulation de tous les rassemblements publics
    18 mars 2020 : Confinement généralisé
    4 mai 2020 : Déconfinement progressif
    """
    # s = (date(2020,3,14) - dates[0]).days
    # e = (date(2020,5,4) - dates[0]).days
    # ax1.axvspan(s, e, color='red', alpha=0.5)

    """
    19 octobre 2020 : un couvre-feu est mis en place de minuit à 5 h du matin5 ; un couvre-feu de 1h à 6h était déjà appliqué dans les provinces du Brabant Wallon et du Luxembourg depuis respectivement les 13 et 14 octobre146. De plus, les bars et restaurants sont à nouveau fermés ; les contacts rapprochés sont limités à 1 personne maximum ; les rassemblements privés sont limités à quatre personnes pendant deux semaines, toujours les mêmes ; les rassemblements sur la voie publique sont limités à quatre personnes maximum ; le télétravail redevient la règle6.
    2 novembre 2020 : nouveau confinement national
    27 novembre 2020 : le Comité de concertation prolonge le confinement national jusqu'au 31 janvier 2021
    """
    # s = (date(2020,10,19) - dates[0]).days
    # e = (date(2021,1,31) - dates[0]).days
    # ax1.axvspan(s, e, color='red', alpha=0.5)

    plot_periods(plt, dates)

    u = ObsEnum.NUM_HOSPITALIZED
    ax2 = ax1.twinx()
    ax2.plot(rows[:, u.value], "--", dashes=(10, 5), c='black') #, label = str(u) + " (real)")

    #plt.xticks([start for start, _ in periods_in_days] + [periods_in_days[-1][-1]], dates)

    for i, pname in enumerate(P_NAMES[:7]):
        if i == 0:
            ax1.plot(period_starts, parameters[:,i], linewidth=2, label=pname, c='black')
        else:
            ax1.scatter(period_starts, parameters[:,i], label=pname)


    ax1.set_ylabel("Parameters values")
    ax2.set_ylabel("# hospitalized")

    ax2.legend()
    ax1.legend()
    fig.tight_layout()
    plt.savefig("img/periods_with_fit.png")
    plt.show()
    exit()



if __name__ == "__main__":

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--stability', help='Stability analysis',
                             action='store_true', required=False, default=False)
    args_parser.add_argument('--minimum', help='Local minimum analysis 2',
                             action='store_true', required=False, default=False)
    args_parser.add_argument('--synth-graph', help='Draw synthesis graph',
                             action='store_true', required=False, default=False)
    args_parser.add_argument('--record-optim', help="Record optimiser's work to given path",
                             required=False, default=None, type=str)
    args_parser.add_argument('--non-prediction', help="Periods to *not* predict (starting from the last)",
                             required=False, default=0, type=int)
    args = args_parser.parse_args()

    # --- Choice of execution ---
    ALL_SCENARIOS = True#False#True # Whether to plot the graphs of all scenarios (requires to have them saved first into csv files) or just one
    EXECUTION = "NO_OPTIMISATION" # "GLOBAL_OPTIMISATION" # "LOCAL_OPTIMISATION" # "NO_OPTIMISATION"
    # "GLOBAL_OPTIMISATION" -> Optimisation by differential evolution via minimum absolute error,
    #                          followed by a local optimisation with the likelihood,
    #                          no use of initial parameters,
    #                          quite long
    # "LOCAL_OPTIMISATION"  -> Local optimisation with the likelihood,
    #                          use of the initial parameters saved from a global optimisation,
    #                          semi long
    # "NO_OPTIMISATION"     -> Display the predicted curves without any optimisation (no fitting),
    #                          use of the initial parameters saved from a global optimisation,
    #                          very quick
    PARAMS_NOISE = 0#0.1 # percentage of random noise to apply to parameters (ideally used before starting a local optimisation) to prevent overfitting
    WITH_VACCINATION = False # Whether we take the effects of vaccination into account
    SAVE_CSV = False#True # Save experiment to a csv file (requires to first create a folder 'csv')
    SAVE_GRAPH = True #True # whether the graphs should be saved
    IMAGE_FOLDER = "img/"  # folder in which graphs are saved
    GRAPH_FORMAT = "png" # format in which the graph should be saved
    FIGURE_SIZE = (8, 6) # size of the figures
    LAST_DATE_FOR_PREDICTION = date(2021, 8, 1) # date when the prediction should stop
                                                # (pay attention to set one_dose_vaccination_forecasts
                                                # and two_dose_vaccination_forecasts accordingly)
    SHOW_PREDICTIONS = True

    EXPERIMENT = 3 # 1, 2 or 3

    if ALL_SCENARIOS:
        EXPERIMENTS = [1, 2, 3]
        GRAPH_PREFIX = "AllPredictions"
    else:
        GRAPH_PREFIX = "Experiment#" + str(EXPERIMENT) # prefix for naming the graph (should concisely describe the execution tested this time)

        if not WITH_VACCINATION:
            GRAPH_PREFIX = "WithoutVaccination_Experiment#" + str(EXPERIMENT)

    # --- Loading data ----
    observations = load_model_data()
    rows = np.array(observations)
    days = len(rows)


    # print(rows[:,StateEnum.HOSPITALIZED.value])
    # z = 1 - (rows[:,StateEnum.HOSPITALIZED.value] -  rows[:,StateEnum.CRITICAL.value]) / (1+rows[:,StateEnum.HOSPITALIZED.value])
    # plt.plot(z)
    # plt.ylim(0,0.040)
    # plt.show()
    # exit()

    # --- Initialisation of periods of similar measures ---
    dates = [observations.DATE.iloc[0].date(), date(2020, 3, 13), date(2020, 5, 4), date(2020, 6, 8),
             date(2020, 7, 25), date(2020, 9, 24), date(2020, 10, 6), date(2020, 11, 2),
             date(2020, 12, 1), date(2021, 1, 27), date(2021, 3, 1), date(2021, 3, 27),
             observations.DATE.iloc[-1].date()]


    # list of tuples (start, end) for each period with significantly distinctive covid-19 measures
    periods_in_days = periods_in_days(dates)
    periods_in_days = periods_in_days[1:] # we start fitting from the 2nd period to start with higher values
    # solution 2, here start from 0. but use the 0 to compute the date so not cool... et marche moins bien que sol 1

    # --- Initialisation of states ---
    N = 11492641 # population belge en 2020
    E0 = 300000
    A0 = round(E0 * 0.181818)
    SP0 = round(A0 * 0.54)
    H0 = rows[periods_in_days[0][0]][ObsEnum.NUM_HOSPITALIZED.value]
    C0 = rows[periods_in_days[0][0]][ObsEnum.NUM_CRITICAL.value]
    R0 = np.sum(rows[:periods_in_days[0][0], ObsEnum.RSURVIVOR.value]) # = 0
    F0 = rows[periods_in_days[0][0]][ObsEnum.NUM_FATALITIES.value]
    S0 = N - E0 - A0 - SP0 - H0 - C0 - R0 - F0

    IC = [S0, E0, A0, SP0, H0, C0, F0, R0]

    # --- Optimal parameters from the global optimization ---

    parameters = [[0.04417909904136072, 0.16666666666666666, 0.25, 0.002607505957964171, 0.04506954288814314, 0.04000391524885744, 0.05945901639344263, 0.09824561403508772, 0.001, 0.09442629590963579, 0.010625, 0.008623746131208974, 0.5360234479440541, 0.017051185637991156],
                  [0.04271173866387728, 0.17873099205973036, 0.48077474701234385, 0.001496307466062664, 0.01795084287535577, 0.01020473022643667, 0.026448479893415137, 0.09981598389810127, 0.06416735909449253, 0.10101010101010101, 0.0329622386461159, 0.4348792278828317, 0.43971725559769304, 0.019722199406887207],
                  [0.18896851158715078, 0.18706606005630683, 0.28005015665764743, 0.0006069235077946143, 0.019381691800506796, 0.021534103103064736, 0.05018119923827652, 0.1919882087569018, 0.030157869507219617, 0.08022611771744065, 0.04912210183222269, 0.011006785812258257, 0.7708926655590773, 0.015971950177058068],
                  [0.14309267694914724, 0.1873442755499069, 0.7048478353537913, 0.0006454553103009291, 0.03823107205152074, 0.014420734454821503, 0.029292199295500258, 0.11829667907225225, 0.0818183341206609, 0.09659714442456281, 0.04798080264921898, 0.029438071975157663, 0.4645005983811566, 0.032704898803983654],
                  [0.23645155645273003, 0.17611768641688785, 0.3477731553741625, 0.0015422358508170686, 0.034200769720435925, 0.0063505387573834105, 0.01287826674248012, 0.19334442910187327, 0.06365726179362154, 0.05206696896409975, 0.03578623628590484, 0.6318146700222789, 0.27396821986554026, 0.07318573239060001],
                  [0.19954839542401634, 0.19675645064497557, 0.3214922227714413, 0.002085859106403714, 0.036734637840919404, 0.015546988039596431, 0.046492157559684465, 0.11447312653816906, 0.007615625535035378, 0.07353946308014339, 0.01744844755718321, 0.05198057193815203, 0.5187808949179322, 0.14530253797727027],
                  [0.12105763349435648, 0.16680652511604088, 0.7373014129558625, 0.0019222179402129946, 0.030947036564513163, 0.032357698870248315, 0.01959951149209873, 0.19326462316731205, 0.037774001791624386, 0.10101010101010101, 0.036301871517823596, 0.03351049557244116, 0.26562217074652733, 0.09436521002366965],
                  [0.18734771954492072, 0.17248396613817968, 0.6117386132553728, 0.0023674667906282113, 0.022585280945423548, 0.028057005218742406, 0.032584061431908826, 0.16777542121682318, 0.0451615840198355, 0.07507796518966266, 0.03572724713458639, 0.060160497625902046, 0.6513565057086848, 0.05590641059638597],
                  [0.20706543319838705, 0.1675073781095676, 0.2895121281202713, 0.0038911643379408176, 0.031058282426042864, 0.013357868185996625, 0.0579718109091564, 0.19582181285061223, 0.061466555070124745, 0.07883248082559581, 0.029048436241333263, 0.04521675298771622, 0.25503968975030133, 0.2398266420122448],
                  [0.3495513813402461, 0.1757375667186865, 0.7419412492861541, 0.0023838733095558067, 0.04147979701047269, 0.007598013172891928, 0.023814050319704832, 0.17290275121104545, 0.04339814744541998, 0.07762910865562846, 0.015252411513868417, 0.018980153002800135, 0.3154897098049746, 0.19559589620971338],
                  [0.22149298682627025, 0.18134490480308654, 0.5682789905320313, 0.002802470215709582, 0.03667414309882367, 0.007785770780693508, 0.025423683520389824, 0.18454088574224045, 0.07432265951138227, 0.05353125003555775, 0.0422617939281555, 0.10336965932369303, 0.5502838931807575, 0.0841251336388422]]
                  #[0.22149298682627025, 0.18134490480308654, 0.5682789905320313, 0.002802470215709582, 0.03667414309882367, 0.007785770780693508, 0.025423683520389824, 0.18454088574224045, 0.07432265951138227, 0.05353125003555775, 0.0422617939281555, 0.10336965932369303, 0.5502838931807575, 0.0841251336388422]]

    parameters = np.array(parameters)


    if args.synth_graph:
        graph_synthesis(parameters, periods_in_days, dates, rows)


    """
    parameters = [[0.05207787586829836, 0.19437643151889256, 0.25882927627499613, 0.002598930041432537, 0.03287024477593973, 0.04453632115704963, 0.043164074052456904, 0.1012552267256648, 0.012232171157036167, 0.07820882346067388, 0.01591714941492931, 0.03179822629884987, 0.531371856470449, 0.015795676481463492],
                  [0.042580445381067644, 0.19855726567799162, 0.4160429190779611, 0.0014213257760221382, 0.015004650714616805, 0.010992118275774239, 0.014521267537479501, 0.09955268056579478, 0.07396785198328328, 0.10101010101010101, 0.010625, 0.3646120976917083, 0.14478147893555207, 0.04061366407215736],
                  [0.15489359314892692, 0.18842828490946634, 0.27351583233288335, 0.0005304404741883941, 0.01804032705080283, 0.022654575127180242, 0.056932759623164894, 0.1769189494535064, 0.0313662292228856, 0.06941526824193904, 0.017247297627200866, 0.0037748375541183563, 0.817361352961781, 0.0129267433464979],
                  [0.1302201137419986, 0.17755726657210547, 0.6452869088901922, 0.0006761670439700634, 0.03457114169047097, 0.016998128521748868, 0.016990837067024485, 0.10535878251828285, 0.08008654051490204, 0.08880191590653053, 0.039236209396366545, 0.028186310961590917, 0.3790734495464846, 0.042489654998359105],
                  [0.22884833859280537, 0.16794027185841648, 0.36436077798529637, 0.0015018622965502658, 0.027635021632397682, 0.007041235713902612, 0.013804766090680381, 0.18220199882811536, 0.06254748499679383, 0.024252572041956404, 0.029851547463444157, 0.29225764826733935, 0.10797517346264168, 0.19028304618871492],
                  [0.252336663645288, 0.19580355524875773, 0.7742179955434318, 0.0018474995266605462, 0.025013656626227596, 0.016631096606959443, 0.03902398034656313, 0.1282046572066014, 0.01746797515679178, 0.017429378077307312, 0.012902413194730896, 0.09323596052848715, 0.2505751132062415, 0.2491891240582348],
                  [0.07858372777031208, 0.19609851995692568, 0.5907115116342876, 0.0017427170957019802, 0.027722191094874723, 0.03012208375367326, 0.02944972556743967, 0.1458556750166045, 0.053334889850102006, 0.08196548271216583, 0.03835589765898502, 0.6934051801205784, 0.10794887224360744, 0.086763243385961],
                  [0.12179702394444493, 0.18121097803108674, 0.5140083656251496, 0.0018889409174304298, 0.018681256821450592, 0.030271122089632136, 0.02340742464661904, 0.10503129400465595, 0.041062325499932155, 0.06970215949440424, 0.023954015461603785, 0.0526663626622299, 0.7602502846573451, 0.04065411298455111],
                  [0.21226316977814352, 0.16666666666666666, 0.3618362784437883, 0.0029726865683662364, 0.01741996833307565, 0.02385226158250797, 0.00841424915269018, 0.17625219269362408, 0.05692536968340551, 0.06874471212926578, 0.025020895630079566, 0.21052404200055003, 0.7329949242046176, 0.06963102651115033],
                  [0.3336891078922028, 0.1712618586688713, 0.6500714506322569, 0.002073418868413918, 0.0347500642968689, 0.012875059419537503, 0.007598149933121957, 0.17877458774699986, 0.04166568929935376, 0.0750847602273164, 0.016382630314136816, 0.43909544509965875, 0.8547447143925153, 0.056584642589059916],
                  [0.2544145137878209, 0.17123596088107365, 0.7535795905274528, 0.002486910877983372, 0.03271485568319763, 0.009054672869941872, 0.022431677032460125, 0.1978573017254884, 0.08550130432592656, 0.043689364771055436, 0.03919024387743541, 0.29973127456274046, 0.8737001639818918, 0.04422319081715245]]
    parameters = np.array(parameters)
    """
    """
    parameters = np.array([
        [5.20778759e-02, 4.25804454e-02, 1.54893593e-01, 1.30220114e-01,
        2.28848339e-01, 2.52336664e-01, 7.85837278e-02, 1.21797024e-01,
        2.12263170e-01, 3.33689108e-01, 2.54414514e-01],
       #[1.94376432e-01, 1.98557266e-01, 1.88428285e-01, 1.77557267e-01,
       # 1.67940272e-01, 1.95803555e-01, 1.96098520e-01, 1.81210978e-01,
       # 1.66666667e-01, 1.71261859e-01, 1.71235961e-01],
       [2.58829276e-01, 4.16042919e-01, 2.73515832e-01, 6.45286909e-01,
        3.64360778e-01, 7.74217996e-01, 5.90711512e-01, 5.14008366e-01,
        3.61836278e-01, 6.50071451e-01, 7.53579591e-01],
       #[2.59893004e-03, 1.42132578e-03, 5.30440474e-04, 6.76167044e-04,
        #1.50186230e-03, 1.84749953e-03, 1.74271710e-03, 1.88894092e-03,
        #2.97268657e-03, 2.07341887e-03, 2.48691088e-03],
       [3.28702448e-02, 1.50046507e-02, 1.80403271e-02, 3.45711417e-02,
        2.76350216e-02, 2.50136566e-02, 2.77221911e-02, 1.86812568e-02,
        1.74199683e-02, 3.47500643e-02, 3.27148557e-02],
       #[4.45363212e-02, 1.09921183e-02, 2.26545751e-02, 1.69981285e-02,
        #7.04123571e-03, 1.66310966e-02, 3.01220838e-02, 3.02711221e-02,
        #2.38522616e-02, 1.28750594e-02, 9.05467287e-03],
       #[4.31640741e-02, 1.45212675e-02, 5.69327596e-02, 1.69908371e-02,
        #1.38047661e-02, 3.90239803e-02, 2.94497256e-02, 2.34074246e-02,
        #8.41424915e-03, 7.59814993e-03, 2.24316770e-02],
       [1.01255227e-01, 9.95526806e-02, 1.76918949e-01, 1.05358783e-01,
        1.82201999e-01, 1.28204657e-01, 1.45855675e-01, 1.05031294e-01,
        1.76252193e-01, 1.78774588e-01, 1.97857302e-01],
       [1.22321712e-02, 7.39678520e-02, 3.13662292e-02, 8.00865405e-02,
        6.25474850e-02, 1.74679752e-02, 5.33348899e-02, 4.10623255e-02,
        5.69253697e-02, 4.16656893e-02, 8.55013043e-02],
       [7.82088235e-02, 1.01010101e-01, 6.94152682e-02, 8.88019159e-02,
        2.42525720e-02, 1.74293781e-02, 8.19654827e-02, 6.97021595e-02,
        6.87447121e-02, 7.50847602e-02, 4.36893648e-02],
       #[1.59171494e-02, 1.06250000e-02, 1.72472976e-02, 3.92362094e-02,
        #2.98515475e-02, 1.29024132e-02, 3.83558977e-02, 2.39540155e-02,
        #2.50208956e-02, 1.63826303e-02, 3.91902439e-02],
       [3.17982263e-02, 3.64612098e-01, 3.77483755e-03, 2.81863110e-02,
        2.92257648e-01, 9.32359605e-02, 6.93405180e-01, 5.26663627e-02,
        2.10524042e-01, 4.39095445e-01, 2.99731275e-01],
       [5.31371856e-01, 1.44781479e-01, 8.17361353e-01, 3.79073450e-01,
        1.07975173e-01, 2.50575113e-01, 1.07948872e-01, 7.60250285e-01,
        7.32994924e-01, 8.54744714e-01, 8.73700164e-01],
       [1.57956765e-02, 4.06136641e-02, 1.29267433e-02, 4.24896550e-02,
        1.90283046e-01, 2.49189124e-01, 8.67632434e-02, 4.06541130e-02,
        6.96310265e-02, 5.65846426e-02, 4.42231908e-02]])
    parameters = parameters.transpose()
    """
    experiment_params = [0.22149298682627025, 0.18134490480308654, 0.5682789905320313, 0.002802470215709582, 0.03667414309882367, 0.007785770780693508, 0.025423683520389824, 0.18454088574224045, 0.07432265951138227, 0.05353125003555775, 0.0422617939281555, 0.10336965932369303, 0.5502838931807575, 0.0841251336388422]

    if EXPERIMENT == 1:
        experiment_params[0] *= 1
    elif EXPERIMENT == 2:
        experiment_params[0] = parameters[5, 0] * 1.7 #0.1794110822420536 * 1.7
    elif EXPERIMENT == 3:
        experiment_params[0] *= 1.65
    else:
        raise ValueError(f"No experiment #{EXPERIMENT} exists...")

    # --- Parameters to keep constant across periods (their value is taken from get_initial_parameters) ---
    # constantParamNames = ("Rho", "Sigma", "Gamma1", "Gamma2", "Gamma3", "Gamma4")  # Must keep the same order of parameters !
    # constantParamNames = ()
    # ms = SEIR_HCD(stocha = False, constantParamNames = constantParamNames)

    # --- Parameters to keep constant across periods (their value must be given) ---
    # constantParams = {"Rho": 0.18264882363547763, "Tau": 0.001794627226065444, "Theta1": 0.02045715228344616, "Theta2": 0.025067173731790935, "Gamma4": 0.024425754548992037}
    constantParams = {}

    # --- Load vaccination information ---
    # Hypothesis: 450 000 vaccines administered per week until the 1st of June,
    # then 550 000 per week until the 1st of July,
    # and then 650 000 per week until the 1st of August.
    one_dose_vaccination_forecasts = {date(2021, 6, 1):4300000, date(2021, 7, 1):5500000, LAST_DATE_FOR_PREDICTION:7500000}
    two_dose_vaccination_forecasts = {date(2021, 6, 1):1700000, date(2021, 7, 1):3200000, LAST_DATE_FOR_PREDICTION:4450000}

    one_dose_efficacy = 0
    two_dose_efficacy = 0
    if WITH_VACCINATION:
        one_dose_efficacy = 0.65
        two_dose_efficacy = 0.8
    vaccination_data = load_vaccination_data(one_dose_vaccination_forecasts, two_dose_vaccination_forecasts)
    vaccination_effect_delay = 14  # hypothesis: 14 days before the vaccine takes effect

    NON_PREDICTED_PERIODS = args.non_prediction


    if not ALL_SCENARIOS:
        # --- Instantiation of the model ---
        ms = SEIR_HCD(stocha = False, constantParams = constantParams)
        ms.set_IC(conditions = IC)

        # --- Run model period after period ---
        sres = np.array([])
        i = 0
        save_params = []
        save_ic = []

        # Running the model means possible doing parameter fitting on a
        # period and always make prediction for that period.
        # The NON_PREDICTED_PERIODS means we won't do that for the last
        # NON_PREDICTED_PERIODS periods. Then after the loop over periods
        # we will make predictions (not fitting) for the remaining periods.
        # These periods are counted for those *bfore* the vaccination
        # periods.

        if NON_PREDICTED_PERIODS:
            rng = periods_in_days[0:-NON_PREDICTED_PERIODS]
        else:
            rng = periods_in_days

        for period_ndx, period in enumerate(rng):
            print(f"\n\nPeriod [{period_ndx}]: [{period[0]}, {period[1]}]. sres is {sres.shape}")
            nonConstantParamNames = [pName for pName in ms._paramNames if pName not in constantParams]
            params = dict(zip(nonConstantParamNames, parameters[i] + ((np.random.random() * 2) - 1) * parameters[i] * PARAMS_NOISE))
            period_duration = period[1] - period[0]
            start = max(period[0] - vaccination_effect_delay, 0)
            end = start + period_duration
            ms.set_vaccination(vaccination_data.iloc[start:end], one_dose_efficacy, two_dose_efficacy)
            sres_temp = None
            save_ic.append(ms.initialConditionsAsArray)
            if EXECUTION == "GLOBAL_OPTIMISATION":
                optimal_params = ms.fit_parameters(data = rows[period[0]:period[1], :], params = params, is_global_optimisation = True, params_random_noise = PARAMS_NOISE) # parameters[i])
                save_params.append(list(optimal_params.values()))
                if args.record_optim:
                    ms.dump_recorded_params(f"{args.record_optim}/opti_params{period_ndx}.pickle")
                sres_temp = ms.predict()
            elif EXECUTION == "LOCAL_OPTIMISATION":
                optimal_params = ms.fit_parameters(data = rows[period[0]:period[1], :], params = params)  # parameters[i])
                save_params.append(list(optimal_params.values()))
                sres_temp = ms.predict()
            elif EXECUTION == "NO_OPTIMISATION":
                sres_temp = ms.predict(end = period[1] - period[0], parameters = params)
                save_params.append(list(params.values()))
            else:
                raise Exception('ERROR: The variable EXECUTION was not set to a valid value '
                                '("GLOBAL_OPTIMISATION", "LOCAL_OPTIMISATION", "NO_OPTIMISATION")')

            if sres_temp.any():
                # sres_temp : compartments values, day by day
                print(f"REINIT IC {sres.shape} + {sres_temp.shape}", sres_temp[-1, 0:8])
                ms.set_IC(conditions = sres_temp[-1, 0:8])
                if not sres.any():
                    sres = sres_temp[:13,:] * 0
                    sres = np.concatenate((sres, sres_temp))
                    #sres = sres_temp
                else:
                    sres = np.concatenate((sres, sres_temp))
                    assert (sres[-1] == sres_temp[-1]).all()
            i += 1

        print(f"\n\nParameters used at each period:\n{save_params}")

        # --- Compute predictions from last parameters computation to last day of data ---

        init_cond = sres[-1, 0:8]
        ms.set_IC(conditions = init_cond)

        # At this point, dates doesn't include the vaccination periods
        n_prediction_days = (LAST_DATE_FOR_PREDICTION - dates[-1-NON_PREDICTED_PERIODS]).days
        start = periods_in_days[-1-NON_PREDICTED_PERIODS][-1] - vaccination_effect_delay
        end = start + n_prediction_days
        ms.set_vaccination(vaccination_data.iloc[start:end],
                           one_dose_efficacy, two_dose_efficacy)

        print(dates)
        print(f"last_date_for_prediction={LAST_DATE_FOR_PREDICTION} - dates[-1]={dates[-1]} => {n_prediction_days} n_prediction_days")
        print(f"Predictin start {start}")

        sres_temp = None

        if args.stability:
            print("S0, E0, A0, SP0, H0, C0, F0, R0")
            print(init_cond)
            # We run this on the latest parameters
            #stability(ms, init_cond, period[1] - period[0], save_params[-1], 1, 0.1)
            stability(ms, save_ic[-1], period[1] - period[0], save_params[-1], 1, 0.1)
            stability(ms, save_ic[-1], period[1] - period[0], save_params[-1], 2, 0.1)
            plt.show()
            exit()
        elif args.minimum:
            # We run this on the last computed parameters
            print("-"*80)
            print(period)
            print(save_params[-1])
            study_minimum(ms, rows[period[0]:period[1], :], sres[period[0], 0:8], save_params[-1])
            plt.show()
            exit()

        #elif EXECUTION == "GLOBAL_OPTIMISATION" or EXECUTION == "LOCAL_OPTIMISATION":
            #sres_temp = ms.predict(end = n_prediction_days)
        else:
            sres_temp = ms.predict(end = n_prediction_days, parameters = dict(zip(ms._paramNames, experiment_params))) #parameters[i - 1])))

        sres = np.concatenate((sres, sres_temp))

    else:
        data = np.array([])
        for experiment in EXPERIMENTS:
            csv_path = "csv/" + "WithoutVaccination" + "_Experiment#" + str(experiment) + '.csv'
            csv_data = pd.read_csv(csv_path).to_numpy()
            if not data.any():
                data = csv_data[np.newaxis, ...]
            else:
                data = np.append(data, csv_data[np.newaxis, ...], axis=0)

        for experiment in EXPERIMENTS:
            csv_path = "csv/" + "Experiment#" + str(experiment) + '.csv'
            csv_data = pd.read_csv(csv_path).to_numpy()
            data = np.append(data, csv_data[np.newaxis, ...], axis=0)

        all_predictions = data[:, :, 2:]
        all_vaccination_data = data[:, :, 0:2]

    dates += list(one_dose_vaccination_forecasts.keys())

    # --- Plot graphs ---
    plt.figure(figsize=FIGURE_SIZE)
    plt.title('Vaccination')
    if ALL_SCENARIOS:
        #for experiment in EXPERIMENTS:
        plt.plot(all_vaccination_data[1, :, 0], label = f"Cumulative Number of Partially Vaccinated People - Hypothesis #{experiment}")
        plt.plot(all_vaccination_data[1, :, 1], label = f"Cumulative Number of Fully Vaccinated People - Hypothesis #{experiment}")
    else:
        plt.plot(vaccination_data.VACCINATED_ONCE, label = f"Cumulative Number of Partially Vaccinated People")
        plt.plot(vaccination_data.VACCINATED_TWICE, label = f"Cumulative Number of Fully Vaccinated People")
    plot_periods(plt, dates)
    plt.legend(loc='upper left')
    if SAVE_GRAPH:
        plt.savefig('{}{}-vaccination.{}'.format(IMAGE_FOLDER, GRAPH_PREFIX, GRAPH_FORMAT, dpi=400))
    plt.show()


    last_fitted_day = periods_in_days[-NON_PREDICTED_PERIODS-1][1] - 2 # FIXME - 2 is a hack
    s = 0
    for i,p in enumerate(periods_in_days):
        b,e=p
        s += e-b
        print(f"({b},{e}) {e-b} days, s={s+periods_in_days[0][0]} - start={dates[i+2]}, {(dates[i+2]-dates[i+1]).days}")
    if not ALL_SCENARIOS:
        print(last_fitted_day, len(sres))


    plt.figure(figsize=FIGURE_SIZE)
    plt.title('Hospitalized Per Day')
    t = StateEnum.DHDT
    u = ObsEnum.DHDT
    plt.plot(rows[:, u.value], "--", label = str(u) + " (real)", c="orange")
    if ALL_SCENARIOS:
        plt.plot(range(last_fitted_day),
                     all_predictions[len(EXPERIMENTS), :last_fitted_day, t.value], label = str(t) + " (model)")
        plt.plot(range(last_fitted_day),
                     all_predictions[0, :last_fitted_day, t.value], label = str(t) + " (model) - Without Vaccination")
        for experiment in EXPERIMENTS:
            experiment2 = experiment + len(EXPERIMENTS) - 1
            plt.plot(range(last_fitted_day, len(all_predictions[experiment2])),
                     all_predictions[experiment2, last_fitted_day:, t.value], label = str(t) + " (prediction)" + f" - Hypothesis #{experiment}")
            plt.plot(range(last_fitted_day, len(all_predictions[experiment])),
                     all_predictions[experiment-1, last_fitted_day:, t.value], label = str(t) + " (model)" + f" - Hypothesis #{experiment}" + " - Without Vaccination")
    else:
        plt.plot(range(last_fitted_day),
                 sres[:last_fitted_day, t.value], label = str(t) + " (model)")
        if SHOW_PREDICTIONS:
            plt.plot(range(last_fitted_day, len(sres)),
                     sres[last_fitted_day:, t.value], label = str(t) + " (prediction)")

    plot_periods(plt, dates)
    plt.legend(loc='upper left')
    if SAVE_GRAPH:
        plt.savefig('{}{}-dhdt.{}'.format(IMAGE_FOLDER, GRAPH_PREFIX, GRAPH_FORMAT))
    plt.show()

    plt.figure(figsize=FIGURE_SIZE)
    plt.title('Hospitalized')
    t = StateEnum.HOSPITALIZED
    u = ObsEnum.NUM_HOSPITALIZED
    plt.plot(rows[:, u.value], "--", label = str(u) + " (real)", c="orange")

    if ALL_SCENARIOS:
        plt.plot(range(last_fitted_day),
                     all_predictions[len(EXPERIMENTS), :last_fitted_day, t.value], label = str(t) + " (model)")
        plt.plot(range(last_fitted_day),
                     all_predictions[0, :last_fitted_day, t.value], label = str(t) + " (model) - Without Vaccination")
        for experiment in EXPERIMENTS:
            experiment2 = experiment + len(EXPERIMENTS) - 1
            plt.plot(range(last_fitted_day, len(all_predictions[experiment2])),
                     all_predictions[experiment2, last_fitted_day:, t.value], label = str(t) + " (prediction)" + f" - Hypothesis #{experiment}")
            plt.plot(range(last_fitted_day, len(all_predictions[experiment])),
                     all_predictions[experiment-1, last_fitted_day:, t.value], label = str(t) + " (model)" + f" - Hypothesis #{experiment}" + " - Without Vaccination")
    else:
        plt.plot(range(last_fitted_day),
                 sres[:last_fitted_day, t.value], label = str(t) + " (model)")
        if SHOW_PREDICTIONS:
            plt.plot(range(last_fitted_day, len(sres)),
                     sres[last_fitted_day:, t.value], label = str(t) + " (prediction)")

    plot_periods(plt, dates)
    plt.legend(loc='upper left')
    if SAVE_GRAPH:
        plt.savefig('{}{}-hospitalised.{}'.format(IMAGE_FOLDER, GRAPH_PREFIX, GRAPH_FORMAT, dpi=400))
    plt.show()

    plt.figure(figsize=FIGURE_SIZE)
    plt.title('Critical')
    t = StateEnum.CRITICAL
    u = ObsEnum.NUM_CRITICAL
    #plt.plot(sres[:, t.value], label = str(t) + " (model)")
    plt.plot(rows[:, u.value], "--", label = str(u) + " (real)", c="orange")
    if ALL_SCENARIOS:
        plt.plot(range(last_fitted_day),
                     all_predictions[len(EXPERIMENTS), :last_fitted_day, t.value], label = str(t) + " (model)")
        plt.plot(range(last_fitted_day),
                     all_predictions[0, :last_fitted_day, t.value], label = str(t) + " (model) - Without Vaccination")
        for experiment in EXPERIMENTS:
            experiment2 = experiment + len(EXPERIMENTS) - 1
            plt.plot(range(last_fitted_day, len(all_predictions[experiment2])),
                     all_predictions[experiment2, last_fitted_day:, t.value], label = str(t) + " (prediction)" + f" - Hypothesis #{experiment}")
            plt.plot(range(last_fitted_day, len(all_predictions[experiment])),
                     all_predictions[experiment-1, last_fitted_day:, t.value], label = str(t) + " (model)" + f" - Hypothesis #{experiment}" + " - Without Vaccination")
    else:
        plt.plot(range(last_fitted_day),
                 sres[:last_fitted_day, t.value], label = str(t) + " (model)")
        if SHOW_PREDICTIONS:
            plt.plot(range(last_fitted_day, len(sres)),
                     sres[last_fitted_day:, t.value], label = str(t) + " (prediction)")

    plot_periods(plt, dates)
    plt.legend(loc='upper left')
    if SAVE_GRAPH:
        plt.savefig('{}{}-critical.{}'.format(IMAGE_FOLDER, GRAPH_PREFIX, GRAPH_FORMAT, dpi=400))
    plt.show()

    plt.figure(figsize=FIGURE_SIZE)
    plt.title('Fatalities')
    t = StateEnum.FATALITIES
    u = ObsEnum.NUM_FATALITIES
    plt.plot(rows[:, u.value], "--", label = str(u) + " (real)", c="orange")
    if ALL_SCENARIOS:
        plt.plot(range(last_fitted_day),
                     all_predictions[len(EXPERIMENTS), :last_fitted_day, t.value], label = str(t) + " (model)")
        plt.plot(range(last_fitted_day),
                     all_predictions[0, :last_fitted_day, t.value], label = str(t) + " (model) - Without Vaccination")
        for experiment in EXPERIMENTS:
            experiment2 = experiment + len(EXPERIMENTS) - 1
            plt.plot(range(last_fitted_day, len(all_predictions[experiment2])),
                     all_predictions[experiment2, last_fitted_day:, t.value], label = str(t) + " (prediction)" + f" - Hypothesis #{experiment}")
            plt.plot(range(last_fitted_day, len(all_predictions[experiment])),
                     all_predictions[experiment-1, last_fitted_day:, t.value], label = str(t) + " (model)" + f" - Hypothesis #{experiment}" + " - Without Vaccination")
    else:
        plt.plot(range(last_fitted_day),
                 sres[:last_fitted_day, t.value], label = str(t) + " (model)")
        if SHOW_PREDICTIONS:
            plt.plot(range(last_fitted_day, len(sres)),
                     sres[last_fitted_day:, t.value], label = str(t) + " (prediction)")

    plot_periods(plt, dates)
    plt.legend(loc='upper left')
    if SAVE_GRAPH:
        plt.savefig('{}{}-fatalities.{}'.format(IMAGE_FOLDER, GRAPH_PREFIX, GRAPH_FORMAT))
    plt.show()

    plt.figure(figsize=FIGURE_SIZE)
    plt.title('Fatalities Per Day')
    t = StateEnum.DFDT
    u = ObsEnum.DFDT
    plt.plot(rows[:, u.value], "--", label = str(u) + " (real)", c="orange")
    if ALL_SCENARIOS:
        plt.plot(range(last_fitted_day),
                     all_predictions[len(EXPERIMENTS), :last_fitted_day, t.value], label = str(t) + " (model)")
        plt.plot(range(last_fitted_day),
                     all_predictions[0, :last_fitted_day, t.value], label = str(t) + " (model) - Without Vaccination")
        for experiment in EXPERIMENTS:
            experiment2 = experiment + len(EXPERIMENTS) - 1
            plt.plot(range(last_fitted_day, len(all_predictions[experiment2])),
                     all_predictions[experiment2, last_fitted_day:, t.value], label = str(t) + " (prediction)" + f" - Hypothesis #{experiment}")
            plt.plot(range(last_fitted_day, len(all_predictions[experiment])),
                     all_predictions[experiment-1, last_fitted_day:, t.value], label = str(t) + " (model)" + f" - Hypothesis #{experiment}" + " - Without Vaccination")
    else:
        plt.plot(range(last_fitted_day),
                 sres[:last_fitted_day, t.value], label = str(t) + " (model)")
        if SHOW_PREDICTIONS:
            plt.plot(range(last_fitted_day, len(sres)),
                     sres[last_fitted_day:, t.value], label = str(t) + " (prediction)")

    plot_periods(plt, dates)
    plt.legend(loc='upper left')
    if SAVE_GRAPH:
        plt.savefig('{}{}-dfdt.{}'.format(IMAGE_FOLDER, GRAPH_PREFIX, GRAPH_FORMAT))
    plt.show()

    plt.figure(figsize=FIGURE_SIZE)
    plt.title('Number of People Tested Per Day')
    t = StateEnum.DTESTEDDT
    u = ObsEnum.NUM_TESTED
    plt.plot(rows[:, u.value], "--", label = str(u) + " (real)", c="orange")
    if ALL_SCENARIOS:
        plt.plot(range(last_fitted_day),
                     all_predictions[len(EXPERIMENTS), :last_fitted_day, t.value], label = str(t) + " (model)")
        plt.plot(range(last_fitted_day),
                     all_predictions[0, :last_fitted_day, t.value], label = str(t) + " (model) - Without Vaccination")
        for experiment in EXPERIMENTS:
            experiment2 = experiment + len(EXPERIMENTS) - 1
            plt.plot(range(last_fitted_day, len(all_predictions[experiment2])),
                     all_predictions[experiment2, last_fitted_day:, t.value], label = str(t) + " (prediction)" + f" - Hypothesis #{experiment}")
            plt.plot(range(last_fitted_day, len(all_predictions[experiment])),
                     all_predictions[experiment-1, last_fitted_day:, t.value], label = str(t) + " (model)" + f" - Hypothesis #{experiment}" + " - Without Vaccination")
    else:
        plt.plot(range(last_fitted_day),
                 sres[:last_fitted_day, t.value], label = str(t) + " (model)")
        if SHOW_PREDICTIONS:
            plt.plot(range(last_fitted_day, len(sres)),
                     sres[last_fitted_day:, t.value], label = str(t) + " (prediction)")

    plot_periods(plt, dates)
    plt.legend(loc='upper left')
    if SAVE_GRAPH:
        plt.savefig('{}{}-dtesteddt.{}'.format(IMAGE_FOLDER, GRAPH_PREFIX, GRAPH_FORMAT))
    plt.show()

    plt.figure(figsize=FIGURE_SIZE)
    plt.title('Number of People Tested Positive Per Day')
    t = StateEnum.DTESTEDPOSDT
    u = ObsEnum.NUM_POSITIVE
    plt.plot(rows[:, u.value], "--", label = str(u) + " (real)", c="orange")
    if ALL_SCENARIOS:
        plt.plot(range(last_fitted_day),
                     all_predictions[len(EXPERIMENTS), :last_fitted_day, t.value], label = str(t) + " (model)")
        plt.plot(range(last_fitted_day),
                     all_predictions[0, :last_fitted_day, t.value], label = str(t) + " (model) - Without Vaccination")
        for experiment in EXPERIMENTS:
            experiment2 = experiment + len(EXPERIMENTS) - 1
            plt.plot(range(last_fitted_day, len(all_predictions[experiment2])),
                     all_predictions[experiment2, last_fitted_day:, t.value], label = str(t) + " (prediction)" + f" - Hypothesis #{experiment}")
            plt.plot(range(last_fitted_day, len(all_predictions[experiment])),
                     all_predictions[experiment-1, last_fitted_day:, t.value], label = str(t) + " (model)" + f" - Hypothesis #{experiment}" + " - Without Vaccination")
    else:
        plt.plot(range(last_fitted_day),
                 sres[:last_fitted_day, t.value], label = str(t) + " (model)")
        if SHOW_PREDICTIONS:
            plt.plot(range(last_fitted_day, len(sres)),
                     sres[last_fitted_day:, t.value], label = str(t) + " (prediction)")

    plot_periods(plt, dates)
    plt.legend(loc='upper left')
    if SAVE_GRAPH:
        plt.savefig('{}{}-dtestedposdt.{}'.format(IMAGE_FOLDER, GRAPH_PREFIX, GRAPH_FORMAT))
    plt.show()

    if not ALL_SCENARIOS and SAVE_CSV:
        df = pd.DataFrame(np.concatenate((vaccination_data[['VACCINATED_ONCE', 'VACCINATED_TWICE']], sres), axis=1))
        df.to_csv(f"csv/{GRAPH_PREFIX}.csv", index=False)
