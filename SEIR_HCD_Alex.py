# coding=utf-8
import random
import numpy as np
import math

from scipy.optimize import minimize as scipy_minimize
from scipy.optimize import differential_evolution

from utils import Model, ObsEnum, StateEnum, ObsFitEnum, StateFitEnum, load_model_data, residual_sum_of_squares, periods_in_days, plot_periods, residuals_error

import matplotlib.pyplot as plt

from scipy.stats import binom

from datetime import date

random.seed(1001)
np.random.seed(1001)

class SEIR_HCD(Model):
    """ 'stocha' -> modèle stochastique ou pas
        'immunity' -> Les gens développent une immunité ou pas
        'errorFct' à fournir si pas stochastique
        'nbExpériments' pour le fit
    """
    def __init__ (self, stocha = True, immunity = True, errorFct = None, nbExperiments = 100, constantParamNames = {}):
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

        if not(immunity):
            self._paramNames += ['Alpha']

    def set_IC(self, conditions):
        assert len(conditions) == len(self._compartmentNames), \
            "Number of initial conditions given not matching with the model."

        self._initialConditions = dict(zip(self._compartmentNames, conditions))
        self._currentState = dict(zip(self._compartmentNames, conditions))
        self._population = sum(conditions)
        self._ICInitialized = True
        return

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
                       params = None):

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
        nonConstantParamNames = [pName for pName in self._paramNames if pName not in self._constantParamNames]
        # Find first set of parameters
        initialParams, bounds = self.get_initial_parameters(paramNames = nonConstantParamNames, randomPick = randomPick, picks = picks)
        constantParams, _ = self.get_initial_parameters(paramNames = self._constantParamNames, randomPick = randomPick, picks = picks)
        bounds = [bound for bound in bounds.values()]

        if not(params == None):
            x0 = params
        else:
            x0 = [p for p in initialParams.values()]

        #print(f"Initial guess for the parameters:\n{x0}")
        #for pName, (pMin, pMax) in zip(nonConstantParamNames, bounds):
            #print("{:10s} [{:.4f} - {:.4f}] : {:.4f}".format(pName, pMin, pMax, initialParams[pName]))

        if optimizer == 'LBFGSB':
            print(constantParams)
            res = differential_evolution(self.plumb,
                                         bounds = bounds,
                                         args = (constantParams, False),
                                         popsize = 30,
                                         mutation = (1, 1.9),
                                         recombination = 0.3)
            print('Status : %s' % res['message'])
            print('Total Evaluations: %d' % res['nfev'])
            solution = res['x']
            print(f'Solution:\n{solution}')

            res = scipy_minimize(self.plumb,
                                 x0 = res.x, # x0,
                                 args = (constantParams, True),
                                 method = 'L-BFGS-B',
                                 bounds = bounds)
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

        return

    """ Fonction à nettoyer ! """

    def get_initial_parameters(self, paramNames = None, randomPick = False, picks = 1000):
        # min_incubation_time = 5
        # max_incubation_time = 6
        #
        # min_presymptomatic_time = 1
        # max_presymptomatic_time = 3
        #
        # min_symptomatic_time = 5
        # max_symptomatic_time = 10
        #
        # mortality_rate_in_ICU = 0.279
        # mortality_rate_in_simple_hospital_beds = 0.168
        #
        # avg_stay_in_ICU_in_case_of_death = 19.3
        # avg_stay_in_simple_hospital_beds_in_case_of_death = 6.1
        #
        # avg_stay_in_ICU_in_case_of_recovery = 9.9
        # avg_stay_in_hospital_simple_beds_in_case_of_recovery = 8


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
        delta_max = 8 * fraction_of_hospitalized_transfering_to_ICU_in_case_of_eventual_recovery
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
        gamma2_min = 0.01  # blind hypothesis
        gamma2_0 = (1 - mortality_rate_in_simple_hospital_beds) * fraction_of_hospitalized_not_transfering_to_ICU_in_case_of_eventual_recovery
        # (1 - mortality_rate_in_simple_hospital_beds) / avg_stay_in_hospital_simple_beds_in_case_of_recovery
        gamma2_max = fraction_of_hospitalized_not_transfering_to_ICU_in_case_of_eventual_recovery  # blind hypothesis

        # ----------------------------------
        # Gamma3 (C -> R) # -> probably constant over time
        gamma3_min = 0.01  # blind hypothesis
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
            return self._plumb_mle(parameters)
        else:
            return self._plumb_deterministic(parameters)


    def _plumb_deterministic(self, parameters):
        days = self._fittingPeriod[1]-self._fittingPeriod[0]
        params = dict(zip(self._paramNames, parameters))

        res = self.predict(end = days, parameters = params)

        fittingSelect = [ObsEnum.DHDT.value,
                         #ObsEnum.NUM_TESTED.value,
                         ObsEnum.NUM_POSITIVE.value]#,
                         #ObsEnum.DFDT.value]
        fittingObservations = self._data[self._fittingPeriod[0]:self._fittingPeriod[1], fittingSelect]
        #fittingObservations = np.concatenate((fittingObservations, self._data[self._fittingPeriod[0]:self._fittingPeriod[1], [ObsEnum.NUM_HOSPITALIZED.value, ObsEnum.NUM_CRITICAL.value]]), axis=1)
        rselect = [StateEnum.SYMPTOMATIQUE.value,
                   #StateEnum.DSPDT.value,
                   StateEnum.DTESTEDDT.value]#,
                   #StateEnum.CRITICAL.value]
        statesToFit = np.array([params['Tau'], params['Eta']]) * res[:,rselect]#np.array([params['Tau'], params['Mu'], params['Eta']]) * res[:,rselect]#np.array([params['Tau'], params['Mu'], params['Eta'], params['Theta']]) * res[:,rselect]
        # statesToFit = np.concatenate((statesToFit, res[:, [StateEnum.HOSPITALIZED.value, StateEnum.CRITICAL.value]]), axis=1)
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


        #if self._stochastic:
        lhs = dict()
        for state, obs, param in [(StateEnum.SYMPTOMATIQUE, ObsEnum.DHDT, params['Tau']),
                                  #(StateEnum.DSPDT, ObsEnum.NUM_TESTED, params['Mu']),
                                  (StateEnum.DTESTEDDT, ObsEnum.NUM_POSITIVE, params['Eta'])]:#, #]:
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
                    log_bin = binom.logpmf(observation, np.round(np.mean(prediction)), param)
                    if prediction == 0:
                        log_bin = 0
                except FloatingPointError as exception:
                    log_bin = -999
                log_likelihood += log_bin
                #if log_likelihood == float("-inf"):
                    #print("Error likelihood")

            lhs[obs] = log_likelihood
        return -sum(lhs.values())


    """ - Va simuler 'end' days mais ne retournera que ceux après 'start'
        - Si on ne fournit pas 'parameters' on utilise les paramètres trouvés
          par le fit.
    """
    def predict(self, start = 0, end = None, parameters = None):
        if not(end):
            end = len(self._data)
        params = parameters
        if not(parameters):
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

            dSdt, dEdt, dAdt, dSPdt, dHdt, dCdt, dFdt, dRdt, dHIndt, dFIndt, dSPIndt, DTESTEDDT, DTESTEDPOSDT = self.model(ys, params)

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

    def model(self, state, parameters):
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


if __name__ == "__main__":
    observations = load_model_data()
    rows = np.array(observations)
    days = len(rows)
    dates = [observations.DATE.iloc[0].date(), date(2020, 3, 13), date(2020, 5, 4), date(2020, 6, 8),
             date(2020, 7, 25), date(2020, 9, 24), date(2020, 10, 6), date(2020, 11, 2),
             date(2020, 12, 1), date(2021, 1, 27), date(2021, 3, 1), date(2021, 3, 27),
             observations.DATE.iloc[-1].date()]
    # list of tuples (start, end) for each period with significantly distinctive covid-19 measures
    periods_in_days = periods_in_days(dates)
    periods_in_days = periods_in_days[1:] # we start fitting from the 2nd period to start with higher values
    # solution 2, here start from 0. but use the 0 to compute the date so not cool... et marche moins bien que sol 1

    # Parameters to keep constant across periods
    #constantParamNames = ("Rho", "Sigma", "Gamma1", "Gamma2", "Gamma3", "Gamma4")  # Must keep the same order of parameters !
    constantParamNames = ()
    ms = SEIR_HCD(stocha = False, constantParamNames = constantParamNames)

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
    print(IC)
    ms.set_IC(conditions = IC)

    sres = np.array([])
    i = 0
    for period in periods_in_days:
        print(f"\n\nPeriod: [{period[0]}, {period[1]}]")
        ms.fit_parameters(data = rows[period[0]:period[1], :], randomPick = False, picks = 10000)#,params = parameters[i])

        sres_temp = ms.predict()
        if sres_temp.any():
            print(sres_temp[-1, 0:8])
            ms.set_IC(conditions = sres_temp[-1, 0:8])
            if not sres.any():
                sres = sres_temp[:13,:] * 0 #solution 1, artificielement mettre des 0 pour les X premier jours, où plus propre, mettre IC 13 fois à voir.
                sres = np.concatenate((sres, sres_temp)) # fait partie de solution 1
                #sres = sres_temp
            else:
                sres = np.concatenate((sres, sres_temp))
        i += 1

    info = "WithA->T_SP->T_WithoutConstantParams_FitT->TP_SP->H_2"

    plt.figure()
    plt.title('HOSPITALIZED / PER DAY fit')
    t = StateEnum.DHDT
    plt.plot(sres[:, t.value], label = str(t) + " (model)")
    u = ObsEnum.DHDT
    plt.plot(rows[:, u.value], "--", label = str(u) + " (real)")
    plot_periods(plt, dates)
    plt.savefig('img/{}-dhdt.pdf'.format(info))
    plt.show()

    plt.figure()
    plt.title('Hospitalized')
    t = StateEnum.HOSPITALIZED
    plt.plot(sres[:, t.value], label = str(t) + " (model)")
    u = ObsEnum.NUM_HOSPITALIZED
    plt.plot(rows[:, u.value], "--", label = str(u) + " (real)")
    plot_periods(plt, dates)
    plt.savefig('img/{}-hospitalized.pdf'.format(info))
    plt.show()

    plt.figure()
    plt.title('Critical')
    t = StateEnum.CRITICAL
    plt.plot(sres[:, t.value], label = str(t) + " (model)")
    u = ObsEnum.NUM_CRITICAL
    plt.plot(rows[:, u.value], "--", label = str(u) + " (real)")
    plot_periods(plt, dates)
    plt.savefig('img/{}-critical.pdf'.format(info))
    plt.show()

    plt.figure()
    plt.title('FATALITIES')
    t = StateEnum.FATALITIES
    plt.plot(sres[:, t.value], label = str(t) + " (model)")
    u = ObsEnum.NUM_FATALITIES
    plt.plot(rows[:, u.value], "--", label = str(u) + " (real)")
    plot_periods(plt, dates)
    plt.savefig('img/{}-FATALITIES.pdf'.format(info))
    plt.show()

    plt.figure()
    plt.title('FATALITIES / PER DAY fit')
    t = StateEnum.DFDT
    plt.plot(sres[:, t.value], label = str(t) + " (model)")
    u = ObsEnum.DFDT
    plt.plot(rows[:, u.value], "--", label = str(u) + " (real)")
    plot_periods(plt, dates)
    plt.savefig('img/{}-dftf.pdf'.format(info))
    plt.show()

    plt.figure()
    plt.title('NUM_tested / PER DAY fit')
    t = StateEnum.DTESTEDDT
    plt.plot(sres[:, t.value], label = str(t) + " (model)")
    u = ObsEnum.NUM_TESTED
    plt.plot(rows[:, u.value], "--", label = str(u) + " (real)")
    plot_periods(plt, dates)
    plt.savefig('img/{}-dtesteddt.pdf'.format(info))
    plt.show()

    plt.figure()
    plt.title('NUM_Positive / PER DAY fit')
    t = StateEnum.DTESTEDPOSDT
    plt.plot(sres[:, t.value], label = str(t) + " (model)")
    u = ObsEnum.NUM_POSITIVE
    plt.plot(rows[:, u.value], "--", label = str(u) + " (real)")
    plot_periods(plt, dates)
    plt.savefig('img/{}-dtestedposdt.pdf'.format(info))
    plt.show()
