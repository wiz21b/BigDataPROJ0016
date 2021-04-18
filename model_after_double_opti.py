# coding=utf-8
import random
import numpy as np
import math

from lmfit import Parameters
from scipy.optimize import minimize as scipy_minimize

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
    def __init__ (self, stocha = True, immunity = True, errorFct = None, nbExperiments = 100):
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
                            'Theta',
                            'Theta2',
                            'Gamma1',
                            'Gamma2',
                            'Gamma3',
                            'Gamma4',
                            'Mu',
                            'Eta']

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

        # Find first set of parameters
        parameters = self.get_initial_parameters(randomPick = randomPick, picks = picks)
        bounds = np.array([(p.min, p.max) for pName, p in parameters.items()])

        # Group parameters
        #print('Parameter bounds:')
        #for pName, p in parameters.items():
            #print("{:10s} [{:.2f} - {:.2f}]".format(pName, p.min, p.max))

        x0 = [p.value for p_name, p in parameters.items()]

        if not(params == None):
            x0 = params

        print("Initial guess for the parameters: {}".format(x0))

        if optimizer == 'LBFGSB':
            res = scipy_minimize(self.plumb,
                                x0 = x0,
                                method = 'L-BFGS-B',
                                bounds = bounds)

            self._optimalParams = dict(zip(self._paramNames, res.x))
            self._fitted = True

            print('Optimal parameters after the fitting:')
            for pName, p in parameters.items():
                print("{:10s} [{:.4f} - {:.4f}] : {:.4f}".format(pName, p.min, p.max,
                                                                 self._optimalParams[pName]))
        else:
            print("Other method to implement")

        return

    """ Fonction à nettoyer ! """

    def get_initial_parameters(self, randomPick = False, picks = 1000):
        min_incubation_time = 5
        max_incubation_time = 6

        min_presymptomatic_time = 1
        max_presymptomatic_time = 3

        min_symptomatic_time = 5
        max_symptomatic_time = 10

        mortality_rate_in_ICU = 0.279
        mortality_rate_in_simple_hospital_beds = 0.168

        avg_stay_in_ICU_in_case_of_death = 19.3
        avg_stay_in_simple_hospital_beds_in_case_of_death = 6.1

        avg_stay_in_ICU_in_case_of_recovery = 9.9
        avg_stay_in_hospital_simple_beds_in_case_of_recovery = 8

        # ----------------------------------
        # Tau (SP -> H) # -> won't be constant over time
        avg_time_for_transfer_from_SP_to_H = 5.7
        tau_0 = 0.01 / avg_time_for_transfer_from_SP_to_H  # 1 symptomatic out of 100 goes to the hospital # blind hypothesis
        tau_min = 0.0001 / avg_time_for_transfer_from_SP_to_H  # 1 symptomatic out of 10000 goes to the hospital # blind hypothesis
        tau_max = 0.1 / avg_time_for_transfer_from_SP_to_H  # 1 symptomatic out of 10 goes to the hospital # blind hypothesis

        # ----------------------------------
        # Gamma 4 (A -> R) # -> probably constant over time
        gamma4_max = 1 / min_incubation_time
        gamma4_min = 1 / (max_incubation_time + max_symptomatic_time)
        gamma4_0 = (gamma4_max + gamma4_min) / 2

        # ----------------------------------
        # Gamma1 (SP -> R) # -> probably constant over time
        gamma1_max = 1 / min_symptomatic_time
        gamma1_min = 1 / max_symptomatic_time
        gamma1_0 = (gamma1_max + gamma1_min) / 2

        # ----------------------------------
        # Gamma2 (H -> R) # -> probably constant over time
        gamma2_min = 0.2  # blind hypothesis
        gamma2_0 = (1 - mortality_rate_in_simple_hospital_beds) / avg_stay_in_hospital_simple_beds_in_case_of_recovery
        gamma2_max = 0.4  # blind hypothesis

        # ----------------------------------
        # Gamma3 (C -> R) # -> probably constant over time
        gamma3_min = 0.1  # blind hypothesis
        gamma3_0 = 0.05  # blind hypothesis
        gamma3_max = (1 - mortality_rate_in_ICU) / avg_stay_in_ICU_in_case_of_recovery

        # Discuter du bazard en dessous
        # ----------------------------------
        # Beta (S -> E) # -> will vary a lot over time
        R0_min = 0.1  # should be set < 1 if we want to permit a fall after a peak
        R0_max = 4
        R0_avg = (R0_min + R0_max) / 2
        infectious_time = (min_symptomatic_time + max_symptomatic_time) / 2
        beta_0 = R0_avg / infectious_time
        beta_min = R0_min / max_symptomatic_time
        beta_max = R0_max / min_symptomatic_time

        # ----------------------------------
        # Delta (H -> C) # -> should vary with the influence of the British variant
        fraction_of_hospitalized_not_transfering_to_ICU = 0.753
        delta_min = 0.01  # blind hypothesis
        delta_max = 0.06  # blind hypothesis
        delta_0 = (1 - fraction_of_hospitalized_not_transfering_to_ICU) / \
                  ((avg_stay_in_hospital_simple_beds_in_case_of_recovery + avg_stay_in_ICU_in_case_of_death) / 2)  # semi-blind hyptohesis

        # ----------------------------------
        # Rho (E -> A) # -> probably constant over time
        rho_max = 1 / min_incubation_time
        rho_0 = 2 / (min_incubation_time + max_incubation_time)
        rho_min = 1 / max_incubation_time

        # ----------------------------------
        # Theta (C -> F) # -> should vary with the influence of the British variant
        # Hypothesis: stay and mortality in simple hospital beds lower bounds the corresponding numbers in ICU
        theta_min = mortality_rate_in_simple_hospital_beds / avg_stay_in_ICU_in_case_of_death  # semi-blind hypothesis
        theta_max = mortality_rate_in_ICU / avg_stay_in_simple_hospital_beds_in_case_of_death  # semi-blind hypothesis
        theta_0 = mortality_rate_in_ICU / avg_stay_in_ICU_in_case_of_death

        # ----------------------------------
        # Sigma (A -> SP) # -> probably constant over time
        sigma_max = 1 / min_presymptomatic_time
        sigma_min = 1 / max_presymptomatic_time

        sigma_0 = (sigma_max + sigma_min) / 2

        # ----------------------------------
        # Mu (sigma * A -> T) # -> will vary over time with the test capacity and the testing rules
        mu_max = 0.9  # blind hypothesis
        mu_min = 0.4  # blind hypothesis
        mu_0 = (mu_min + mu_max) / 2  # blind hypothesis

        # ----------------------------------
        # Eta (T -> TP) # -> will vary a lot over time with the peak of contamination
        eta_max = 0.3288
        eta_min = 0.009
        eta_0 = 0.07

        # ----------------------------------
        # Alpha
        #alpha_min = 0.001
        #alpha_max = 0.999
        #alpha_0 = 0.01
        alpha_min = 0
        alpha_max = 0
        alpha_0 = 0
        #alpha_bounds = [0.001, 0.01, 0.95]
        
        theta2_min = 0
        theta2_max = 0.1
        theta2_0 = 0.02

        # ----------------------------------
        bestParams = [beta_0, rho_0, sigma_0, tau_0, delta_0, theta_0, theta2_0, gamma1_0, gamma2_0,
                      gamma3_0, gamma4_0, mu_0, eta_0]
        bestParams = dict(zip(self._paramNames, bestParams))

        if not(self._immunity):
            bestParams += [alpha_0]

        if randomPick:
            best = 0
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
                theta = random.uniform(theta_min, theta_max)
                mu = random.uniform(mu_min, mu_max)
                eta = random.uniform(eta_min, eta_max)

                paramValues = [beta, rho, sigma, tau, delta, theta, gamma1, gamma2,
                               gamma3, gamma4, mu, eta]
                if not(self._immunity):
                    alpha = random.uniform(alpha_min, alpha_max)
                    paramValues += [alpha]


                # Pas en dict ici car ça poserait un problème dans fit_parameters()
                score = self.plumb(paramValues)
                if score > best:
                    best = score
                    print("Score preprocessing parameters: {}".format(score))
                    bestParams = paramValues

            bestParams = dict(zip(self._paramNames, bestParams))
            print('Best preprocessing parameters: {}'.format(bestParams))
        #else:


        gamma1_bounds = [gamma1_min, bestParams['Gamma1'], gamma1_max]
        gamma2_bounds = [gamma2_min, bestParams['Gamma2'], gamma2_max]
        gamma3_bounds = [gamma3_min, bestParams['Gamma3'], gamma3_max]
        gamma4_bounds = [gamma4_min, bestParams['Gamma4'], gamma4_max]
        beta_bounds = [beta_min, bestParams['Beta'], beta_max]
        tau_bounds = [tau_min, bestParams['Tau'], tau_max]
        delta_bounds = [delta_min, bestParams['Delta'], delta_max]
        sigma_bounds = [sigma_min, bestParams['Sigma'], sigma_max]
        rho_bounds = [rho_min, bestParams['Rho'], rho_max]
        theta_bounds = [theta_min, bestParams['Theta'], theta_max]
        theta2_bounds = [theta2_min, bestParams['Theta'], theta2_max]
        mu_bounds = [mu_min, bestParams['Mu'], mu_max]
        eta_bounds = [eta_min, bestParams['Eta'], eta_max]

        bounds = [beta_bounds, rho_bounds, sigma_bounds, tau_bounds, delta_bounds,
                    theta_bounds, theta2_bounds, gamma1_bounds, gamma2_bounds, gamma3_bounds,
                    gamma4_bounds, mu_bounds, eta_bounds]

        if not(self._immunity):
            alpha_bounds = [alpha_min, bestParams['Alpha'], alpha_max]
            bounds += [alpha_bounds]

        params = Parameters()

        for name, bound in zip(self._paramNames, bounds):
            params.add(name, value = bound[1], min = bound[0], max = bound[2])

        return params

    """Partie déterminieste à faire"""
    def plumb(self, parameters):
        # TO DO: Partie déterministe!
        days = self._fittingPeriod[1]-self._fittingPeriod[0]
        params = dict(zip(self._paramNames, parameters))

        res = self.predict(end = days, parameters = params)

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


        if self._stochastic:
            lhs = dict()
            for state, obs, param in [(StateEnum.SYMPTOMATIQUE, ObsEnum.DHDT, params['Tau']),
                                      (StateEnum.DSPDT, ObsEnum.NUM_TESTED, params['Mu']),
                                      (StateEnum.DTESTEDDT, ObsEnum.NUM_POSITIVE, params['Eta'])]:
                # donc 1) depuis le nombre predit de personne SymPtomatique et le parametre tau, je regarde si l'observations dhdt est probable
                #      2) depuis le nombre predit de personne Critical et le parametre theta, je regarde si l'observations dfdt est probable
                #      3) sur la transition entre Asymptomatique et Symptomatique ( sigma*A -> dSPdt) avec le parmetre de test(mu), je regarde si l'observation num_tested est probable
                log_likelihood = 0
                for day in np.arange(0, days):
                    # Take all the values of experiments on a given day day_ndx
                    # for a given measurement (state.value)

                    observation = max(1, self._data[day + self._fittingPeriod[0]][obs.value])
                    values = experiments[:, day, state.value]  # binomial
                    prediction = np.mean(values)
                    try:
                        x = binom.pmf(observation, np.ceil(np.mean(prediction)), param)
                        log_bin = np.log(x)
                    except FloatingPointError as exception:
                        log_bin = -999
                    log_likelihood += log_bin

                lhs[obs] = log_likelihood
            return -sum(lhs.values())

        else:
            res = self.predict(end = days, parameters = params) # CASES_MUNI_CUM, CASES_AGESEX, CASES_MUNI, HOSP, MORT, TESTS, VACC
            residuals = np.sum(np.abs(residuals_error(res[self._fittingPeriod[0]:self._fittingPeriod[1],StateEnum.HOSPITALIZED.value], self._data[self._fittingPeriod[0]:self._fittingPeriod[1],ObsEnum.NUM_HOSPITALIZED.value])))
            residuals2 = np.sum(np.abs(residuals_error(res[self._fittingPeriod[0]:self._fittingPeriod[1],StateEnum.CRITICAL.value], self._data[self._fittingPeriod[0]:self._fittingPeriod[1],ObsEnum.NUM_CRITICAL.value])))
            residuals3 = np.sum(np.abs(residuals_error(res[self._fittingPeriod[0]:self._fittingPeriod[1],StateEnum.FATALITIES.value], self._data[self._fittingPeriod[0]:self._fittingPeriod[1],ObsEnum.NUM_FATALITIES.value])))
            #least_squares = np.sum(residuals*residuals)
            res = 1.5* residuals + 1.25* residuals2 + residuals3
            return res
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
        """
        gamma1 = parameters['Gamma1']
        gamma2 = parameters['Gamma2']
        gamma3 = parameters['Gamma3']
        gamma4 = parameters['Gamma4']
        beta = parameters['Beta']
        tau = parameters['Tau']
        delta = parameters['Delta']
        sigma = parameters['Sigma']
        rho = parameters['Rho']
        theta = parameters['Theta']
        mu = parameters['Mu']
        eta = parameters['Eta']
        
         """
         
        beta = parameters['Beta']
        rho = 0.18099572158739496
        sigma = 0.7000896123948177
        tau = 0.020977210561699513
        delta = 0.036657117821975185
        theta = 0.019505527274802967
        theta2 = 0.02934087511630468
        gamma1 = 0.143661768759768
        gamma2 = 0.20007998590287754
        gamma3 = 0.0736554358577431
        gamma4 = 0.12235615548208775
        mu = 0.65
        eta = 0.07
        alpha = 0
        if not(self._immunity):
            alpha = parameters['Alpha']

        if self._stochastic:
            betaS = self.population_leave(beta, S * (A + SP) / N)
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
            dHdt = tau * SP - delta * H - gamma2 * H - theta2 * H
            dCdt = delta * H - theta * C - gamma3 * C
            dFdt = theta * C + theta2 * H
            dRdt = gamma1 * SP + gamma2 * H + gamma3 * C + gamma4 * A - alphaR

            dHIndt = tau * SP
            dFIndt = theta * C
            dSPIndt = sigma * A
            DTESTEDDT = dSPIndt * mu
            DTESTEDPOSDT = DTESTEDDT * eta

        return [dSdt, dEdt, dAdt, dSPdt, dHdt, dCdt, dFdt, dRdt, dHIndt, dFIndt, dSPIndt, DTESTEDDT, DTESTEDPOSDT]


if __name__ == "__main__":
    observations = load_model_data()
    rows = np.array(observations)
    days = len(rows)
    #periods_in_days = [(12, 63), (64, 98), (99, 145), (146, 206), (207, 218), (219, 245), (246, 274), (275, 331), (332, 364), (365, 390), (391, 404),(405, 420)]
    dates = [observations.DATE.iloc[0].date(), date(2020, 3, 13), date(2020, 5, 4), date(2020, 6, 8),
             date(2020, 7, 25), date(2020, 9, 24), date(2020, 10, 6), date(2020, 11, 2),
             date(2020, 12, 1), date(2021, 1, 27), date(2021, 3, 1), date(2021, 3, 27), date(2021, 4, 12),
             observations.DATE.iloc[-1].date()]
    # list of tuples (start, end) for each period with significantly distinctive covid-19 measures
    periods_in_days = periods_in_days(dates)
    periods_in_days = periods_in_days[1:] # we start fitting from the 2nd period to start with higher values
    # solution 2, here start from 0. but use the 0 to compute the date so not cool... et marche moins bien que sol 1
    
    print("period in days: {}".format(periods_in_days))
    ms = SEIR_HCD(stocha = False)

    #ci 1
    N = 11492641 # population belge en 2020
    E0 = 80000
    A0 = 14544
    SP0 = 9686
    H0 = rows[periods_in_days[0][0]][ObsEnum.NUM_HOSPITALIZED.value]
    C0 = rows[periods_in_days[0][0]][ObsEnum.NUM_CRITICAL.value]
    R0 = np.sum(rows[:periods_in_days[0][0], ObsEnum.RSURVIVOR.value]) # = 0
    F0 = rows[periods_in_days[0][0]][ObsEnum.NUM_FATALITIES.value]
    S0 = N - E0 - A0 - SP0 - H0 - C0 - R0 - F0
    
    """
    N = 11492641 # population belge en 2020
    E0 = 500000
    A0 = round(E0 * 0.181818)
    SP0 = round(A0 * 0.666666)
    H0 = rows[periods_in_days[0][0]][ObsEnum.NUM_HOSPITALIZED.value]
    C0 = rows[periods_in_days[0][0]][ObsEnum.NUM_CRITICAL.value]
    R0 = np.sum(rows[:periods_in_days[0][0], ObsEnum.RSURVIVOR.value]) # = 0
    F0 = rows[periods_in_days[0][0]][ObsEnum.NUM_FATALITIES.value]
    S0 = N - E0 - A0 - SP0 - H0 - C0 - R0 - F0"""

    IC = [S0, E0, A0, SP0, H0, C0, F0, R0]
    print(IC)
    ms.set_IC(conditions = IC)

    sres = np.array([])
    for period in periods_in_days:
        print(f"Period: [{period[0]}, {period[1]}]")
        ms.fit_parameters(data = rows[period[0]:period[1], :], randomPick = False, picks = 1000)

        sres_temp = ms.predict()
        print(sres_temp.shape)
        if sres_temp.any():
            ms.set_IC(conditions = sres_temp[-1, 0:8])
            if not np.any(sres):
                sres = sres_temp[:13,:] * 0 #solution 1, artificielement mettre des 0 pour les X premier jours, où plus propre, mettre IC 13 fois à voir.
                sres = np.concatenate((sres, sres_temp)) # fait partie de solution 1
                # sres = sres_temp
            else:
                sres = np.concatenate((sres, sres_temp))

    version = 3
    print("sres")
    print(sres.shape)

    plt.figure()
    plt.title('HOSPITALIZED / PER DAY fit')
    t = StateEnum.DHDT
    plt.plot(sres[:, t.value], label = str(t) + " (model)")
    u = ObsEnum.DHDT
    plt.plot(rows[:, u.value], "--", label = str(u) + " (real)")
    #plt.savefig('img/v{}-dhdt.pdf'.format(version))
    plt.show()

    plt.figure()
    plt.title('Hospitalized')
    t = StateEnum.HOSPITALIZED
    plt.plot(sres[:, t.value], label = str(t) + " (model)")
    u = ObsEnum.NUM_HOSPITALIZED
    plt.plot(rows[:, u.value], "--", label = str(u) + " (real)")
    plot_periods(plt, dates)
    #plt.savefig('img/v{}-hospitalized.pdf'.format(version))
    plt.show()

    plt.figure()
    plt.title('Critical')
    t = StateEnum.CRITICAL
    plt.plot(sres[:, t.value], label = str(t) + " (model)")
    u = ObsEnum.NUM_CRITICAL
    plt.plot(rows[:, u.value], "--", label = str(u) + " (real)")
    plot_periods(plt, dates)
    #plt.savefig('img/v{}-critical.pdf'.format(version))
    plt.show()
    plt.figure()
    plt.title('FATALITIES')
    plot_periods(plt, dates)
    t = StateEnum.FATALITIES
    plt.plot(sres[:, t.value], label = str(t) + " (model)")
    u = ObsEnum.NUM_FATALITIES
    plt.plot(rows[:, u.value], "--", label = str(u) + " (real)")
    #plt.savefig('img/v{}-FATALITIES.pdf'.format(version))
    plt.show()
    plt.figure()
    plt.title('FATALITIES / PER DAY fit')
    t = StateEnum.DFDT
    plt.plot(sres[:, t.value], label = str(t) + " (model)")
    u = ObsEnum.DFDT
    plt.plot(rows[:, u.value], "--", label = str(u) + " (real)")
    #plt.savefig('img/v{}-dftf.pdf'.format(version))
    plt.show()
    plt.figure()
    plt.title('NUM_tested / PER DAY fit')
    t = StateEnum.DTESTEDDT
    plt.plot(sres[:, t.value], label = str(t) + " (model)")
    u = ObsEnum.NUM_TESTED
    plt.plot(rows[:, u.value], "--", label = str(u) + " (real)")
    #plt.savefig('img/v{}-dtesteddt.pdf'.format(version))
    plt.show()
    plt.figure()
    plt.title('NUM_Positive / PER DAY fit')
    t = StateEnum.DTESTEDPOSDT
    plt.plot(sres[:, t.value], label = str(t) + " (model)")
    u = ObsEnum.NUM_POSITIVE
    plt.plot(rows[:, u.value], "--", label = str(u) + " (real)")
    #plt.savefig('img/v{}-dtestedposdt.pdf'.format(version))
    plt.show()