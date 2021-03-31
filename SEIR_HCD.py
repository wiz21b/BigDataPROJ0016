
import random
import numpy as np
import math

from lmfit import Parameters
from scipy.optimize import minimize as scipy_minimize

from utils import Model, ObsEnum, StateEnum, ObsFitEnum, StateFitEnum, residuals_error, load_model_data, residual_sum_of_squares, log_residual_sum_of_squares, COLORS_DICT

import matplotlib.pyplot as plt
from scipy.stats import binom

random.seed(1000)
np.random.seed(1000)

class SEIR_HCD(Model):
    def __init__ (self, stocha = True, immunity = False, errorFct = None, nbExperiments = 100):
        super().__init__(stocha = stocha, errorFct = errorFct, nbExperiments = nbExperiments)
        self._immunity = immunity # ne sert pas à l'instant xar posait un problème
                                  # si set à False car alors on avait un paramètre
                                  # dont les bounds étaient 0

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
                            'Gamma1',
                            'Gamma2',
                            'Gamma3',
                            'Gamma4',
                            'Mu',
                            'Eta']

    def set_IC(self, conditions):
        # ATTENTION: Pour le moment les deux conditions initiales sont différentes.
        # Dans la version précédente il y avait infected_per_day, R_out_HC, cumulI
        # A voir si on doit les intégrer !
        if not(len(conditions) == len(self._compartmentNames)):
            print("ERROR: Number of initial conditions given not matching with the model.")
            return

        self._initialConditions = dict(zip(self._compartmentNames, conditions))
        self._currentState = dict(zip(self._compartmentNames, conditions))
        self._population = sum(conditions)
        self._ICInitialized = True
        return

    # J'ai mis ça là mais je ne sais pas encore si je l'utiliserai
    def set_param(self, parameters):
        requiredParameters = len(self._paramNames)

        if not(len(parameters) == requiredParameters):
            print("ERROR: Number of parameters given not matching with the model.")
            return

        if not(self._immunity):
            self._params = dict(zip(self._paramNames, parameters + [0]))
        else:
            self._params = dict(zip(self._paramNames, parameters))
        self._paramInitialized = True
        return

    # J'ai mis ça là mais je ne sais pas encore si je l'utiliserai
    def set_state(self, compartments):
        if not(len(compartments) == len(self._compartmentNames)):
            print("ERROR: Number of initial conditions given not matching with the model.")
            return

        self._currentState = dict(zip(self._compartmentNames, compartments))
        return

    def fit_parameters(self, data = None, optimizer = 'LBFGSB', errorFct = None, 
                       randomPick = False, 
                       picks = 1000):

        if not(self._ICInitialized):
            print('ERROR: Inital conditions not initialized.')
            return
        if isinstance(data, np.ndarray):
            self._data = data
            self._dataLength = data.shape[0]
        else:
            print("ERROR: Data required")
            return
        
        

        # L-BFGS-B accepts bounds
        np.seterr(all = 'raise')

        # Find first set of parameters
        parameters = self.get_initial_parameters(randomPick = randomPick, picks = picks)
        bounds = np.array([(p.min, p.max) for pName, p in parameters.items()])

        # Group parameters
        print('Parameters\' bounds:')
        for pName, p in parameters.items():
            print("{:10s} [{:.2f} - {:.2f}]".format(pName, p.min, p.max))

        x0 = [p.value for p_name, p in parameters.items()]
        print("Initial guess for the parameters: {}".format(x0))


        if optimizer == 'LBFGSB':
            res = scipy_minimize(self.plumb, #plum scipy stocha
                                x0 = x0,
                                method = 'L-BFGS-B',
                                bounds = bounds)

            self._optimalParams = dict(zip(self._paramNames, res.x))
            self._fitted = True

            print('Optimal parameters after the fitting:')
            for pName, p in parameters.items():
                print("{:10s} [{:.2f} - {:.2f}] : {:.2f}".format(pName, p.min, p.max, 
                                                                 self._optimalParams[pName]))
        else:
            print("Other method to implement")

        return

    # Fonction à nettoyer !
    def get_initial_parameters(self, randomPick = False, picks = 1000):
        min_incubation_time = 1
        max_incubation_time = 5
        min_symptomatic_time = 4
        max_symptomatic_time = 10

        # ----------------------------------
        # Tau (SP -> H)
        tau_0 = 0.01
        tau_min = 0.001
        tau_max = 0.1

        # ----------------------------------
        # Gamma 4 ()
        gamma4_max = 1 / min_incubation_time
        gamma4_min = 1 / (max_incubation_time + max_symptomatic_time)
        gamma4_0 = 0.12

        # ----------------------------------
        # Gamma1 (SP -> R)
        gamma1_max = 1 / min_symptomatic_time
        gamma1_min = 1 / max_symptomatic_time
        gamma1_0 = 0.23

        # ----------------------------------
        # Gamma2 (H -> R) & Gamma3 (C -> R)
        gamma2_min = 1 / 15
        gamma2_0 = 1 / 13  # 0.2  # arbitrary choice
        gamma2_max = 0.5

        # ----------------------------------
        # Gamma3 ()
        gamma3_min = 1 / 20
        gamma3_0 = 1 / 19
        gamma3_max = 0.5
        
        # Discuter du bazard en dessous
        # ----------------------------------
        # Beta
        #R0_min = 1  # or else the virus is not growing exponentially
        #R0_max = 2.8 * 1.5  # the most virulent influenza pandemic
        #R0_avg = (R0_min + R0_max) / 2
        #infectious_time = (min_symptomatic_time + max_symptomatic_time) / 2
        # beta_0 = R0_avg / infectious_time  
        # beta_min = R0_min / max_symptomatic_time
        # beta_max = R0_max / min_symptomatic_time
        beta_0 = 0.25
        beta_min = 0.2
        beta_max = 0.55

        # ----------------------------------
        # Delta ()
        delta_min = 1 / 100  # 1/10
        delta_max = 57 / 1297
        delta_0 = 0.025

        # ----------------------------------
        # Rho ()
        rho_max = 1
        rho_0 = 0.89  # 2 / (min_incubation_time + max_incubation_time)
        rho_min = 1 / max_incubation_time

        # ----------------------------------
        # Theta ()
        theta_min = 0.01
        theta_max = 0.2
        theta_0 = 0.04

        # ----------------------------------
        # Sigma ()
        sigma_max = 0.7
        sigma_min = 0.5
        sigma_0 = 0.6

        # ----------------------------------
        # Mu () 
        mu_max = 0.90
        mu_min = 0.5
        mu_0 = 0.67

        # ----------------------------------
        # Eta ()
        eta_max = 0.85
        eta_min = 0.7
        eta_0 = 0.8

        # ----------------------------------
        # Alpha 
        """if not(self._immunity):
            alpha_max = 0
            alpha_min = 0
            alpha_0 = 0.000000000000001
        else:
            alpha_min = 0.001
            alpha_max = 0.95
            alpha_0 = 0.01
            #alpha_bounds = [0.001, 0.01, 0.95]"""

        # ----------------------------------
        bestParams = [beta_0, rho_0, sigma_0, tau_0, delta_0, theta_0, gamma1_0, gamma2_0,
                      gamma3_0, gamma4_0, mu_0, eta_0]

        # Avant on imposait la valeur initiale que les paramètres avaient
        # A discuter de pourquoi on avait fait ça et à enlever !
        bestParams = dict(zip(self._paramNames, [0.25702686734733055, 0.8299887995317639,
                                                 0.5941738834504704, 0.019951926351256127,
                                                 0.015355404916721221, 0.18273883639218028,
                                                 0.10698028629231761, 0.36806168249356885,
                                                 0.19689582431827574, 0.5158771155529966,
                                                 0.8222813790202658, 0.7463314605020426,
                                                 0.0]))

        if randomPick:
            best = 0
            for test in range(picks):
                if (test % (picks/10) == 0):
                    print("Pre test of the parameters: {} of {}".format(test + 1, picks))

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
                #alpha = random.uniform(alpha_min, alpha_max)
                paramValues = [beta, rho, sigma, tau, delta, theta, gamma1, gamma2, 
                               gamma3, gamma4, mu, eta]

                # Pas en dict ici car ça poserait un problème dans fit_parameters()
                score = self.plumb(paramValues)
                if score > best:
                    best = score
                    print("Score preprocessing parameters: {}".format(score))
                    bestParams = paramValues

            bestParams = dict(zip(self._paramNames, bestParams))
            print('Best preprocessing parameters: {}'.format(bestParams))

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
            mu_bounds = [mu_min, bestParams['Mu'], mu_max]
            eta_bounds = [eta_min, bestParams['Eta'], eta_max]
            #alpha_bounds = [alpha_min, bestParams['Alpha'], alpha_max]

        bounds = [beta_bounds, rho_bounds, sigma_bounds, tau_bounds, delta_bounds,
                    theta_bounds, gamma1_bounds, gamma2_bounds, gamma3_bounds,
                    gamma4_bounds, mu_bounds, eta_bounds]

        params = Parameters()

        for name, bound in zip(self._paramNames, bounds):
            params.add(name, value = bound[1], min = bound[0], max = bound[2])

        return params


    def plumb(self, parameters, skipDays = 10):
        # TO DO: Partie déterministe!
        days = len(self._data)
        params = dict(zip(self._paramNames, parameters))

        if self._stochastic:
            experiments = []  # dims : [experiment #][day][value]

            for i in range(self._nbExperiments):
                res = self.predict(end = days, parameters = params)
                experiments.append(res)
            # print("... done running experiments")

            experiments = np.stack(experiments)
        else:
            res = self._predict(end = days, parameters = params)

        lhs = dict()

        for state, obs, param in [(StateEnum.SYMPTOMATIQUE, ObsEnum.DHDT, params['Tau']),
                                  (StateEnum.DSPDT, ObsEnum.NUM_TESTED, params['Mu']),
                                  (StateEnum.DTESTEDDT, ObsEnum.NUM_POSITIVE, params['Eta'])]:
            # donc 1) depuis le nombre predit de personne SymPtomatique et le parametre tau, je regarde si l'observations dhdt est probable
            #      2) depuis le nombre predit de personne Critical et le parametre theta, je regarde si l'observations dfdt est probable
            #      3) sur la transition entre Asymptomatique et Symptomatique ( sigma*A -> dSPdt) avec le parmetre de test(mu), je regarde si l'observation num_tested est probable
            log_likelihood = 0
            for day in np.arange(skipDays, days):
                # Take all the values of experiments on a given day day_ndx
                # for a given measurement (state.value)

                observation = max(1, self._data[day][obs.value])
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

    def predict(self, start = 0, end = None, parameters = None):
        if not(end):
            end = len(self._data)
        params = parameters
        if not(parameters):
            if self._fitted:
                params = self._optimalParams
            else:
                print('ERROR: Finding optimal parameters is required!')
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
        theta = parameters['Theta']
        mu = parameters['Mu']
        eta = parameters['Eta']
        #alpha = parameters['Alpha']

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

            dSdt = -betaS
            dEdt = betaS - rhoE
            dAdt = rhoE - sigmaA - gamma4A
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
        else:
            dSdt = -beta * S * (A + SP) / N
            dEdt = beta * S * (A + SP) / N - rho * E
            dAdt = rho * E - sigma * A - gamma4 * A
            dSPdt = sigma * A - tau * SP - gamma1 * SP
            dHdt = tau * SP - delta * H - gamma2 * H
            dCdt = delta * H - theta * C - gamma3 * C
            dFdt = theta * C
            dRdt = gamma1 * SP + gamma2 * H + gamma3 * C + gamma4 * A

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

    ms = SEIR_HCD()

    N = 11000000
    E0 = 3000
    A0 = 2000
    SP0 = 1000
    H0 = rows[0][ObsEnum.NUM_HOSPITALIZED.value]
    C0 = rows[0][ObsEnum.NUM_CRITICAL.value]
    R0 = 0
    F0 = rows[0][ObsEnum.NUM_FATALITIES.value]
    S0 = N - E0 - A0 - SP0 - H0 - C0 - R0 - F0

    IC = [S0, E0, A0, SP0, H0, C0, F0, R0]

    ms.set_IC(conditions = IC)

    ms.fit_parameters(data = rows, randomPick = True, picks = 100)
    
    params = [0.6, 0.89, 0.65, 0.1, 0.04, 0.08, 0.12, 0.33, 0.28, 0.94, 0.62, 0.7]
    params = dict(zip(ms._paramNames, params))

    sres = ms.predict(end = days + 100, parameters = params)

    version = 3

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
    #plt.savefig('img/v{}-hospitalized.pdf'.format(version))
    plt.show()

    plt.figure()
    plt.title('Critical')
    t = StateEnum.CRITICAL
    plt.plot(sres[:, t.value], label = str(t) + " (model)")
    u = ObsEnum.NUM_CRITICAL
    plt.plot(rows[:, u.value], "--", label = str(u) + " (real)")
    #plt.savefig('img/v{}-critical.pdf'.format(version))
    plt.show()

    plt.figure()
    plt.title('FATALITIES')
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
