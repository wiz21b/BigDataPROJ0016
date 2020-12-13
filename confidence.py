import matplotlib.pyplot as plt
import numpy as np

from utils import ObsEnum, StateEnum, ObsFitEnum, StateFitEnum, Model, residuals_error, load_data, residual_sum_of_squares, log_residual_sum_of_squares, COLORS_DICT


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
    def _params_array_to_dict(self, params):
        return dict(
            zip(['gamma1', 'gamma2', 'gamma3', 'gamma4', 'beta', 'tau', 'delta', 'sigma','rho','theta'],
                params))

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
        self._iterations = 0

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

    def predict(self, days):
        res = self._predict(self._initial_conditions, days, self._fit_params)
        return res

    def predict_stochastic(self, days):
        res = self._predict(self._initial_conditions, days, self._fit_params,stochastic = True)
        return res

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

        S, E, A, SP, H, C, F, R, infected_per_day, R_out_HC, cumulI = initial_conditions
        cumulH = 0

        data = []

        for d in range(days):
            ys = [S, E, A, SP, H, C, F, R]

            if stochastic:
                dSdt, dEdt, dAdt, dSPdt, dHdt, dCdt, dFdt, dRdt, dHOutdt = self._model_stochastic(ys, gamma1, gamma2, gamma3, gamma4, beta, tau, delta, sigma, rho, theta)
            else:
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

        # actually_infected=modelsimul(rho,....)

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



    def _model_stochastic(self, ys, gamma1, gamma2, gamma3, gamma4, beta, tau, delta, sigma, rho, theta):
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

        dHOutdt = tau*SP

        return [dSdt, dEdt, dAdt, dSPdt, dHdt, dCdt, dFdt, dRdt, dHOutdt]


if __name__ == "__main__":
    head, observations, rows = load_data()
    rows = np.array(rows)
    days = len(observations)

    ms = SarahStat(rows, 1000000)
    ms._fit_params = ms._params_array_to_dict([0.06902555, 0.63201217, 0.8476672,  0.40937882, 0.5985962,  0.73778851, 0.13290143, 0.08208902, 0.88753062, 0.10339359])

    NB_EXPERIMENTS = 100
    PREDICTED_DAYS = 80

    print(f"Running {NB_EXPERIMENTS} experiments")
    experiments = [] # dims : [experiment #][day][value]

    for i in range(NB_EXPERIMENTS):
        sres = ms.predict_stochastic(PREDICTED_DAYS)
        experiments.append(sres)
    print("... done running experiments")

    experiments = np.stack(experiments)

    for state, obs in [(StateEnum.HOSPITALIZED, ObsEnum.NUM_HOSPITALIZED),
                       (StateEnum.CRITICAL, ObsEnum.NUM_CRITICAL),
                       (StateEnum.FATALITIES, ObsEnum.NUM_FATALITIES),
                       (StateEnum.RSURVIVOR, ObsEnum.RSURVIVOR)]:

        percentiles = np.stack(
            [np.percentile(experiments[:,day,state.value],[0,5,50,95,100])
             for day in range(PREDICTED_DAYS)])

        color = COLORS_DICT[state]

        plt.figure()
        plt.fill_between(range(PREDICTED_DAYS), percentiles[:,0],percentiles[:,4], facecolor=None, color=color,alpha=0.25,linewidth=0.0)
        plt.fill_between(range(PREDICTED_DAYS), percentiles[:,1],percentiles[:,3], facecolor=None, color=color,alpha=0.25,linewidth=0.0)
        plt.plot(range(PREDICTED_DAYS), percentiles[:,2], color=color)

        plt.plot(rows[:, obs.value], "--", c=COLORS_DICT[obs], label=f"{obs} (real)")


        plt.title(str(state))
    plt.show()
