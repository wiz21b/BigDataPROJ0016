import random
import matplotlib.pyplot as plt
import numpy as np
import random
from utils import ObsEnum, StateEnum, ObsFitEnum, StateFitEnum, Model, residuals_error, load_data, residual_sum_of_squares, log_residual_sum_of_squares, COLORS_DICT

from simul import simulation_model, model_update,partition_persons,persons,InfectablePool

random.seed(24)
np.random.seed(24)

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
            zip(['gamma1', 'gamma2', 'gamma3', 'gamma4', 'beta', 'tau', 'delta', 'sigma','rho','theta','mu','eta'],
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

        E0 = 15 * 2
        A0 = 10 * 2
        SP0 = 5 * 2
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
        mu = params['mu']
        eta = params['eta']

        S, E, A, SP, H, C, F, R, infected_per_day, R_out_HC, cumulI = initial_conditions
        cumulH = 0

        data = []

        for d in range(days):
            print("Days = "+ str(d) + "----------------------------------------")
            ys = [S, E, A, SP, H, C, F, R]

            if stochastic:
                if(d<= self._nb_observations+2):
                    dSdt, dEdt, dAdt, dSPdt, dHdt, dCdt, dFdt, dRdt, dHIndt,dFIndt,dSPIndt,DTESTEDDT,DTESTEDPOSDT= self._model_stochastic(ys, gamma1, gamma2, gamma3, gamma4, beta, tau, delta, sigma, rho, theta,mu,eta)
                    if ( d == self._nb_observations+2 ):
                        repartition = {
                            "S" : S+dSdt,
                            "E" : E+dEdt,
                            "A" : A+dAdt,
                            "SP" : SP+dSPdt,
                            "HCRF" : H + dHdt + C + dCdt + F + dFdt + R + dRdt
                        }
                        partition_persons(persons, repartition)
                        infected_people = [p for p in filter(lambda p: p.infected_A or p.infected_SP, persons)]
                        self.infectedPool = InfectablePool(infected_people)
                else:
                    dSdt, dEdt, dAdt, dSPdt, dHdt, dCdt, dFdt, dRdt, dHIndt,dFIndt,dSPIndt,DTESTEDDT,DTESTEDPOSDT= self._model_stochastic_measure(ys, gamma1, gamma2, gamma3, gamma4, beta, tau, delta, sigma, rho, theta,mu,eta)
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

        betaS = round(population_leave(beta, S * (A+SP) / N))
        rhoE = round(population_leave(rho, E))
        sigmaA = round(population_leave(sigma, A))
        gamma4A = round(population_leave(gamma4, A))
        tauSP = round(population_leave(tau, SP))
        gamma1SP = round(population_leave(gamma1, SP))
        deltaH = round(population_leave(delta, H))
        gamma2H = round(population_leave(gamma2, H))
        thetaC = round(population_leave(theta, C))
        gamma3C = round(population_leave(gamma3, C))
        muSP = round(population_leave(mu,sigmaA))# pas sur qu'il faut pas la moyenne
        etaSP = round(population_leave(eta,muSP))

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

    def _model_stochastic_measure(self, ys, gamma1, gamma2, gamma3, gamma4, beta, tau, delta, sigma, rho, theta,mu,eta):
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

        #betaS = population_leave((1- 0.2)* beta* S/N,  0.8* (A+SP) )
        #betaS2 = population_leave(beta*S/N,  0.1* (A+SP) )

        print("E = ------------------------------" + str(E))
        rhoE = round(population_leave(rho, E))
        sigmaA = round(population_leave(sigma, A))
        gamma4A = round(population_leave(gamma4, A))
        tauSP = round(population_leave(tau, SP))
        gamma1SP = round(population_leave(gamma1, SP))
        deltaH = round(population_leave(delta, H))
        gamma2H = round(population_leave(gamma2, H))
        thetaC = round(population_leave(theta, C))
        gamma3C = round(population_leave(gamma3, C))
        muSP = round(population_leave(mu,sigmaA))# pas sur qu'il faut pas la moyenne
        etaSP = round(population_leave(eta,muSP))

        betaS = simulation_model(persons,beta,self.infectedPool)

        print(betaS)
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

        model_update(persons,rhoE,sigmaA,gamma4A,tauSP,gamma1SP)

        return [dSdt, dEdt, dAdt, dSPdt, dHdt, dCdt, dFdt, dRdt, dHIndt,dFIndt,dSPIndt,DTESTEDDT,DTESTEDPOSDT]


if __name__ == "__main__":
    head, observations, rows = load_data()
    rows = np.array(rows)
    days = len(observations)
    gamma1 = 0.23
    gamma2 = 1/13
    gamma3 = 1/17
    gamma4 = 0.12
    beta = 0.32
    tau = 0.01
    delta = 0.025
    sigma = 0.6
    rho = 0.89
    theta = 0.04
    mu = 0.67
    eta = 0.8

    ms = SarahStat(rows, 1000324)
    ms._fit_params = ms._params_array_to_dict([gamma1, gamma2, gamma3,  gamma4, beta,  tau, delta, sigma, rho, theta,mu,eta])

    NB_EXPERIMENTS = 100
    PREDICTED_DAYS = 150

    print(f"Running {NB_EXPERIMENTS} experiments")
    experiments = [] # dims : [experiment #][day][value]

    for i in range(NB_EXPERIMENTS):
        sres = ms.predict_stochastic(PREDICTED_DAYS)
        experiments.append(sres)
    print("... done running experiments")

    experiments = np.stack(experiments)
    for state, obs in [(StateEnum.HOSPITALIZED, ObsEnum.NUM_HOSPITALIZED),
                        (StateEnum.DHDT,ObsEnum.DHDT),
                        (StateEnum.DFDT,ObsEnum.DFDT),
                        (StateEnum.DTESTEDDT,ObsEnum.NUM_TESTED),
                        (StateEnum.DTESTEDPOSDT,ObsEnum.NUM_POSITIVE),
                       (StateEnum.CRITICAL, ObsEnum.NUM_CRITICAL),
                       (StateEnum.FATALITIES, ObsEnum.NUM_FATALITIES)]:

        percentiles = np.stack(
            [np.percentile(experiments[:,day,state.value],[5,50,95])
             for day in range(PREDICTED_DAYS)])

        color = COLORS_DICT[state]

        plt.figure()
        plt.fill_between(range(PREDICTED_DAYS), percentiles[:,0],percentiles[:,2], facecolor=None, color=color,alpha=0.25,linewidth=0.0)
        #plt.fill_between(range(PREDICTED_DAYS), percentiles[:,1],percentiles[:,3], facecolor=None, color=color,alpha=0.25,linewidth=0.0)
        plt.plot(range(PREDICTED_DAYS), percentiles[:,1], color=color)

        plt.plot(rows[:, obs.value], "--", c=COLORS_DICT[obs], label=f"{obs} (real)")
        plt.savefig("img_final/no_mesure_"+str(state)+".pdf")

        plt.title(str(state))
    plt.show()

    PREDICTED_DAYS = 150

    plt.figure()
    for state in [(StateEnum.HOSPITALIZED),(StateEnum.CRITICAL),StateEnum.EXPOSED,StateEnum.RECOVERED,StateEnum.FATALITIES,
                    StateEnum.ASYMPTOMATIQUE,StateEnum.SYMPTOMATIQUE,StateEnum.SUSCEPTIBLE]:

        percentiles = np.stack(
            [np.percentile(experiments[:,day,state.value],[5,50,95])
             for day in range(PREDICTED_DAYS)])

        color = COLORS_DICT[state]

        plt.fill_between(range(PREDICTED_DAYS), percentiles[:,0],percentiles[:,2], facecolor=None, color=color,alpha=0.25,linewidth=0.0)
        plt.plot(range(PREDICTED_DAYS), percentiles[:,1], color=color, label=f"{obs}")

        #plt.plot(rows[:, state.value], "--", c=COLORS_DICT[state], label=f"{state}")

    plt.legend()
    plt.title("preview")
    plt.savefig("img_final/mask_allState.pdf")
    plt.show()
