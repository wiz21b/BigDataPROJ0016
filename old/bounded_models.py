# coding=utf-8
import numpy as np
import math
from lmfit import minimize, Parameters, report_fit
from geneticalgorithm import geneticalgorithm as ga
from scipy.optimize import minimize as scipy_minimize

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
        tau_0 = (0.1+0.25)/2
        tau_bounds = [1/10, tau_0, 1/4]

        """
        # J'ai pas réussi à faire fonctionner les formules qui suivent
        # pour obtenir une estimation de gamma1_0.
        # Mais voilà quand même le raisonnement que j'avais suivi.
        #
        # R_survivors = cumulative_hospitalizations - H - C
        H = self._observations[:, ObsEnum.HOSPITALIZED.value]
        C = self._observations[:, ObsEnum.CRITICAL.value]
        R_survivors = cumulative_hospitalizations - H - C
        # R(t) = cumulative_positives(t-time_IR)
        #        - cumulative_hospitalizations(t)
        #        + R_survivors(t)
        # For t = 0, ..., time_IR - 1, R(t) = 0, no one has yet recovered,
        # we are at the beginning of the pandemic.
        # math.ceil is used to ensure that indexes are integers
        # np.maximum is used to ensure that no value is < 0
        time_IR = math.ceil(min_symptomatic_time * 0.5 + max_symptomatic_time * 0.5)
        R = np.maximum(0, cumulative_positives[:-time_IR]
            - cumulative_hospitalizations[time_IR:]
            + R_survivors[time_IR:])
        R = np.concatenate((np.zeros(time_IR, np.int8), R))
        # R_out_HC(t) = R_survivors(t) - R_survivors(t-1)
        # For t = 0, R_out_HC(t) = 0
        R_out_HC = np.maximum(0, R_survivors[1:] - R_survivors[:-1])
        R_out_HC = np.concatenate((np.zeros(1, np.int8), R_out_HC))
        # R_out_I = R - R_out_HC
        R_out_I = np.maximum(0, R - R_out_HC)
        # I -> R : - gamma1 * I
        # R_out_I(t) = gamma1 * I(t-1)
        # gamma1 = R_out_I(t) / I(t-1)
        I = self._observations[:, ObsEnum.POSITIVE.value]
        gamma1_0 = R_out_I / I
        gamma1_0[np.isnan(gamma1_0)] = 0 # replaces nan values by 0
        gamma1_0 = np.mean(gamma1_0)
        ##### PROBLEM with the value of  gamma1_0 #####
        #### gamma1_0 should be included in [0, 1] but is bigger than 1 ####
        """

        # ----------------------------------------------------
        # gamma1 : the rate at which people leave the infected
        # state to go to the recovered state. That is, people
        # who recover by themselves, without going to the
        # hospital.

        # "best-case": if people recover all in min_symptomatic_time
        gamma1_max = 1
        # "worst-case": if people do not recover and go to the H state
        gamma1_min = 0.02 # chosen arbitrarily

        # We want the probability of exiting in 1 day
        # the symptomatic period such that the probability distribution is uniformly distributed
        # between day 1 and day 14 (= min_symptomatic_time (4) + max_symptomatic_time (10)).
        # With such a probability distribution, from day 4 to day 10, we will
        # have 7 days out of the 14 to recover and be in the right symptomatic time,
        # ie: 50% of chance to recover during the right time period
        gamma1_0 = 0.2

        gamma1_bounds = [gamma1_min, gamma1_0, gamma1_max]

        # ------------------------------------------------------------
        # gamma2 : the rate at which people leave the H state to go to the R state
        # gamma3 : the rate at which people leave the C state to go to the R state

        # R_out_HC = gamma2 * H + gamma3 * C
        # Since everyone finally recovers in our model (no dead),
        # hospitalized and critical people will recover.
        # We just don't know in how much time they will recover ?
        # Thus, it is difficult to estimate gamma2 and gamma3 currently
        gamma2_0 = 0.2 # arbitrary choice
        gamma3_0 = 0.2 # arbitrary choice
        gamma2_bounds = [0.02, gamma2_0, 1]
        gamma3_bounds = [0.02, gamma3_0, 1]

        # ------------------------------------------------------------
        # gamma4 : the rate at which people leave the E state to go to the R state
        # "best-case": if people recover all directly after min_incubation_time
        gamma4_max = 1
        # "worst-case": if even after max_incubation_time, people do not recover because they are
        # asymptomatic for a long time, corresponding exactly to the time a symptomatic who is never hospitalised
        # would take to recover (max_incubation_time + max_symptomatic_time).
        gamma4_min = 0.02
        # "avg-case":
        gamma4_0 = 0.2
        gamma4_bounds = [gamma4_min, gamma4_0, gamma4_max]

        # ------------------------------------------------------------
        # Now we estimate the rate at which S changes.

        # The reproduction number R0 is the number of people
        # each infected person will infect during the time he is infectious
        # R0 = beta * infectious_time
        # We can make the assumption that once an individual becomes symptomatic,
        # he will not infect anyone else anymore as he will isolate himself under
        # the doctor's recommendations.
        # Under this assumption, infectious_time is
        # included in [min_incubation_time, max_incubation_time]
        # TO DISCUSS: For a SEIR model: a more complex formula of R0
        # found at https://en.wikipedia.org/wiki/Basic_reproduction_number
        # could be used later
        #R0_min = 1 # or else the virus is not growing exponentially
        #R0_max = 18 # the most virulent disease of all time: measles
        #R0_max = 2.8 * 1.5 # the most virulent influenza pandemic
        # and we were told that covid-20 looked similar to influenza
        # We multiply by 1.5 (arbitrary choice) because covid-20 might
        # become more virulent than the most virulent influenza pandemic
        # (which is the case for covid-19 with a R0 that got to 3-4 at peak period)
        #R0 = (R0_min + R0_max) / 2
        #infectious_time = (min_incubation_time + max_incubation_time) / 2
        #beta_0 = R0 / infectious_time
        #beta_min = R0_min / max_incubation_time
        #beta_max = R0_max / min_incubation_time

        #beta_min,beta_max = 0.1, 0.9
        #R0 = (R0_min + R0_max) / 2
        #infectious_time = (min_incubation_time + max_incubation_time) / 2
        #beta_0 = R0 / infectious_time
        #beta_min = R0_min / max_incubation_time
        #beta_max = R0_max / min_incubation_time

        beta_0 = 0.5  # on average each exposed person in contact with a susceptible person
        # will infect him with a probability 1/2
        beta_min = 0.01  # on average each exposed person in contact with a susceptible person
        # will infect him with a probability 1/100
        beta_max = 1  # on average each exposed person in contact with a susceptible person will infect him

        beta_bounds = [beta_min, beta_0, beta_max]

        # ------------------------------------------------------------
        # Now we study the rate at which C changes.

        # In the "worst-case" scenario (very low probability), every time someone
        # came in intensive care, he left the day after. In this case, we could
        # compute the cumulative number of people that went in intensive care by
        # summing all values of C
        cumulative_criticals_max = np.sum(self._observations[:, ObsEnum.NUM_CRITICAL.value])
        # In the "best-case" scenario (low probability), no one has left the intensive
        # care since the beginning. In this case, C represent the cumulative number of
        # people that went or are in intensive care.
        cumulative_criticals_min = self._observations[-1, ObsEnum.NUM_CRITICAL.value]
        # Similarly to what we did to evaluate tau_0, we could evaluate the
        # ratio of people that went from HOSPITALIZED to CRITICAL:
        # delta = cumulative_criticals / cumulative_hospitalizations
        delta_max = cumulative_criticals_max / cumulative_hospitalizations[-2]
        delta_max = max(0, min(delta_max, 1))
        delta_min = cumulative_criticals_min / cumulative_hospitalizations[-2]
        delta_min = min(max(delta_min, 0), 1)
        # The "best-case" scenario seems more probable
        # than the "worst-case" scenario, so
        # the weights are different (0.7 and 0.3, chosen arbitrarily)
        delta_0 = delta_min * 0.7 + delta_max * 0.3
        delta_bounds = [delta_min, delta_0, delta_max]
        
        # For the period of incubation
        rho_max = 1
        rho_0 = 3/5
        rho_min = 1/5
        rho_bounds = [rho_min,rho_0,rho_max]
        
        #For the death...
        theta_min = 0.005
        theta_max = 1
        theta_0 = 0.2
        theta_bounds = [theta_min,theta_0,theta_max]
        
        

        # In the case of an exponential GROWTH, we have
        # "worst-case": 100% of the exposed people will eventually become infected
        # Therefore, cumulative_positives(t) = cumulative_exposed(t - incubation_time)
        # Or alternatively, cumulative_positives(t + incubation_time) = cumulative_exposed(t)
        # sigma = cumulative_positives / cumulative_exposed
        # + The incubation time is very short, the number of exposed people
        # will move quicker to the infected state
        sigma_max = 1
        # "best-case": 0% of the exposed people will become infected, probably because they are all asymptomatic
        # and they do no test themselves. We would never study a virus that does not cause symptoms. So, let's
        # say that the virus cause 1 symptomatic for 100 exposed people each day
        sigma_min = 0.02 # = 1 / 100
        # "avg-case": 
        sigma_0 = 0.3
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

    def _model(self, ys, gamma1, gamma2, gamma3, gamma4, beta, tau, delta, sigma, rho, theta):
        S, E, A, SP, H, C, F, R = ys

        N = self._N

        # Interprétations selon NOTRE modèle:

        # Notre modèle ne fait que des liens simples ( S -> E -> I -> H -> C ).
        # Dans celui-ci, la somme des "num_positive" nous donne le total des
        # gens qui ont été infectés par le covid, "num_cumulative_positive".

        # Si on calcule:
        # "num_cumulative_hospitalizations / num_cumulative_positive"
        # Ceci nous donne la proportions de gens infectés qui ont été admis
        # à l'hopital. On a donc une première approximation du paramètre "tau"
        # qui représente le taux de transition des personnes infectées vers
        # l'hopital.

        # Puisque notre modèle ne considère pas que les gens meurent du virus
        # on peut connaitre à tout temps le nombre total de personnes qui se
        # sont rétablies en étant soit à l'hopital soit en soins intensifs.
        # Appelons ces persones "R_survivants".
        # R_survivants = num_cumulative_hospitalizations - num_hospitalised
        #                                                - num_critical


        # A chaque temps t on sait dans déterminer combien de personnes sortent
        # SOIT des soins intensifs SOIT de l'hopital, appelons cela "R_out_HC".
        # Pour cela il suffit de calculer:
        # "R_out_HC(t) = R_survivants(t) - R_survivants(t-1)"

        # "R_out_HC" corresponds, dans notre système d'équations, à:
        # "R_out_HC = gamma2 * H + gamma3 * C"
        # Ceci nous fournit donc une contrainte supplémentaire pouvant nous
        # permettre d'avoir une meilleure estimation des paramètres "gamma2"
        # ainsi que "gamma3".
        # TO THINK/DISCUSS: Comment intégrer cette équation? Elle ne doit pas
        # rentrer dans le modèle puisqu'il se fait intégrer. Peut-être quelque
        # part dans la fonction qui calcule l'erreur.

        # On ne peut par contre pas connaitre exactement le nombre de personne
        # faisant partie de la catégorie R à chaque instant. En effet, écrire
        # "R = num_cumulative_positive - num_cumulative_hospitalizations
        #                              + R_survivants"
        # serait inexacte. Cela reviendrait à considérer que les personnes
        # infectées guériraient du virus instantanément. Si l'on connaissait
        # le temps qu'il faut à une personne infectée pour devenir "saine" et
        # qu'on appelle cela "time_IR" on pourrait alors plus ou moins avoir
        # une estimation de R en écrivant:
        # "R(t) = num_cumulative_positive(t-time_IR)
        #                       - num_cumulative_hospitalizations(t)
        #                       + R_survivants(t)"
        # Car les personnes qui sont sorties de la catégorie I pour aller
        # dans la catégorie R au temps t vaut
        # "num_cumulative_positive(t-time_IR)
        # car les seulent personnes déclarées positives ces time_IR derniers
        # jours sont considérées comme encore infectées.

        # Note: R_cummul = R puisqu'une fois dans la catégorie R on y reste.
        # Donc R a déjà une notion de cummul.

        # Au temps t on a "num_positive" nouveaux cas mais on a réellement
        # "num_positive / sensitivity" nouveaux cas.
        # Pour l'instant dans le modèle, les gens sont susceptibles d'être
        # infectés non pas seulement par les gens qui sont actuellement
        # infectés et detectés mais aussi par ceux qui sont infectés et qui
        # n'ont pas été détectés.

        # Dans le futur, si jamais on nous indique un nombre d'occupation
        # maximal dans les hopitaux ou en soins intensifs, il faudra surement
        # le prendre en compte dans les équations car si tous les lits sont
        # occupés, personnes ne pourra donc se déplacer vers la catégorie
        # dans laquelle il manque de la place.

        # Modèle SEIHCR
        # Liens et paramètres:
        # S -> E : - (beta * S * E )/ N
        # E <- S : (beta * S * E )/ N
        # E -> I : - sigma * E
        # I <- E : sigma * E
        # I -> R : - gamma1 * I |&| I -> H : - tau * I
        # H <- I : tau * I
        # H -> R : - gamma2 * H |&| H -> C : - delta * H
        # C <- H : delta * H
        # C -> R : - gamma3 * C
        # R <- I : gamma1 * I |&| R <- H : gamma2 * H |&| R <- C : gamma3 * C

        # Le système d'équations serait le suivant:

# ROBIN afin de garder en tête que sigma * E est notre num_positive
# on a sigma * E qui est bien les nouveaux positive par jour ! 
# rho*E sont les personnes qui sortent d'incubation et deviennent infectueuse
# donc rho est de l'ordre de 1-4jours
# Ensuite on est infectueux et là on reste infecteux jusque la fin 
# la fin c'est quoi : soit on est testé postive soit non
# si on est testé positif, on est envoyé vers I avec le parametre sigma*E qui 
# correspond toujours a notre num_positif.
# Et si on est pas testé positif on reste dans la case des infectious pouvant etre testé
# positif a tout moment ou recover avec le paramètre gamma4.
# # HYP : On doit être testé positif pour etre a l'hopital 

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

        # The genetic algorithm uses an error represented
        # as a single float.

        #print( res[rselect])
        #return error_func(res[rselect], self._fittingObservations)

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
    sres = ms.predict(170)

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

    # """
    # plt.figure()
    # for u in [ObsEnum.POSITIVE, ObsEnum.HOSPITALIZED, ObsEnum.CRITICAL]:
    #     plt.plot(rows[:, u.value], label=f"{u} (real)")
    # plt.title('Simple Observations')
    # plt.xlabel('Days')
    # plt.ylabel('Individuals')
    # plt.legend()
    # plt.show()
    # """

    # """
    # plt.figure()
    # for u in [ObsEnum.CUMULATIVE_HOSPITALIZATIONS, ObsEnum.CUMULATIVE_POSITIVE, ObsEnum.CUMULATIVE_TESTED]:
    #     plt.plot(rows[:, u.value], label=f"{u} (real)")
    # plt.title('Cumulative Observations')
    # plt.xlabel('Days')
    # plt.ylabel('Individuals')
    # plt.legend()
    # plt.show()
    # """

    # # -------------------------------------------------------------
    # """
    # ms = Sarah1GA(rows, 1000000)
    # ms.fit_parameters(residuals_error)
    # sres = ms.predict(50)
    # plt.figure()
    # for t, u in zip([StateEnum.INFECTIOUS, StateEnum.HOSPITALIZED], [ObsEnum.POSITIVE, ObsEnum.HOSPITALIZED]):
    #     plt.plot(sres[:, t.value], label=f"{t} (model)")
    #     plt.plot(rows[:, u.value], label=f"{u} (real)")
    # plt.title('GA fit')
    # plt.xlabel('Days')
    # plt.ylabel('Individuals')
    # plt.ylim(0, 1000)
    # plt.legend()
    # plt.show()
    # """
