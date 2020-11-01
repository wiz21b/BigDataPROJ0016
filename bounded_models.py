# coding=utf-8
import numpy as np
import math
from lmfit import minimize, Parameters, report_fit
from geneticalgorithm import geneticalgorithm as ga

import matplotlib.pyplot as plt
from utils import ObsEnum, StateEnum, ObsFitEnum, StateFitEnum, Model, residuals_error, load_data, residual_sum_of_squares, log_residual_sum_of_squares

class Sarah1(Model):

    def __init__(self, observations, N):

        self._N = N
        nb_observations = observations.shape[0]

        self._observations = np.array(observations)
        self._fittingObservations = observations[np.ix_(range(nb_observations),
                                                        list(map(int, ObsFitEnum)))]

        # Je n'ai pas trouvé de conditions initiales pour E0 et I0 qui m'évite le message:
        # Warning: uncertainties could not be estimated
        E0 = 8
        I0 = 3
        H0 = self._observations[0][ObsEnum.HOSPITALIZED.value]
        C0 = self._observations[0][ObsEnum.CRITICAL.value]
        R0 = 0
        S0 = self._N - E0 - I0 - R0 - H0 - C0

        self._initial_conditions = [S0, E0, I0, H0, C0, R0]

        print(self._initial_conditions)
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
        cumulative_positives = np.cumsum(self._observations[:, ObsEnum.POSITIVE.value])
        cumulative_hospitalizations = self._observations[:, ObsEnum.CUMULATIVE_HOSPITALIZATIONS.value]
        # There is a difference of 1 between the indexes because to have the possibility
        # to go from one state to another, at least 1 day must pass. We can not go from
        # infected to hospitalized the same day.
        tau_0 = cumulative_hospitalizations[-1] / cumulative_positives[-2]
        tau_bounds = [tau_0 * (1 - error_margin), tau_0, max(1, tau_0 * (1 + error_margin))]

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
        gamma1_max = 1 / min_symptomatic_time
        # "worst-case": if people do not recover and go to the H state
        gamma1_min = 0.02 # chosen arbitrarily
        # "avg-case"

        # We want the probability of exiting in 1 day
        # the symptomatic period such that the probability distribution is uniformly distributed
        # between day 1 and day 14 (= min_symptomatic_time (4) + max_symptomatic_time (10)).
        # With such a probability distribution, from day 4 to day 10, we will
        # have 7 days out of the 14 to recover and be in the right symptomatic time,
        # ie: 50% of chance to recover during the right time period
        gamma1_0 = 1 / (max_symptomatic_time + min_symptomatic_time)

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
        gamma4_max = 1 / min_incubation_time
        # "worst-case": if even after max_incubation_time, people do not recover because they are
        # asymptomatic for a long time, corresponding exactly to the time a symptomatic who is never hospitalised
        # would take to recover (max_incubation_time + max_symptomatic_time).
        gamma4_min = 1 / (max_incubation_time + max_symptomatic_time)
        # "avg-case":
        gamma4_0 = (gamma4_max + gamma4_min) / 2
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
        R0_min = 1 # or else the virus is not growing exponentially
        #R0_max = 18 # the most virulent disease of all time: measles
        R0_max = 2.8 * 1.5 # the most virulent influenza pandemic
        # and we were told that covid-20 looked similar to influenza
        # We multiply by 1.5 (arbitrary choice) because covid-20 might
        # become more virulent than the most virulent influenza pandemic
        # (which is the case for covid-19 with a R0 that got to 3-4 at peak period)
        R0 = (R0_min + R0_max) / 2
        infectious_time = (min_incubation_time + max_incubation_time) / 2
        beta_0 = R0 / infectious_time
        beta_min = R0_min / max_incubation_time
        beta_max = R0_max / min_incubation_time
        beta_bounds = [beta_min, beta_0, beta_max]

        # ------------------------------------------------------------
        # Now we study the rate at which C changes.

        # In the "worst-case" scenario (very low probability), every time someone
        # came in intensive care, he left the day after. In this case, we could
        # compute the cumulative number of people that went in intensive care by
        # summing all values of C
        cumulative_criticals_max = np.sum(self._observations[:, ObsEnum.CRITICAL.value])
        # In the "best-case" scenario (low probability), no one has left the intensive
        # care since the beginning. In this case, C represent the cumulative number of
        # people that went or are in intensive care.
        cumulative_criticals_min = self._observations[-1, ObsEnum.CRITICAL.value]
        # Similarly to what we did to evaluate tau_0, we could evaluate the
        # average ratio of people that went from HOSPITALIZED to CRITICAL:
        # delta = cumulative_criticals / cumulative_hospitalizations
        delta_max = cumulative_criticals_max / cumulative_hospitalizations[-2]
        delta_max = max(0, min(delta_max, 1))
        delta_min = cumulative_criticals_min / cumulative_hospitalizations[-2]
        delta_min = min(max(delta_min, 0), 1)
        # The "best-case" scenario seems more probable
        # than the "worst-case scenario", so
        # the weights are different (0.7 and 0.3, chosen arbitrarily)
        delta_0 = delta_min * 0.7 + delta_max * 0.3
        delta_bounds = [delta_min, delta_0, delta_max]


        # In the case of an exponential GROWTH, we have
        # "worst-case": 100% of the exposed people will eventually become infected
        # Therefore, cumulative_positives(t) = cumulative_exposed(t - incubation_time)
        # Or alternatively, cumulative_positives(t + incubation_time) = cumulative_exposed(t)
        # sigma = cumulative_positives / cumulative_exposed
        # + The incubation time is very short, the number of exposed people
        # will move quicker to the infected state
        cumulative_exposed = cumulative_positives[-1]
        sigma_max = cumulative_positives[-min_incubation_time-1] / cumulative_exposed
        sigma_max = max(0, min(sigma_max, 1))
        # "best-case": 0% of the exposed people will become infected, probably because they are all asymptomatic
        # and they do no test themselves. We would never study a virus that does not cause symptoms. So, let's
        # say that the virus cause 1 symptomatic for 100 exposed people each day
        sigma_min = 0.01 # = 1 / 100
        # "avg-case":
        sigma_0 = (sigma_max + sigma_min) / 2
        sigma_bounds = [sigma_min, sigma_0, sigma_max]

        bounds = [gamma1_bounds, gamma2_bounds, gamma3_bounds, gamma4_bounds, beta_bounds, tau_bounds, delta_bounds, sigma_bounds]
        param_names = ['gamma1', 'gamma2', 'gamma3', 'gamma4', 'beta', 'tau', 'delta', 'sigma']
        params = Parameters()

        for param_str, param_bounds in zip(param_names, bounds):
            params.add(param_str, value=param_bounds[1], min=param_bounds[0], max=param_bounds[2])

        return params

    def fit_parameters(self, error_func):

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

        values = initial_conditions
        states_over_days = [values + [0]]
        S0, E0, I0, H0, C0, R0 = initial_conditions

        # days - 1 because day 0 is the initial conditions
        for day in range(days - 1):
            dSdt, dEdt, dIdt, dHdt, dCdt, dRdt = self._model(values, gamma1, gamma2, gamma3, gamma4, beta, tau, delta, sigma)
            S, E, I, H, C, R = values
            infected_per_day = sigma * E
            S = S+dSdt
            E = E+dEdt
            I = I+dIdt
            H = H+dHdt
            C = C+dCdt
            R = R+dRdt

            values = [S, E, I, H, C, R]
            states_over_days.append(values + [infected_per_day])

        return np.array(states_over_days)

    def _model(self, y, gamma1, gamma2, gamma3, gamma4, beta, tau, delta, sigma):
        S, E, I, H, C, R = y

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
        # S -> E : - (beta * S * I )/ N
        # E <- S : (beta * S * I )/ N
        # E -> I : - sigma * E
        # I <- E : sigma * E
        # I -> R : - gamma1 * I |&| I -> H : - tau * I
        # H <- I : tau * I
        # H -> R : - gamma2 * H |&| H -> C : - delta * H
        # C <- H : delta * H
        # C -> R : - gamma3 * C
        # R <- I : gamma1 * I |&| R <- H : gamma2 * H |&| R <- C : gamma3 * C

        # Le système d'équations serait le suivant:

        dSdt = -beta * S * I / N
        dEdt = beta * S * I / N - sigma * E - gamma4 * E
        dIdt = sigma * E - gamma1 * I - tau * I
        dHdt = tau * I - gamma2 * H - delta * H
        dCdt = delta * H - gamma3 * C
        dRdt = gamma1 * I + gamma2 * H + gamma3 * C + gamma4 * E

        return [dSdt, dEdt, dIdt, dHdt, dCdt, dRdt]

    def _plumb_lmfit(self, params, days, error_func):
        assert error_func == residuals_error, "lmfit requires residuals errors"

        res = self._predict(self._initial_conditions, days, params)

        rselect = np.ix_(range(res.shape[0]),
                         list(map(int, StateFitEnum)))

        return error_func(res[rselect], self._fittingObservations).ravel()

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

    def _params_array_to_dict(self, params):
        return dict(
            zip(['gamma1', 'gamma2', 'gamma3', 'gamma4', 'beta', 'tau', 'delta', 'sigma'],
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
    ms.fit_parameters(residuals_error)
    #ms.fit_parameters_ga(log_residual_sum_of_squares)
    sres = ms.predict(300)

    plt.figure()
    for t in [StateEnum.HOSPITALIZED, StateEnum.CRITICAL]:
        plt.plot(sres[:, t.value], label=f"{t} (model)")

    for u in [ObsEnum.HOSPITALIZED, ObsEnum.CRITICAL]:
        plt.plot(rows[:, u.value], label=f"{u} (real)")
    plt.title('LM fit')
    plt.xlabel('Days')
    plt.ylabel('Individuals')
    prediction_days = 10 # prediction at prediction_days
    plt.xlim(0, days + prediction_days)
    plt.ylim(0, 80)
    plt.legend()
    plt.show()

    plt.figure()
    for t in [StateEnum.INFECTIOUS, StateEnum.HOSPITALIZED, StateEnum.CRITICAL]:
        plt.plot(sres[:, t.value], label=f"{t} (model)")

    plt.title('Infectious - Hospitalized - Critical')
    plt.xlabel('Days')
    plt.ylabel('Individuals')
    plt.legend()
    plt.show()

    plt.figure()
    for t in [StateEnum.SUCEPTIBLE, StateEnum.INFECTIOUS, StateEnum.HOSPITALIZED,
              StateEnum.CRITICAL, StateEnum.RECOVERED]:
        plt.plot(sres[:, t.value], label=f"{t} (model)")
    plt.title('States')
    plt.xlabel('Days')
    plt.ylabel('Individuals')
    plt.legend()
    plt.show()

    """
    plt.figure()
    for u in [ObsEnum.POSITIVE, ObsEnum.HOSPITALIZED, ObsEnum.CRITICAL]:
        plt.plot(rows[:, u.value], label=f"{u} (real)")
    plt.title('Simple Observations')
    plt.xlabel('Days')
    plt.ylabel('Individuals')
    plt.legend()
    plt.show()
    """

    """
    plt.figure()
    for u in [ObsEnum.CUMULATIVE_HOSPITALIZATIONS, ObsEnum.CUMULATIVE_POSITIVE, ObsEnum.CUMULATIVE_TESTED]:
        plt.plot(rows[:, u.value], label=f"{u} (real)")
    plt.title('Cumulative Observations')
    plt.xlabel('Days')
    plt.ylabel('Individuals')
    plt.legend()
    plt.show()
    """

    # -------------------------------------------------------------
    """
    ms = Sarah1GA(rows, 1000000)
    ms.fit_parameters(residuals_error)
    sres = ms.predict(50)

    plt.figure()
    for t, u in zip([StateEnum.INFECTIOUS, StateEnum.HOSPITALIZED], [ObsEnum.POSITIVE, ObsEnum.HOSPITALIZED]):
        plt.plot(sres[:, t.value], label=f"{t} (model)")
        plt.plot(rows[:, u.value], label=f"{u} (real)")

    plt.title('GA fit')
    plt.xlabel('Days')
    plt.ylabel('Individuals')
    plt.ylim(0, 1000)
    plt.legend()
    plt.show()
    """
