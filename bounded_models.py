import numpy as np
import math
from scipy.integrate import odeint
from lmfit import minimize, Parameters, report_fit
from geneticalgorithm import geneticalgorithm as ga

import matplotlib.pyplot as plt
from utils import ObsEnum, StateEnum, Model, residuals_error, load_data

class Sarah1(Model):

    def __init__(self, observations, N):

        self._N = N
        nb_observations = observations.shape[0]

        self._observations = np.array(observations)
        self._ydata = observations[np.ix_(range(nb_observations),
                                          [ObsEnum.POSITIVE.value,
                                           ObsEnum.HOSPITALIZED.value,
                                           ObsEnum.CRITICAL.value])]

        # Je n'ai pas trouvé de conditions initiales pour E0 et I0 qui m'évite le message:
        # Warning: uncertainties could not be estimated
        E0 = 10
        I0 = 5
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

        # Compute tau = cumulative_hospitalizations / cumulative_positives
        cumulative_positives = np.cumsum(self._observations[:, ObsEnum.POSITIVE.value])
        cumulative_hospitalizations = self._observations[:, ObsEnum.CUMULATIVE_HOSPITALIZATIONS.value]
        # There is a difference of 1 between the indexes because to have the possibility
        # to go from one state to another, at least 1 day must pass. We can not go from
        # infected to hospitalized the same day
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
        # We want the probability of exiting in 1 day
        # the incubation period to go to the infected state
        # such that the probability distribution is uniformly distributed
        # between day 1 and day 14 (= min_symptomatic_time (4) + max_symptomatic_time (10)).
        # With such a probability distribution, from day 4 to day 10, we will
        # have 7 days out of the 14 to recover and be in the righ incubation time,
        # ie: 50% of chance to recover during the right time period

        # "best-case": if people recover all in min_symptomatic_time
        gamma1_max = 1 / min_symptomatic_time
        # "worst-case": if people recover all in max_symptomatic_time
        gamma1_min = 1 / max_symptomatic_time
        # "avg-case"
        gamma1_0 = (gamma1_min + gamma1_max) / 2

        gamma1_bounds = [gamma1_min, gamma1_0, gamma1_max]

        # R_out_HC = gamma2 * H + gamma3 * C
        # Since everyone finally recovers in our model (no dead),
        # hospitalized and critical people will recover.
        # We just don't know in how much time they will recover ?
        # Thus, it is difficult to estimate gamma2 and gamma3 currently
        gamma2_0 = 0.2 # arbitrary choice
        gamma3_0 = 0.2 # arbitrary choice
        gamma2_bounds = [0.02, gamma2_0, 1]
        gamma3_bounds = [0.02, gamma3_0, 1]


        # The reproduction number R0 is the number of people
        # each infected person will infect during the time he is infectious
        # R0 = beta * infectious_time
        # We can make the assumption that once an individual becomes symptomatic,
        # he will not infect anyone else anymore as he will isolate himself under
        # the doctor's recommendations. Under this assumption, infectious_time is
        # included in [min_incubation_time, max_incubation_time]
        # TO DISCUSS: For a SEIR model: a more complex formula of R0
        # found at https://en.wikipedia.org/wiki/Basic_reproduction_number
        # could be used later
        R0_min = 1 # or else the virus is not growing exponentially
        #R0_max = 18 # the most virulent disease of all time: measles
        R0_max = 2.8 # the most virulent influenza pandemic
        # and we were told that covid-20 looked similar to influenza
        R0 = (R0_min + R0_max) / 2
        infectious_time = (min_incubation_time + max_incubation_time) / 2
        beta_0 = R0 / infectious_time
        beta_min = R0_min / max_incubation_time
        beta_max = R0_max / min_incubation_time
        beta_bounds = [beta_min, beta_0, beta_max]

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


        # In our model, 100% of the exposed people will eventually become infected
        # Therefore, cumulative_positives(t) = cumulative_exposed(t - incubation_time)
        # Or alternatively, cumulative_positives(t + incubation_time) = cumulative_exposed(t)
        # sigma = cumulative_positives / cumulative_exposed
        incubation_time = math.ceil((min_incubation_time + max_incubation_time) / 2)
        cumulative_exposed = cumulative_positives[-1]
        sigma_0 = cumulative_positives[-incubation_time-1] / cumulative_exposed
        # In the case of an exponential GROWTH, we have
        # "worst-case": the incubation time is very short, the number of exposed people
        # will move quicker to the infected state
        sigma_max = cumulative_positives[-min_incubation_time-1] / cumulative_exposed
        sigma_max = max(0, min(sigma_max, 1))
        # "best-case": the incubation time is very long, the number of exposed people
        # will move slower to the infected state
        sigma_min = cumulative_positives[-max_incubation_time-1] / cumulative_exposed
        sigma_min = min(max(sigma_min, 0), 1)
        sigma_bounds = [sigma_min, sigma_0, sigma_max]

        bounds = [gamma1_bounds, gamma2_bounds, gamma3_bounds, beta_bounds, tau_bounds, delta_bounds, sigma_bounds]
        param_names = ['gamma1', 'gamma2', 'gamma3', 'beta', 'tau', 'delta', 'sigma']
        params = Parameters()

        for param_str, param_bounds in zip(param_names, bounds):
            params.add(param_str, value=param_bounds[1], min=param_bounds[0], max=param_bounds[2])

        return params

    def fit_parameters(self, error_func):

        params = self.get_initial_parameters()
        # lmfit's minimize just use
        result = minimize(self._plumb_lmfit,
                          params,
                          args=(len(self._observations),
                                error_func),
                          method='leastsq')

        report_fit(result)

        self._fit_params = result.params

    def predict(self, days):
        res = self._predict(self._initial_conditions, days, self._fit_params)

        return res

    def _predict(self, initial_conditions, days, params):
        tspan = np.arange(0, days, 1)

        S0, E0, I0, H0, C0, R0 = initial_conditions
        gamma1 = params['gamma1']
        gamma2 = params['gamma2']
        gamma3 = params['gamma3']
        beta = params['beta']
        tau = params['tau']
        delta = params['delta']
        sigma = params['sigma']

        # Integrate ODE over time span [0,days]
        res = odeint(self._model, [S0, E0, I0, H0, C0, R0],
                     tspan, args=(gamma1, gamma2, gamma3, beta, tau, delta, sigma))
        return res

    def _model(self, y, t, gamma1, gamma2, gamma3, beta, tau, delta, sigma):
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

        # Positif = la populatin testée SEULEMENT le jour j(donc une toute petite partie de la population, par ex. on a testé 32 personnes le jour J)

        # I = Infecté = partie de la population infectée
        # le jour I (augmente au début de l'épidémie,
        # diminue à la fin)

        # Cumul = utile pour svoir quelle proportion de la
        # population, au final, a été infectée.

        dSdt = -beta * S * I / N
        dEdt = beta * S * I / N - sigma * E
        dIdt = sigma * E - gamma1 * I - tau * I
        dHdt = tau * I - gamma2 * H - delta * H
        dCdt = delta * H - gamma3 * C
        dRdt = gamma1 * I + gamma2 * H + gamma3 * C

        return [dSdt, dEdt, dIdt, dHdt, dCdt, dRdt]


    # [ObsEnum.POSITIVE.value,
    #  ObsEnum.HOSPITALIZED.value,
    #  ObsEnum.CRITICAL.value

    def _plumb_lmfit(self, params, days, error_func):
        res = self._predict(self._initial_conditions, days, params)

        # ydata = les observartions
        # res = les prédictions

        rselect = np.ix_(range(res.shape[0]),
                         [StateEnum.INFECTIOUS.value,
                          StateEnum.HOSPITALIZED.value,
                          StateEnum.CRITICAL.value])

        # lmfit only
        return error_func(res[rselect], self._ydata).ravel()


class Sarah1GA(Model):

    def __init__(self, observations, N):

        self._N = N
        nb_observations = observations.shape[0]

        self._observations = np.array(observations)
        self._ydata = observations[np.ix_(range(nb_observations),
                                          [ObsEnum.POSITIVE.value,
                                           ObsEnum.HOSPITALIZED.value,
                                           ObsEnum.CRITICAL.value])]

        # Je n'ai pas trouvé de conditions initiales pour E0 et I0 qui m'évite le message:
        # Warning: uncertainties could not be estimated
        E0 = 10
        I0 = 5
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

        # Compute tau = cumulative_hospitalizations / cumulative_positives
        cumulative_positives = np.cumsum(self._observations[:, ObsEnum.POSITIVE.value])
        cumulative_hospitalizations = self._observations[:, ObsEnum.CUMULATIVE_HOSPITALIZATIONS.value]
        # There is a difference of 1 between the indexes because to have the possibility
        # to go from one state to another, at least 1 day must pass. We can not go from
        # infected to hospitalized the same day
        tau_0 = cumulative_hospitalizations[-1] / cumulative_positives[-2]
        tau_bounds = [tau_0 * (1 - error_margin), tau_0, max(1, tau_0 * (1 + error_margin))]


        # We want the probability of exiting in 1 day
        # the incubation period to go to the infected state
        # such that the probability distribution is uniformly distributed
        # between day 1 and day 14 (= min_symptomatic_time (4) + max_symptomatic_time (10)).
        # With such a probability distribution, from day 4 to day 10, we will
        # have 7 days out of the 14 to recover and be in the righ incubation time,
        # ie: 50% of chance to recover during the right time period
        # "best-case": if people recover all in min_symptomatic_time
        gamma1_max = 1 / min_symptomatic_time
        # "worst-case": if people recover all in max_symptomatic_time
        gamma1_min = 1 / max_symptomatic_time
        # "avg-case"
        gamma1_0 = (gamma1_min + gamma1_max) / 2

        gamma1_bounds = [gamma1_min, gamma1_0, gamma1_max]

        # R_out_HC = gamma2 * H + gamma3 * C
        # Since everyone finally recovers in our model (no dead),
        # hospitalized and critical people will recover.
        # We just don't know in how much time they will recover ?
        # Thus, it is difficult to estimate gamma2 and gamma3 currently
        gamma2_0 = 0.2 # arbitrary choice
        gamma3_0 = 0.2 # arbitrary choice
        gamma2_bounds = [0.02, gamma2_0, 1]
        gamma3_bounds = [0.02, gamma3_0, 1]


        # The reproduction number R0 is the number of people
        # each infected person will infect during the time he is infectious
        # R0 = beta * infectious_time
        # We can make the assumption that once an individual becomes symptomatic,
        # he will not infect anyone else anymore as he will isolate himself under
        # the doctor's recommendations. Under this assumption, infectious_time is
        # included in [min_incubation_time, max_incubation_time]
        # TO DISCUSS: For a SEIR model: a more complex formula of R0
        # found at https://en.wikipedia.org/wiki/Basic_reproduction_number
        # could be used later
        R0_min = 1 # or else the virus is not growing exponentially
        #R0_max = 18 # the most virulent disease of all time: measles
        R0_max = 2.8 # the most virulent influenza pandemic
        # and we were told that covid-20 looked similar to influenza
        R0 = (R0_min + R0_max) / 2
        infectious_time = (min_incubation_time + max_incubation_time) / 2
        beta_0 = R0 / infectious_time
        beta_min = R0_min / max_incubation_time
        beta_max = R0_max / min_incubation_time
        beta_bounds = [beta_min, beta_0, beta_max]

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


        # In our model, 100% of the exposed people will eventually become infected
        # Therefore, cumulative_positives(t) = cumulative_exposed(t - incubation_time)
        # Or alternatively, cumulative_positives(t + incubation_time) = cumulative_exposed(t)
        # sigma = cumulative_positives / cumulative_exposed
        incubation_time = math.ceil((min_incubation_time + max_incubation_time) / 2)
        cumulative_exposed = cumulative_positives[-1]
        sigma_0 = cumulative_positives[-incubation_time-1] / cumulative_exposed
        # In the case of an exponential GROWTH, we have
        # "worst-case": the incubation time is very short, the number of exposed people
        # will move quicker to the infected state
        sigma_max = cumulative_positives[-min_incubation_time-1] / cumulative_exposed
        sigma_max = max(0, min(sigma_max, 1))
        # "best-case": the incubation time is very long, the number of exposed people
        # will move slower to the infected state
        sigma_min = cumulative_positives[-max_incubation_time-1] / cumulative_exposed
        sigma_min = min(max(sigma_min, 0), 1)
        sigma_bounds = [sigma_min, sigma_0, sigma_max]

        bounds = [gamma1_bounds, gamma2_bounds, gamma3_bounds, beta_bounds, tau_bounds, delta_bounds, sigma_bounds]
        param_names = ['gamma1', 'gamma2', 'gamma3', 'beta', 'tau', 'delta', 'sigma']
        params = Parameters()

        for param_str, param_bounds in zip(param_names, bounds):
            params.add(param_str, value=param_bounds[1], min=param_bounds[0], max=param_bounds[2])

        return params

    def fit_parameters(self, error_func):

        params = self.get_initial_parameters()

        print(params.items())

        bounds = np.array([(p.min, p.max) for p_name, p in params.items()])
        self._error_func = error_func

        gamodel = ga(function=self._plumb,
                     dimension=len(bounds),
                     variable_type='real',
                     variable_boundaries=bounds)
        gamodel.run()

        self._fit_params = self._params_array_to_dict(
            gamodel.output_dict['variable'])

    def _params_array_to_dict(self, params):
        return dict(
            zip(['gamma1', 'gamma2', 'gamma3', 'beta', 'tau', 'delta', 'sigma'],
                params))

    def predict(self, days):
        res = self._predict(self._initial_conditions, days, self._fit_params)

        return res

    def _predict(self, initial_conditions, days, params):
        tspan = np.arange(0, days, 1)

        S0, E0, I0, H0, C0, R0 = initial_conditions
        gamma1 = params['gamma1']
        gamma2 = params['gamma2']
        gamma3 = params['gamma3']
        beta = params['beta']
        tau = params['tau']
        delta = params['delta']
        sigma = params['sigma']

        # Integrate ODE over time span [0,days]
        res = odeint(self._model, [S0, E0, I0, H0, C0, R0],
                     tspan, args=(gamma1, gamma2, gamma3, beta, tau, delta, sigma))
        return res

    def _model(self, y, t, gamma1, gamma2, gamma3, beta, tau, delta, sigma):
        S, E, I, H, C, R = y

        N = self._N

        dSdt = -beta * S * I / N
        dEdt = beta * S * I / N - sigma * E
        dIdt = sigma * E - gamma1 * I - tau * I
        dHdt = tau * I - gamma2 * H - delta * H
        dCdt = delta * H - gamma3 * C
        dRdt = gamma1 * I + gamma2 * H + gamma3 * C

        return [dSdt, dEdt, dIdt, dHdt, dCdt, dRdt]

    def _plumb(self, params):

        days = len(self._observations)

        # Sarah's function prefers params as a dictionary
        # so we convert.
        params_as_dict = self._params_array_to_dict(params)

        res = self._predict(self._initial_conditions, days, params_as_dict)

        rselect = np.ix_(range(res.shape[0]),
                         [StateEnum.INFECTIOUS.value,
                          StateEnum.HOSPITALIZED.value,
                          StateEnum.CRITICAL.value])

        # The genetic algorithm uses an error represented
        # as a single float => so we can't use a vector.
        residuals = res[rselect] - self._ydata
        least_squares = np.sum(residuals*residuals)
        return least_squares


if __name__ == "__main__":
    head, observations, rows = load_data()
    rows = np.array(rows)

    ms = Sarah1(rows, 1000000)
    ms.fit_parameters(residuals_error)
    sres = ms.predict(100)

    plt.figure()
    for t, u in zip([StateEnum.INFECTIOUS, StateEnum.HOSPITALIZED], [ObsEnum.POSITIVE, ObsEnum.HOSPITALIZED]):
        plt.plot(sres[:, t.value], label=f"{t} (model)")
        plt.plot(rows[:, u.value], label=f"{u} (real)")

    plt.title('LM fit')
    plt.xlabel('Days')
    plt.ylabel('Individuals')
    plt.ylim(0, 500)
    plt.legend()
    plt.show()

    exit()

    # -------------------------------------------------------------

    ms = Sarah1GA(rows, 1000000)
    ms.fit_parameters(residuals_error)
    sres = ms.predict(30)

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
