from copy import copy
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize as opt_minimize
from lmfit import minimize, Parameters, report_fit
from geneticalgorithm import geneticalgorithm as ga

import matplotlib.pyplot as plt
from utils import ObsRow, Model, residuals_error, load_data


def mean_square_error(results, observations):
    d = results - observations
    return np.sum(np.abs(d))


class Stefan(Model):
    def __init__(self, observations, N):
        self._N = N
        self._observations = observations
        self._fit_params = None

        self._nb_observations = observations.shape[0]

        self._observations = observations[
            np.ix_(range(self._nb_observations),
                   [ObsRow.CUMULATIVE_POSITIVE.value,
                    ObsRow.CUMULATIVE_HOSPITALIZATIONS.value,
                    ObsRow.CRITICAL.value])]

    def fit_parameters(self, error_func):
        self._error_func = error_func
        z = opt_minimize(
            self._pfunc,
            [1, 1, 1, 1, 0.2, 0.2],
            bounds=[[0,200],[1,20],[0.001,5],[0.001,10],[0.001,10],[0.01,10]],
            options={'disp': True})

        i_start, h_start, alpha, beta, gamma1, gamma2 = z.x
        c_start = 0

        self._fit_params = [i_start, h_start, alpha, beta,
                            gamma1, gamma2, c_start]

    def predict(self, days):
        i_start, h_start, alpha, beta, gamma1, gamma2, c_start = self._fit_params

        print(self._fit_params)

        res = self._predict(
            [self._N, i_start, h_start, c_start],
            [alpha, beta, gamma1, gamma2],
            days)

        return [ObsRow.CUMULATIVE_POSITIVE,
                ObsRow.CUMULATIVE_HOSPITALIZATIONS,
                ObsRow.CRITICAL], res[:, 1:4]

    def _predict(self, initial_conds, params, days):
        INITIAL_POP = initial_conds[0]

        values = copy(initial_conds)
        days_data = [values]

        alpha, beta, gamma1, gamma2 = params

        for day in range(days-1):

            # s = suscpetible (ie not yet infected)
            # i = infected
            # h = hospitalised
            # c = critical

            s, i, h, c = values

            #print(f"{alpha} {i} {s}")
            s_to_i = alpha*i # This screws the minimizer completely : *(s/INITIAL_POP)

            print(f"{day}\t{s_to_i}, i={i}, s={s}")

            ds = - s_to_i
            di = + s_to_i - beta*i
            dh = + beta*i - gamma1*h
            dc = + gamma1*h - gamma2*h

            # if s + ds < 0:
            #     print("s too small")

            # s = max(0, s+ds)
            # i = max(0, i+di)
            # h = max(0, h+dh)
            # c = max(0, c+dc)

            s = s+ds
            i = i+di
            h = h+dh
            c = c+dc

            values = [s, i, h, c]

            days_data.append(values)

        return np.array(days_data)

    def _pfunc(self, params):
        # Plumbing function to run the optimizer

        i_start, h_start, alpha, beta, gamma1, gamma2 = params
        c_start = 0

        v = self._predict(
                [self._N, i_start, h_start, c_start],
                [alpha, beta, gamma1, gamma2],
                self._nb_observations)

        # We have no basis to compare S(suspect) to...

        e = self._error_func(v[:, 1:4], self._observations)
        #e = self._error_func(v[:, 1], self._observations[:, 0])

        return e



class Sarah1(Model):

    def __init__(self, observations, N):

        self._N = N
        nb_observations = observations.shape[0]

        self._observations = observations[
            np.ix_(range(nb_observations),
                   [ObsRow.CUMULATIVE_POSITIVE.value,
                    ObsRow.CUMULATIVE_HOSPITALIZATIONS.value,
                    ObsRow.CRITICAL.value])]

        I0 = 1 # self._observations[0][0]
        H0 = self._observations[0][1]
        C0 = self._observations[0][2]
        R0 = 0
        S0 = self._N - I0 - R0 - H0 - C0

        self._initial_conditions = [S0, I0, H0, C0, R0]

        print(self._initial_conditions)
        self._fit_params = None

    def fit_parameters(self, error_func):
        gamma1 = (1/10 + 1/5)/2
        gamma2 = 0.02
        gamma3 = 0.02
        beta = 0.14

        num_cumulative_positiv = np.sum(self._observations[:, 0])

        print(f"num_cumulative_positiv {num_cumulative_positiv}")
        print(self._observations[:, 0])

        tau = 28/num_cumulative_positiv

        delta = 5/18

        params = Parameters()
        params.add('gamma1', value=gamma1, min=1/10, max=1/5)
        params.add('gamma2', value=gamma2, min=0, max=5)
        params.add('gamma3', value=gamma3, min=0, max=5)
        params.add('beta', value=beta, min=0.01, max=0.5)
        params.add('tau', value=tau, min=tau*0.8, max=tau*1.2)
        params.add('delta', value=delta, min=0.6*delta, max=1.4*delta)

        result = minimize(self._plumb,
                          params,
                          args=(len(self._observations), error_func),
                          method='leastsq')

        report_fit(result)

        self._fit_params = result.params

    def predict(self, days):
        res = self._predict(self._initial_conditions, days, self._fit_params)

        return [ObsRow.SUSPECT,
                ObsRow.CUMULATIVE_POSITIVE,
                ObsRow.CUMULATIVE_HOSPITALIZATIONS,
                ObsRow.CRITICAL,
                ObsRow.RECOVERED], res

    def _predict(self, initial_conditions, days, params):
        tspan = np.arange(0, days, 1)

        S0, I0, H0, C0, R0 = initial_conditions
        gamma1 = params['gamma1']
        gamma2 = params['gamma2']
        gamma3 = params['gamma3']
        beta = params['beta']
        tau = params['tau']
        delta = params['delta']

        # Integrate ODE over time span [0,days]
        res = odeint(self._SIHCR_model, [S0, I0, H0, C0, R0],
                     tspan, args=(gamma1, gamma2, gamma3, beta, tau, delta))
        return res

    def _SIHCR_model(self, y, t, gamma1, gamma2, gamma3, beta, tau, delta):
        S, I, H, C, R = y

        # Modèle SIHCR
        # Liens et paramètres:
        # S -> I : - (beta * S * I )/ N
        # I <- S : (beta * S * I )/ N
        # I -> R : - gamma1 * I |&| I -> H : - tau * I
        # H <- I : tau * I
        # H -> R : - gamma2 * H |&| H -> C : - delta * H
        # C <- H : delta * H
        # C -> R : - gamma3 * C
        # R <- I : gamma1 * R |&| R <- H : gamma2 * H |&| R <- C : gamma3 * C

        # Interprétations selon NOTRE modèle:

        # Notre modèle ne fait que des liens simples ( S -> I -> H -> C ).
        # Dans celui-ci, la somme des "num_positive" nous donne le total des
        # gens qui ont été infectés par le covid, "num_cumulative_positive".

        # Si on calcule:
        # "num_cumulative_hospitalizations / num_cumulative_positive"
        # Ceci nous donne la proportions de gens infectés qui ont étés admis
        # à l'hopital. On a donc une première approximation du paramètre "tau"
        # qui représente le taux de transition des personnes infectées vèrs
        # l'hopital.

        # Puisque notre modèle ne considère pas que le gens meurent du virus
        # on peut connaitre à tout temps le nombre total de personnes qui se
        # sont rétablies en étant soit à l'hopital soit en soins intesifs.
        # Appelons ces persones "R_survivants".
        # R_survivants = num_cumulative_hospitalizations - num_hospitalised
        #                                                - num_critical


        # À chaque temps t on sait dans déterminer combien de personnes sortent
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

        N = self._N

        dSdt = -(beta * I) * (S / N)
        dIdt = beta * S * I / N - gamma1 * I - tau * I
        dHdt = tau * I - gamma2 * H - delta * H
        dCdt = delta * H - gamma3 * C
        dRdt = gamma1 * I + gamma2 * H + gamma3 * C

        # Si jamais on voulait considérer un système comprenant un temps
        # d'incubation.

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
        # R <- I : gamma1 * R |&| R <- H : gamma2 * H |&| R <- C : gamma3 *

        # Le système d'équations serait le suivant:
        """
        dSdt = -beta * S * I / N
        dEdt = beta * S * I / N - sigma * E
        dIdt = sigma * E - gamma1 * I - tau * I
        dHdt = tau * I - gamma2 * H - delta * H
        dCdt = delta * H - gamma3 * C
        dRdt = gamma1 * I + gamma2 * H + gamma3 * C
        """
        return [dSdt, dIdt, dHdt, dCdt, dRdt]

    def _plumb(self, params, days, error_func):

        # The plumb function is here to connect two worlds.
        # The world of our own model and the world
        # of the optimizer (lmfit, scipy, etc.).
        # There should be one plumb function per optimizer.
        # The plumb function translates calls from the
        # optimizer to calls to our own functions.
        # It adds a level of redirection to be able to
        # change the optimizer without having to rewrite
        # our model.

        res = self._predict(self._initial_conditions, days, params)

        rselect = np.ix_(range(res.shape[0]), [1, 2, 3])

        return error_func(res[rselect],
                          self._observations).ravel()



class Sarah1GA(Model):

    def __init__(self, observations, N):

        self._N = N
        nb_observations = observations.shape[0]

        self._observations = observations[
            np.ix_(range(nb_observations),
                   [ObsRow.CUMULATIVE_POSITIVE.value,
                    ObsRow.CUMULATIVE_HOSPITALIZATIONS.value,
                    ObsRow.CRITICAL.value])]

        I0 = 1 # self._observations[0][0]
        H0 = self._observations[0][1]
        C0 = self._observations[0][2]
        R0 = 0
        S0 = self._N - I0 - R0 - H0 - C0

        self._initial_conditions = [S0, I0, H0, C0, R0]

        print(self._initial_conditions)
        self._fit_params = None

    def fit_parameters(self, error_func):
        gamma1 = (1/10 + 1/5)/2
        gamma2 = 0.02
        gamma3 = 0.02
        beta = 0.14

        num_cumulative_positiv = np.sum(self._observations[:, 0])

        print(f"num_cumulative_positiv {num_cumulative_positiv}")
        print(self._observations[:, 0])

        tau = 28/num_cumulative_positiv

        delta = 5/18

        params = Parameters()
        params.add('gamma1', value=gamma1, min=1/10, max=1/5)
        params.add('gamma2', value=gamma2, min=0, max=5)
        params.add('gamma3', value=gamma3, min=0, max=5)
        params.add('beta', value=beta, min=0.01, max=0.5)
        params.add('tau', value=tau, min=tau*0.8, max=tau*1.2)
        params.add('delta', value=delta, min=0.6*delta, max=1.4*delta)

        # result = minimize(self._plumb,
        #                   params,
        #                   args=(len(self._observations), error_func),
        #                   method='leastsq')

        # report_fit(result)

        # self._fit_params = result.params

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
            zip(['gamma1', 'gamma2', 'gamma3', 'beta', 'tau', 'delta'],
                params))

    def predict(self, days):
        res = self._predict(self._initial_conditions, days, self._fit_params)

        return [ObsRow.SUSPECT,
                ObsRow.CUMULATIVE_POSITIVE,
                ObsRow.CUMULATIVE_HOSPITALIZATIONS,
                ObsRow.CRITICAL,
                ObsRow.RECOVERED], res

    def _predict(self, initial_conditions, days, params):
        tspan = np.arange(0, days, 1)

        S0, I0, H0, C0, R0 = initial_conditions
        gamma1 = params['gamma1']
        gamma2 = params['gamma2']
        gamma3 = params['gamma3']
        beta = params['beta']
        tau = params['tau']
        delta = params['delta']

        # Integrate ODE over time span [0,days]
        res = odeint(self._SIHCR_model, [S0, I0, H0, C0, R0],
                     tspan, args=(gamma1, gamma2, gamma3, beta, tau, delta))
        return res

    def _SIHCR_model(self, y, t, gamma1, gamma2, gamma3, beta, tau, delta):
        S, I, H, C, R = y

        N = self._N

        dSdt = -(beta * I) * (S / N)
        dIdt = beta * S * I / N - gamma1 * I - tau * I
        dHdt = tau * I - gamma2 * H - delta * H
        dCdt = delta * H - gamma3 * C
        dRdt = gamma1 * I + gamma2 * H + gamma3 * C

        return [dSdt, dIdt, dHdt, dCdt, dRdt]

    def _plumb(self, params):

        days = len(self._observations)

        # Sarah's function prefers params as a dictionary
        # so we convert.
        params_as_dict = self._params_array_to_dict(params)

        res = self._predict(self._initial_conditions, days, params_as_dict)

        rselect = np.ix_(range(res.shape[0]), [1, 2, 3])

        # The genetic algorithm uses an error represented
        # as a single float => so we can't use a vector.
        residuals = res[rselect] - self._observations
        least_squares = np.sum(residuals*residuals)
        return least_squares


if __name__ == "__main__":
    head, observations, rows = load_data()
    rows = np.array(rows)

    # m = Stefan(rows, 10000)
    # m.fit_parameters(mean_square_error)
    # res_ndx, res = m.predict(20)
    # res_dict = dict(zip(res_ndx, range(len(res_ndx))))
    # for t in [ObsRow.CUMULATIVE_POSITIVE, ObsRow.CUMULATIVE_HOSPITALIZATIONS]:
    #     plt.plot(res[:, res_dict[t]], label=f"{t} (model)")
    #     plt.plot(rows[:, t.value], label=f"{t} (real)")

    # plt.xlabel('Days')
    # plt.ylabel('Individuals')
    # plt.legend()
    # plt.show()

    # -------------------------------------------------------------

    ms = Sarah1(rows, 1000000)
    ms.fit_parameters(residuals_error)
    sres_ndx, sres = ms.predict(30)

    res_dict = dict(zip(sres_ndx, range(len(sres_ndx))))

    plt.figure()
    for t in [ObsRow.CUMULATIVE_POSITIVE,
              ObsRow.CUMULATIVE_HOSPITALIZATIONS]:
        plt.plot(sres[:, res_dict[t]], '--', color=ObsRow.color(t), label=f"{t} (model)")
        plt.plot(rows[:, t.value], color=ObsRow.color(t), label=f"{t} (real)")

    plt.title('LM fit')
    plt.xlabel('Days')
    plt.ylabel('Individuals')
    plt.ylim(0, 1000)
    plt.legend()
    plt.show()

    # -------------------------------------------------------------

    ms = Sarah1GA(rows, 1000000)
    ms.fit_parameters(residuals_error)
    sres_ndx, sres = ms.predict(30)

    res_dict = dict(zip(sres_ndx, range(len(sres_ndx))))

    plt.figure()
    for t in [ObsRow.CUMULATIVE_POSITIVE,
              ObsRow.CUMULATIVE_HOSPITALIZATIONS]:
        plt.plot(sres[:, res_dict[t]], '--', color=ObsRow.color(t), label=f"{t} (model)")
        plt.plot(rows[:, t.value], color=ObsRow.color(t), label=f"{t} (real)")

    plt.title('GA fit')
    plt.xlabel('Days')
    plt.ylabel('Individuals')
    plt.ylim(0, 1000)
    plt.legend()
    plt.show()
