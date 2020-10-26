from copy import copy
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize as opt_minimize
from lmfit import minimize, Parameters, report_fit
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
                   [ObsRow.POSITIVE.value, ObsRow.HOSPITALIZED.value,
                    ObsRow.CRITICAL.value])]

        I0 = 5 # self._observations[0][0]
        H0 = self._observations[0][1]
        C0 = self._observations[0][2]
        R0 = 0
        S0 = self._N - I0 - R0 - H0 - C0

        self._initial_conditions = [S0, I0, H0, C0, R0]

        print(self._initial_conditions)
        self._fit_params = None

    def fit_parameters(self, error_func):
        gamma1 = 0.02
        gamma2 = 0.02
        gamma3 = 0.02
        beta = 1.14
        tau = 0.02
        delta = 0.02

        params = Parameters()
        params.add('gamma1', value=gamma1, min=0, max=5)
        params.add('gamma2', value=gamma2, min=0, max=5)
        params.add('gamma3', value=gamma3, min=0, max=5)
        params.add('beta', value=beta, min=0, max=5)
        params.add('tau', value=tau, min=0, max=5)
        params.add('delta', value=delta, min=0, max=5)

        result = minimize(self._plumb,
                          params,
                          args=(len(self._observations), error_func),
                          method='leastsq')

        report_fit(result)

        self._fit_params = result.params

    def predict(self, days):
        res = self._predict(self._initial_conditions, days, self._fit_params)

        return [ObsRow.SUSPECT,
                ObsRow.POSITIVE,
                ObsRow.HOSPITALIZED,
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

        # Vu que le modèle ne fait que des liens simples ( on sait
        # passer de S --> I et de I --> H mais pas de S --> H), on est
        # onligé de si on est infecté de passer pas la case I. Et donc
        # le cumul de num_positif nous donne le nombre totaux de
        # personnes positives au covid. Donc si on utilise cumulH /
        # cumul_num_positif, on a un ratio qui représente le nombre de
        # personnes qui ont eu le covid et qui sont passée sur un lit
        # d'hopital. ( tau )

        # nombre de gens qui sont sortis de l'hospital total
        # Rsurvivants = num_cumulative_hospitalizations -
        # num_hospitalised - num_critical - num_fatalities

        # S --> I --> H --> C --> F

        # nombre total R = cumul_num_positif -
        # num_cumulative_hospitalizations + Rsurvivants 19 qui ont ete
        # hospitalisée dont 10 qui sont toujours hospitalisée, 4 qui
        # sont en critique, 0 en fatalities donc 9 plus dans H donc on
        # a 5 qui sont sortie de l'hopital et immunisée ( les
        # Rsurvivants)

        N = self._N

        dSdt = -beta * S * I / N
        dIdt = beta * S * I / N - gamma1 * I - tau * I
        dHdt = tau * I - gamma2 * H - delta * H
        dCdt = delta * H - gamma3 * C
        dRdt = gamma1 * I + gamma2 * H + gamma3 * C
        return [dSdt, dIdt, dHdt, dCdt, dRdt]

    def _plumb(self, params, days, error_func):
        res = self._predict(self._initial_conditions, days, params)

        rselect = np.ix_(range(res.shape[0]), [1, 2, 3])

        return error_func(res[rselect],
                          self._observations).ravel()


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

    for t in [ObsRow.POSITIVE, ObsRow.HOSPITALIZED]:
        plt.plot(sres[:, res_dict[t]], label=f"{t} (model)")
        plt.plot(rows[:, t.value], label=f"{t} (real)")

    plt.xlabel('Days')
    plt.ylabel('Individuals')
    plt.legend()
    plt.show()
