# A hack to import utils frop upper directory
import sys
sys.path.insert(0,'..')

import math
import os
import tempfile
from datetime import datetime,date
import random

import numpy as np

import matplotlib.pyplot as plt
from scipy.stats import binom
from lmfit import Parameters
from scipy.optimize import basinhopping, brute, shgo, dual_annealing, differential_evolution

#from utils import ObsEnum, StateEnum, ObsFitEnum, StateFitEnum, Model, residuals_error, load_data, residual_sum_of_squares, log_residual_sum_of_squares, COLORS_DICT, load_model_data
from utils import Model, ObsEnum, StateEnum, ObsFitEnum, StateFitEnum, load_model_data, residual_sum_of_squares, periods_in_days, plot_periods

PARAM_LIST = ['gamma1', 'gamma2', 'gamma3', 'gamma4', 'beta', 'tau', 'delta', 'sigma','rho','theta','mu','eta']

from SEIR_HCD import SEIR_HCD

# For repeatiblity (but less random numbers)

random.seed(1000)
np.random.seed(1000)
np.seterr(all='raise')


class ModelOptimizerTestBench(SEIR_HCD):

    def __init__(self):
        super(ModelOptimizerTestBench,self).__init__(stocha=False)


    """Partie déterminieste à faire"""
    def plumb(self, parameters):
        # TO DO: Partie déterministe!
        days = self._fittingPeriod[1]-self._fittingPeriod[0]
        params = dict(zip(self._paramNames, parameters))

        res = self.predict(end = days, parameters = params)

        assert not self._stochastic, "this benchmark works only with non staochastic stuff"

        # not stochastic => deterministic

        # Robin's code
        if False:
            res = self.predict(end = days, parameters = params) # CASES_MUNI_CUM, CASES_AGESEX, CASES_MUNI, HOSP, MORT, TESTS, VACC
            residuals = res[self._fittingPeriod[0]:self._fittingPeriod[1],StateEnum.HOSPITALIZED.value] - self._data[self._fittingPeriod[0]:self._fittingPeriod[1],ObsEnum.NUM_HOSPITALIZED.value]
            least_squares = np.sum(residuals*residuals)
            return least_squares

        # print(f"Fiting period : {self._fittingPeriod[0]}, {self._fittingPeriod[1]}")
        # print("np.cumsum(self._data[:,ObsEnum.DHDT.value])")
        # print(np.cumsum(self._data[:,ObsEnum.DHDT.value]))
        # print("self._data[:,ObsEnum.NUM_HOSPITALIZED.value]")
        # print(self._data[:,ObsEnum.NUM_HOSPITALIZED.value])
        # exit()

        lhs = dict()
        experiments = self.predict(end = days, parameters = params) # CASES_MUNI_CUM, CASES_AGESEX, CASES_MUNI, HOSP, MORT, TESTS, VACC
        # 1) People moving from SYMPTOMATIQUE to HOSPITALIZED
        # dHdt = tau * SP - delta * H - gamma2 * H

        # 2) People moving from ASYMPTOMATIQUE to TESTED (mu param)
        # dSPdt = sigma * A - tau * SP - gamma1 * SP
        # dSPIndt = sigma * A
        # DTESTEDDT = dSPIndt * mu

        # 3) People moving from TESTED to TESTED_POSITIVE (eta param)
        # dSPdt = sigma * A - tau * SP - gamma1 * SP
        # dSPIndt = sigma * A
        # DTESTEDDT = dSPIndt * mu
        # DTESTEDPOSDT = DTESTEDDT * eta

        # StateEnum.HOSPITALIZED, ObsEnum.NUM_HOSPITALIZED

        for state, obs, param in [(StateEnum.SYMPTOMATIQUE, ObsEnum.DHDT, params['Tau'])]:
            # ,
            #                       (StateEnum.DSPDT, ObsEnum.NUM_TESTED, params['Mu']),
            #                       (StateEnum.DTESTEDDT, ObsEnum.NUM_POSITIVE, params['Eta'])]:

            # For cache optimisation, we discretize the parameter
            # Therefore w emake the assumption that the gradient
            # around the parameter is smooth (else rounding it
            # we'll skip maybe-important holes in the grad)

            p100 = round(param * 1000)
            param = p100 / 100

            log_likelihood = 0
            for day in np.arange(self._fittingPeriod[0], self._fittingPeriod[1]):
                # Take all the values of experiments on a given day day_ndx
                # for a given measurement (state.value)

                # predicted_total = le nombre total de personne dans
                #    un compartiment donné (x)
                # observed_leaving = nombre d'individus qui quittent
                #    le compartiment (dx/dt)

                observed_leaving = self._data[day][obs.value]
                predicted_total = experiments[day, state.value]

                # Déterminer P(observed_leaving | predicted_total, parameter)
                # par un calcul de binomiale.

                # Cache optimisation
                BINOM_LIMIT = 100
                if predicted_total >= BINOM_LIMIT:
                    observed_leaving = BINOM_LIMIT*observed_leaving/predicted_total
                    predicted_total = BINOM_LIMIT

                observed_leaving = round(max(1, observed_leaving))
                predicted_total = max(0, predicted_total)

                t = (observed_leaving, predicted_total, p100)
                if math.isnan(predicted_total):
                    log_likelihood += -999
                elif observed_leaving > predicted_total:
                    # La binomiale n'est pas définie si
                    # observed_leaving > predicted_total
                    # C'est à dire si le nombre de succès
                    # est supérieur au nombre de trials.

                    # On note cette situation comme totalement
                    # improbable mais on garde l'expérience
                    # globale (tous les jours) car ce n'est pas
                    # parce-que un jour est mauvais que tout le
                    # reste l'est.

                    log_likelihood += -999
                elif t in self._pmf_cache:
                    log_likelihood += self._pmf_cache[t]
                    self._cache_hits += 1
                else:

                    try:
                        # Optimisation note : I've tried to replace this with a
                        # numpy version as proposed here :
                        # https://gist.github.com/slowkow/11504548
                        # but it's slower (about 10%). Didn't do the C version though.

                        likelihood_observation = binom.pmf(observed_leaving, predicted_total, param)

                        if likelihood_observation > 1E-30:
                            log_bin = np.log(likelihood_observation)
                        else:
                            log_bin = -999

                    except FloatingPointError as exception:
                        print(f"{exception}, observed_leaving={observed_leaving}, predicted_total={predicted_total}, param={param:f}")
                        # traceback.print_exc()
                        log_bin = -999

                    self._pmf_cache[t] = log_bin
                    # if len(self._pmf_cache) % 1000 == 1:
                    #     print(f"Cache hit/miss ratio : {self._cache_hits / (self._cache_hits+len(self._pmf_cache)):.2f}")

                    log_likelihood += log_bin
            lhs[obs] = log_likelihood

        # print(-sum(lhs.values()))
        return -sum(lhs.values())


    def fit_parameters_brute(self,
                             data: np.ndarray,
                             parameters):

        self._data = data
        bounds = np.array([(p.min, p.max) for p_name, p in parameters.items()])

        brute(self.plumb,
              ranges=bounds)

        # Fun fact ! Throws error :
        # numpy.core._exceptions.MemoryError: Unable to allocate
        # 349. PiB for an array with shape (12, 20, 20, 20, 20, 20,
        # 20, 20, 20, 20, 20, 20, 20) and data type float64

    def fit_parameters_shgo(self, data: np.ndarray, parameters):
        self._data = data
        bounds = np.array([(p.min, p.max) for p_name, p in parameters.items()])
        print(f"shgo : Start {datetime.now()}")
        res = shgo(self.plumb, bounds=bounds)
        print(res)
        print(f"End {datetime.now()}")

    def fit_parameters_dual_annealing(self, data: np.ndarray, parameters):
        self._data = data
        bounds = np.array([(p.min, p.max) for p_name, p in parameters.items()])
        start = datetime.now()
        res = dual_annealing(self.plumb, bounds=bounds,
                             maxfun=1000)
        print(res)
        print(f"Duration {datetime.now() - start}")

    def fit_parameters_differential_evolution(self, data: np.ndarray, parameters):
        self._data = data
        bounds = np.array([(p.min, p.max) for p_name, p in parameters.items()])
        print(f"differential_evolution : Start {datetime.now()}")
        res = differential_evolution(self.plumb, bounds=bounds)
        print(res)
        print(f"End {datetime.now()}")

    def fit_parameters(self,
                       data: np.ndarray,
                       parameters,
                       start = 0,
                       end = None,
                       step_size=0.05):


        self._data = data
        self._dataLength = data.shape[0]

        self._track = []
        bounds = np.array([(p.min, p.max) for p_name, p in parameters.items()])

        if not(end):
            self._fittingPeriod = [start, len(self._data)]
        else:
            self._fittingPeriod = [start, end]

        for p_name, p in parameters.items():
            print( "{:10s} [{:.2f} - {:.2f}] = {:.2f}".format(p_name,p.min, p.max, p.value))


        class MyBounds(object):
            def __init__(self, params):
                #bounds = np.array([(p.min, p.max) for p_name, p in params.items()])

                self.xmin = np.array([p.min for p_name, p in params.items()])
                self.xmax = np.array([p.max for p_name, p in params.items()])
                # print("MyBounds")
                # print(self.xmin)
                # print(self.xmax)

            def __call__(self, **kwargs):
                x = kwargs["x_new"]
                tmax = bool(np.all(x <= self.xmax))
                tmin = bool(np.all(x >= self.xmin))
                return tmax and tmin


        # Python 3.7 : Dictionary order is guaranteed to be insertion order.
        x0 = [p.value for p_name, p in parameters.items()]
        hopin_bounds = MyBounds(parameters)
        minimizer_kwargs = { "method": "L-BFGS-B",
                             "bounds": bounds}
        res = basinhopping(self.plumb,
                           x0, minimizer_kwargs=minimizer_kwargs,
                           stepsize=step_size, accept_test=hopin_bounds)

        # res = scipy_minimize(self._plumb_scipy,
        #                      x0=x0,
        #                      method='L-BFGS-B',
        #                      bounds=bounds,
        #                      args=(error_func,),
        #                      callback=callbackF)


        print(res.x)
        self.fit_params = [v for v in res.x]
        self.fit_error = self.plumb(res.x)

        self._optimalParams = dict(zip(self._paramNames, res.x))
        self._fitted = True

        # for p_name, p in params.items():
        #     print( "{:10s} [{:.2f} - {:.2f}] : {:.2f}".format(p_name,p.min, p.max,self._fit_params[p_name]))



    # def _params_array_to_dict(self, params):
    #     return dict(
    #         zip(PARAM_LIST,
    #             params))


def randomize_initial_parameters(params):
    for name, p in params.items():
        p.set(value=random.uniform(p.min, p.max))
    return params


def team_training():
    global periods_in_days

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

    ms = ModelOptimizerTestBench()

    N = 11492641 # population belge en 2020
    E0 = 80000
    A0 = 14544
    SP0 = 9686
    H0 = rows[periods_in_days[0][0]][ObsEnum.NUM_HOSPITALIZED.value]
    C0 = rows[periods_in_days[0][0]][ObsEnum.NUM_CRITICAL.value]
    R0 = np.sum(rows[:periods_in_days[0][0], ObsEnum.RSURVIVOR.value]) # = 0
    F0 = rows[periods_in_days[0][0]][ObsEnum.NUM_FATALITIES.value]
    S0 = N - E0 - A0 - SP0 - H0 - C0 - R0 - F0

    IC = [S0, E0, A0, SP0, H0, C0, F0, R0]
    print(IC)
    ms.set_IC(conditions = IC)
    initial_params = ms.get_initial_parameters()

    sres = np.array([])
    for ndx_period, period in enumerate(periods_in_days):
        print(f"Period {ndx_period}/{len(periods_in_days)}: [{period[0]}, {period[1]}]")
        # optimizer='GLOBAL',
        ms.fit_parameters(rows[period[0]:period[1], :], initial_params) #,optimizer='GLOBAL',  randomPick = False, picks = 1000)

        sres_temp = ms.predict()
        if sres_temp.any():
            ms.set_IC(conditions = sres_temp[-1, 0:8])
            if not np.any(sres):
                sres = sres_temp[:13,:] * 0 #solution 1, artificielement mettre des 0 pour les X premier jours, où plus propre, mettre IC 13 fois à voir.
                sres = np.concatenate((sres, sres_temp)) # fait partie de solution 1
                # sres = sres_temp
            else:
                sres = np.concatenate((sres, sres_temp))

    version = 3

    """plt.figure()
    plt.title('HOSPITALIZED / PER DAY fit')
    t = StateEnum.DHDT
    plt.plot(sres[:, t.value], label = str(t) + " (model)")
    u = ObsEnum.DHDT
    plt.plot(rows[:, u.value], "--", label = str(u) + " (real)")
    #plt.savefig('img/v{}-dhdt.pdf'.format(version))
    plt.show()"""

    plt.figure()
    plt.title('Hospitalized')
    t = StateEnum.HOSPITALIZED
    plt.plot(sres[:, t.value], label = str(t) + " (model)")
    u = ObsEnum.NUM_HOSPITALIZED
    plt.plot(rows[:, u.value], "--", label = str(u) + " (real)")
    plot_periods(plt, dates)
    #plt.savefig('img/v{}-hospitalized.pdf'.format(version))
    plt.show()


if __name__ == "__main__":
    team_training()
    exit()

    observations = load_model_data() # load_data()
    rows = np.array(observations)
    days = rows.shape[0]

    DATE=datetime.now().strftime("%Y%m%d")

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

    ms = ModelOptimizerTestBench()
    ms.set_IC(conditions = IC)



    initial_params = ms.get_initial_parameters()

    print(f"Start {datetime.now()}")

    # This one ran for 6 hours without giving a result
    # ms.fit_parameters_shgo(rows, initial_params)
    ms.fit_parameters_dual_annealing(rows, initial_params)
    #ms.fit_parameters_differential_evolution(rows, initial_params)

    print(f"End {datetime.now()}")
    exit()


    print(f"Random checks : Start {datetime.now()}")
    with open(f"random{DATE}.csv","w") as fout:
        ms._data = rows
        fout.write( ";".join([f"{p.name}" for p_name, p in initial_params.items()] + ['error']))
        fout.write("\n")
        NB_EXPERIMENT = 100000
        for i in range(NB_EXPERIMENT):
            if i % 100 == 0:
                print(f"{i}/{NB_EXPERIMENT} : {datetime.now()}")
            randomize_initial_parameters(initial_params)
            loss = ms.plumb([p.value for p_name, p in initial_params.items()])
            fout.write(";".join([f"{p.value:.8f}" for p_name, p in initial_params.items()] + [str(loss)]))
            fout.write("\n")
            fout.flush()
    print(f"End {datetime.now()}")
    exit()


    with open("basin{DATE}.csv","w") as fout:
        fout.write( ";".join([f"{p.name}" for p_name, p in initial_params.items()] + ['error']))
        fout.write("\n")

        NB_EXPERIMENT=100
        for i in range(NB_EXPERIMENT):
            randomize_initial_parameters(initial_params)
            step_size = [0.01, 0.05, 0.1, 0.15, 0.2][ int((i / NB_EXPERIMENT)*5)]

            print(f"Experiment {i}/{NB_EXPERIMENT}, stepsize={step_size}, {datetime.now()}")

            ms.fit_parameters(rows, initial_params, step_size)

            r = ms.fit_params + [ms.fit_error]
            print(r)
            fout.write(";".join([f"{v:.8f}" for v in r]))
            fout.write("\n")
            fout.flush()

            np.save(
                os.path.join(tempfile.gettempdir(),
                             f"tracks_{i}.npy"),
                np.array(ms._track))
