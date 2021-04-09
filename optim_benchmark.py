import os
import tempfile
from datetime import datetime
import random
import numpy as np

from lmfit import Parameters
from scipy.optimize import basinhopping, brute, shgo, dual_annealing, differential_evolution

from utils import ObsEnum, StateEnum, ObsFitEnum, StateFitEnum, Model, residuals_error, load_data, residual_sum_of_squares, log_residual_sum_of_squares, COLORS_DICT, load_model_data
PARAM_LIST = ['gamma1', 'gamma2', 'gamma3', 'gamma4', 'beta', 'tau', 'delta', 'sigma','rho','theta','mu','eta']

from SEIR_HCD import SEIR_HCD

# For repeatiblity (but less random numbers)

random.seed(1000)
np.random.seed(1000)
np.seterr(all='raise')


class ModelOptimizerTestBench(SEIR_HCD):

    def __init__(self):
        super(ModelOptimizerTestBench,self).__init__(stocha=False)

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
                       step_size):


        self._data = data
        self._dataLength = data.shape[0]

        self._track = []
        bounds = np.array([(p.min, p.max) for p_name, p in parameters.items()])

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


if __name__ == "__main__":
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
