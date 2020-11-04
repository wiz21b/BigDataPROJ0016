import numpy as np
from bounded_models import Sarah1
from utils import ObsEnum, StateEnum, Model, residuals_error, load_data, residual_sum_of_squares
import matplotlib.pyplot as plt

def r(p_name):
    """ Quickly create a range of value for the
    parameter named p_name.
    """

    p = params[p_name]
    delta = (p.max-p.min) / 10
    return np.arange(p.min, p.max, delta)



head, observations, rows = load_data()
rows = np.array(rows)
model = Sarah1(rows, 1000000)

params = model.get_initial_parameters()

xs, ys, zs, errs = [], [], [], []

# We iterate over 3 parameters because we have only
# three dimensions to drwaw from :-( The other
# parameters are left as their default initial values
# as computed by get_initial_values.

P1 = "beta"
P2 = "sigma"
P3 = "gamma1"



for beta in r(P1):
    params[P1].set(value=beta)
    for sigma in r(P2):
        params[P2].set(value=sigma)
        for gamma1 in r(P3):
            params[P3].set(value=gamma1)

            res = model._predict(model._initial_conditions,
                                 len(observations),
                                 params)
            rselect = np.ix_(range(res.shape[0]),
                             [StateEnum.INFECTIOUS.value,
                              StateEnum.HOSPITALIZED.value,
                              StateEnum.CRITICAL.value])
            err= residual_sum_of_squares(
                res[rselect],
                model._fittingObservations)

            xs.append(beta)
            ys.append(sigma)
            zs.append(gamma1)
            errs.append(err)


        #print(err)


# Normalize error to [0,1] interval
c = np.array( errs)
c = c - np.min(c)
c = c / (np.max(c) - np.min(c))

# Plot everything in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter( xs, ys, zs, c=c, cmap=plt.get_cmap("hot"))
ax.set_xlabel(P1)
ax.set_ylabel(P2)
ax.set_zlabel(P3)

legend1 = ax.legend(*scatter.legend_elements(num=10),
                    loc="lower left", title="Erreur")
ax.add_artist(legend1)

plt.show()
