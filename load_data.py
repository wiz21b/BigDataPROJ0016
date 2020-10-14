import csv
import urllib.request
from collections import namedtuple

import numpy as np
from matplotlib import pyplot as plt
# If you don't have this, run "pip install tabulate"
from tabulate import tabulate


observations = []
rows = []
positive_cumulated = 0

with urllib.request.urlopen("https://raw.githubusercontent.com/ADelau/proj0016-epidemic-data/main/data.csv") as fp:

    data = fp.read().decode("utf-8").split('\n')
    csvreader = csv.reader(data, delimiter=',')

    head = next(csvreader)
    DataTuple = namedtuple("DataTuple", head)

    for r in csvreader:
        if r:
            ir = [int(x) for x in r]
            t = DataTuple(*ir)
            observations.append(t)
            rows.append(ir + [positive_cumulated])

print(tabulate(observations, headers=head))

rows = np.array(rows)


NUM_POSITIVE = 1
NUM_HOSPITALIZED = 2
NUM_CUMULATIVE_HOSPITALIZATIONS = 3
NUM_CRITICAL = 4
NUM_FATALITIES = 5

plt.plot(rows[:, NUM_POSITIVE], label='num_positive')
plt.plot(rows[:, NUM_HOSPITALIZED], label='num_hospitalised')
plt.plot(rows[:, NUM_CUMULATIVE_HOSPITALIZATIONS],
         label='num_cumulative_hospitalizations')
plt.plot(rows[:, NUM_CRITICAL], label='num_critical')
plt.plot(rows[:, NUM_FATALITIES], label='num_fatalities')
plt.legend()
plt.xlabel('Days')
plt.show()
