import csv
import urllib.request
from collections import namedtuple
import numpy as np

# If you don't have this, run "pip install tabulate"

from tabulate import tabulate


observations = []
rows = []

with urllib.request.urlopen("https://raw.githubusercontent.com/ADelau/proj0016-epidemic-data/main/data.csv") as fp:

    data = fp.read().decode("utf-8").split('\n')
    csvreader = csv.reader(data, delimiter=',')

    head = next(csvreader)
    DataTuple = namedtuple("DataTuple", head)


    for r in csvreader:
        if r:
            observations.append(DataTuple(*r))

            rows.append( [int(x) for x in r] )

print(tabulate(observations, headers=head))

rows = np.array(rows)

from matplotlib import pyplot as plt
num_positive=1
num_hospitalised=2
num_cumulative_hospitalizations=3
num_critical=4
num_fatalities=5

plt.plot( rows[:,num_positive], label='num_positive')
plt.plot( rows[:,num_hospitalised], label='num_hospitalised')
plt.plot( rows[:,num_cumulative_hospitalizations], label='num_cumulative_hospitalizations')
plt.plot( rows[:,num_critical], label='num_critical')
plt.plot( rows[:,num_fatalities], label='num_fatalities')
plt.legend()
plt.xlabel('Days')
plt.show()
