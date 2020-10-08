import csv
import urllib.request
from collections import namedtuple
from tabulate import tabulate


observations = []

with urllib.request.urlopen("https://raw.githubusercontent.com/ADelau/proj0016-epidemic-data/main/data.csv") as fp:

    data = fp.read().decode("utf-8").split('\n')
    csvreader = csv.reader(data, delimiter=',')

    head = next(csvreader)
    DataTuple = namedtuple("DataTuple", head)

    for r in csvreader:
        if r:
            observations.append(DataTuple(*r))

print(tabulate(observations, headers=head))
