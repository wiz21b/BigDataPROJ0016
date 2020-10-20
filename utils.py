import csv
from collections import namedtuple
import urllib.request

def load_data():
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

    return head, observations, rows
