from enum import Enum
import csv
from collections import namedtuple
import urllib.request


class ObsRow(Enum):
    # These 5 are the one provided by the teachers.
    # Do not change them

    DAYS = 0
    POSITIVE = 1
    TESTED = 2
    HOSPITALIZED = 3
    CUMULATIVE_HOSPITALIZATIONS = 4
    CRITICAL = 5
    FATALITIES = 6

    # Other data series we work on
    CUMULATIVE_POSITIVE = 7
    RECOVERED = 8
    SUSPECT = 9

    def __str__(self):
        return self.name.replace("_", " ").lower()


class Model:
    def __init__(self, observations):
        # """ observations : a numpy array. Each row of the
        # array corresponds to one type of observation. Each
        # column correspond to one (daily) observation for the
        # corresponding type.

        # The rows' numbers must match `ObsRow`.
        # """
        raise NotImplementedError

    def fit_parameters(self, error_func):
        # Call this method to perform a parameters
        # fit.

        # error_func : the function that will be
        # used to compute the error between prediction
        # and actual data.
        raise NotImplementedError

    def predict(self, days):
        # Once the parameters have been fit, call this
        # function to predict values for `days`.

        # This function must return :
        # - an array of ObsRow describing the result array rows.
        # - a result array of which the columns are indexed by the
        #   ObsRow above. The idea is that the model tells what data it
        #   computes.
        raise NotImplementedError


def residuals_error(results, observations):
    return results - observations


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

                positive_cumulated += t.num_positive
                rows.append(ir + [positive_cumulated])

    return head, observations, rows
