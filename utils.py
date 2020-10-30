from __future__ import annotations
from enum import Enum
import csv
from collections import namedtuple
import urllib.request

COLORS = ['red', 'green', 'blue', 'magenta', 'purple',
          'lime', 'orange', 'chocolate', 'gray',
          'darkgreen', 'darkviolet']

class ObsEnum(Enum):
     # These 5 are the one provided by the teachers.
    # Do not change them

    DAYS = 0

    # number of individuals tested positive on this day.
    POSITIVE = 1

    # number of tests performed during the last day.
    TESTED = 2

    # number of individuals currently at the hospital.
    HOSPITALIZED = 3

    # cumulative number of individuals who were or are
    # being hospitalized.
    CUMULATIVE_HOSPITALIZATIONS = 4

    # number of individuals currently in an ICU (criticals
    # are not counted as part of num_hospitalized).
    CRITICAL = 5

    # cumulative number of deaths
    FATALITIES = 6

    # Other data series we add to the dataset
    CUMULATIVE_POSITIVE = 7
    CUMULATIVE_TESTED = 8

    # Other data not in the dataset
    #RECOVERED = 9
    #SUSPECT = 10

    def __str__(self):
        return self.name.replace("_", " ").lower()
    
    @staticmethod
    def color(t: ObsRow):
        return COLORS[t.value]

class StateEnum(Enum):
    # These 5 are the one provided by the teachers.
    # Do not change them

    SUCEPTIBLE = 0
    EXPOSED = 1
    INFECTIOUS = 2
    HOSPITALIZED = 3
    CRITICAL = 4
    RECOVERED = 5

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
    tested_cumulated = 0

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
                tested_cumulated += t.num_tested
                rows.append(ir + [positive_cumulated, tested_cumulated])

    return head, observations, rows
