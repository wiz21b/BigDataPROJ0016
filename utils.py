from __future__ import annotations
from enum import Enum
import numpy as np
import csv
from collections import namedtuple
import urllib.request

COLORS = ['red', 'green', 'blue', 'magenta', 'purple',
          'lime', 'orange', 'chocolate', 'gray',
          'darkgreen', 'darkviolet']



class ObsEnum(Enum):
    # These 6 are the one provided by the teachers.
    # Do not change them

    DAYS = 0

    # number of individuals tested positive on this day.
    NUM_POSITIVE = 1

    # number of tests performed during the last day.
    NUM_TESTED = 2

    # number of individuals currently at the hospital.
    NUM_HOSPITALIZED = 3

    # cumulative number of individuals who were or are
    # being hospitalized.
    CUMULATIVE_HOSPITALIZATIONS = 4

    # number of individuals currently in an ICU (criticals
    # are not counted as part of num_hospitalized).
    NUM_CRITICAL = 5

    # cumulative number of deaths
    NUM_FATALITIES = 6

    # Other data series we add to the dataset
    CUMULATIVE_TESTED_POSITIVE = 7
    CUMULATIVE_TESTED = 8
    RSURVIVOR = 9
    DHDT = 10
    DFDT = 11

    # Other data not in the dataset
    # RECOVERED = 10
    # SUSPECT = 12


    def __int__(self):
        return self.value

    def __str__(self):
        if self in STRINGS:
            return STRINGS[self]

        return self.name.replace("_", " ").lower()

    @staticmethod
    def color(t: ObsEnum):
        if t in COLORS_DICT:
            return COLORS_DICT[t]

        return COLORS[t.value]


class StateEnum(Enum):
    # The mutually exclusive states of our SEIHCR model
    SUSCEPTIBLE = 0
    EXPOSED = 1
    ASYMPTOMATIQUE = 2
    SYMPTOMATIQUE = 3
    HOSPITALIZED = 4
    CRITICAL = 5
    FATALITIES = 6
    RECOVERED = 7
    DHDT = 8
    DFDT = 9
    DSPDT = 10
    DTESTEDDT = 11
    DTESTEDPOSDT = 12

    # Additional deduced states
    # INFECTED_PER_DAY = 8
    # RSURVIVOR = 9
    # CUMULI = 10


    def __int__(self):
        return self.value

    def __str__(self):
        if self in STRINGS:
            return STRINGS[self]
        return self.name.replace("_", " ").lower()

    @staticmethod
    def color(t: ObsEnum):
        return COLORS[t.value]


# The two underlying Enum classes are used to match the indexes of the
# "fitting and fitted values" between the observations and the states
class ObsFitEnum(Enum):
    DHDT = ObsEnum.DHDT.value
    DFDT = ObsEnum.DFDT.value
    DSPDT = ObsEnum.NUM_POSITIVE.value
    # RSURVIVOR =  ObsEnum.RSURVIVOR.value

    def __int__(self):
        return self.value

    def __str__(self):
        return self.name.replace("_", " ").lower()


class StateFitEnum(Enum):
    DHDT = StateEnum.DHDT.value
    DFTD = StateEnum.DFDT.value
    DSPDT = StateEnum.DSPDT.value
    # RSURVIVOR = StateEnum.RSURVIVOR.value

    def __int__(self):
        return self.value

    def __str__(self):
        return self.name.replace("_", " ").lower()


STRINGS = { ObsEnum.NUM_POSITIVE : "num positive",
            ObsEnum.NUM_TESTED : "Tested / day",
            ObsEnum.CUMULATIVE_TESTED_POSITIVE : "Cumulative tested positive",
            ObsEnum.CUMULATIVE_TESTED : "Cumulative tested",
            ObsEnum.CUMULATIVE_HOSPITALIZATIONS : "Cumulative hospitalizations",
            StateEnum.SYMPTOMATIQUE : "Symptomatic",
            StateEnum.ASYMPTOMATIQUE : "Asymptomatic",
            StateEnum.EXPOSED : "Exposed",
            StateEnum.SUSCEPTIBLE:"Susceptible",
            StateEnum.RECOVERED: "Recovered",
            StateEnum.HOSPITALIZED: "Hospitalized",
            StateEnum.CRITICAL: "Critical",
            StateEnum.FATALITIES: "Fatalities"
           }

COLORS_DICT = {ObsEnum.NUM_POSITIVE: 'green',
               StateEnum.DTESTEDPOSDT:'green',
               StateEnum.DTESTEDDT:'green',
               ObsEnum.CUMULATIVE_TESTED_POSITIVE: "green",
               ObsEnum.NUM_TESTED: 'blue',
               ObsEnum.CUMULATIVE_TESTED: "blue",
               ObsEnum.DHDT: 'purple',
               StateEnum.DHDT: 'purple',
               ObsEnum.DFDT: 'black',
               StateEnum.SYMPTOMATIQUE : 'lime',
               StateEnum.ASYMPTOMATIQUE : 'pink',
               StateEnum.EXPOSED : 'brown',
               StateEnum.SUSCEPTIBLE:'grey',
               StateEnum.RECOVERED: 'cyan',
               StateEnum.DFDT: 'black',
               ObsEnum.NUM_HOSPITALIZED: 'purple',
               ObsEnum.NUM_CRITICAL: 'red',
               ObsEnum.NUM_FATALITIES: 'black',
               ObsEnum.CUMULATIVE_HOSPITALIZATIONS: "purple",
               StateEnum.HOSPITALIZED: 'purple',
               StateEnum.CRITICAL: 'red',
               StateEnum.FATALITIES: 'black'}


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
    # Useful for lmfit when usead as leastsq
    return results - observations


def residual_sum_of_squares(results, observations):
    d = results - observations
    return np.sum(d * d)


def log_residual_sum_of_squares(results, observations):
    # Experimental
    # The idea is to weight more the small errors
    # compared to the big ones
    d = results - observations
    try:
        d = d * d
    except:
        pass

    d = np.where(d == float('-inf'), 0, d)
    return np.sum(d)


def load_data( url="https://raw.githubusercontent.com/ADelau/proj0016-epidemic-data/main/data.csv"):
    observations = []
    rows = []
    positive_cumulated = 0
    tested_cumulated = 0

    with urllib.request.urlopen(url) as fp:

        data = fp.read().decode("utf-8").split('\n')
        csvreader = csv.reader(data, delimiter = ',')

        head = next(csvreader)
        DataTuple = namedtuple("DataTuple", head)

        # for i in range(5):
        #     next(csvreader)

        rsurvivor_1 = 0
        cumuldhdt_1 = 0
        cumuldfdt_1 = 0

        for r in csvreader:
            if r:
                ir = [int(x) for x in r]
                t = DataTuple(*ir)
                observations.append(t)
                positive_cumulated += t.num_positive
                tested_cumulated += t.num_tested
                # rsurvivor = t.num_cumulative_hospitalizations - t.num_hospitalised - t.num_critical -t.num_fatalities
                rsurvivor = t.num_cumulative_hospitalizations - t.num_hospitalised - t.num_critical -t.num_fatalities - rsurvivor_1
                rsurvivor_1 = rsurvivor

                dhdt = t.num_cumulative_hospitalizations - cumuldhdt_1
                cumuldhdt_1 = t.num_cumulative_hospitalizations

                dfdt = t.num_fatalities - cumuldfdt_1
                cumuldfdt_1 += dfdt
                rows.append(ir + [positive_cumulated, tested_cumulated,rsurvivor,dhdt,dfdt])

    return head, observations, rows
