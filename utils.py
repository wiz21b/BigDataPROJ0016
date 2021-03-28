from __future__ import annotations
from enum import Enum
import numpy as np
import csv
from collections import namedtuple
import urllib.request

import os
import datetime
import tempfile
from io import StringIO
import pandas as pd
from functools import reduce


COLORS = ['red', 'green', 'blue', 'magenta', 'purple',
          'lime', 'orange', 'chocolate', 'gray',
          'darkgreen', 'darkviolet']

OBSERVATIONS = ["DATE", "NUM_POSITIVE", "CUMULATIVE_POSITIVE", "NUM_TESTED", "CUMULATIVE_TESTED", ]

class ObsEnum(Enum):
    # the date of the current day
    DATE = 0
    # the number of individuals tested positive on this day.
    NEW_POSITIVE = 1
    # the cumulative number of individuals tested positive until this day.
    CUMULATIVE_POSITIVE = 2
    # the number of tests performed during this day.
    NEW_TESTED = 3
    # the cumulative number of tests until this day.
    CUMULATIVE_TESTED = 4

    # the number of new patients at the hospital (in simple beds or in ICU) on this day.
    NEW_IN = 4 # = DHDT
    # the number of people that moved out of the hospital (from simple beds or from ICU) on this day.
    NEW_OUT = 5  # = RSURVIVOR
    # the number of individuals currently at the hospital excluding those in ICU.
    NUM_HOSPITALIZED = 6
    # the cumulative number of individuals who were or are being hospitalized.
    CUMULATIVE_HOSPITALIZATIONS = 7
    # the number of individuals currently in ICU (num_critical are not counted as part of num_hospitalized).
    NUM_CRITICAL = 8

    # the number of new deaths on the current day.
    NEW_FATALITIES = 9  # = DFDT
    # the cumulative number of deaths until the current day.
    NUM_FATALITIES = 10

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
    # -------- The mutually exclusive states of our extended SEIR model --------

    # The number of people that could catch the virus at the current day.
    SUSCEPTIBLE = 0
    # The number of people that caught the disease but are not infectious yet.
    EXPOSED = 1

    # --- Infectious ---
    # The number of people that currently do not develop symptoms and yet are infectious.
    ASYMPTOMATIC = 2
    # The number of people that currently have symptoms and are infectious.
    SYMPTOMATIC = 3
    # ---

    # --- At the hospital ---
    # The number of people at the hospital in simple beds (excluding ICU beds).
    HOSPITALIZED = 4
    # The number of people at the hospital in ICU.
    CRITICAL = 5
    # ---

    # The number of deaths.
    FATALITIES = 6
    # The number of people that recovered from the virus and are not susceptible to catch the disease.
    RECOVERED = 7
    # The number of new patients at the hospital (in simple beds or in ICU) on this day.
    NEW_IN = 8 # = DHDT
    # The number of new deaths on the current day.
    NEW_FATALITIES = 9 # = DFDT

    #  --------


    # -------- The additional non-mutually exclusive states of our model --------

    # The number of people detected as infected on the current day
    NEW_POSITIVE = 8 # = INFECTED_PER_DAY
    # The cumulative number of people that were detected as infected until the current day.
    CUMULATIVE_POSITIVE = 9 # = CUMULI
    # The number of newly tested people on this day.
    NEW_TESTED = 10 # = DTESTEDDT
    # The number of people that leaves the hospital healthy on the current day.
    NEW_OUT = 11 # = RSURVIVOR

    # --------


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
    NEW_POSITIVE = ObsEnum.NEW_POSITIVE.value  # DSPDT = ObsEnum.NUM_POSITIVE.value
    NEW_IN = ObsEnum.NEW_IN.value # DHDT = ObsEnum.DHDT.value
    NEW_OUT = ObsEnum.NEW_OUT.value  # RSURVIVOR =  ObsEnum.RSURVIVOR.value
    NEW_FATALITIES = ObsEnum.NEW_FATALITIES.value # DFDT = ObsEnum.DFDT.value


    def __int__(self):
        return self.value

    def __str__(self):
        return self.name.replace("_", " ").lower()


class StateFitEnum(Enum):
    NEW_POSITIVE = StateEnum.NEW_POSITIVE.value  # DSPDT = StateEnum.DSPDT.value
    NEW_IN = StateEnum.NEW_IN.value  # DHDT = StateEnum.DHDT.value
    NEW_OUT = StateEnum.NEW_OUT.value  # RSURVIVOR = StateEnum.RSURVIVOR.value
    NEW_FATALITIES = StateEnum.NEW_FATALITIES.value  # DFTD = StateEnum.DFDT.value

    def __int__(self):
        return self.value

    def __str__(self):
        return self.name.replace("_", " ").lower()


STRINGS = { ObsEnum.NEW_POSITIVE : "Tested Positive",
            ObsEnum.NEW_TESTED : "Tested / Day",
            ObsEnum.CUMULATIVE_POSITIVE : "Cumulative Tested Positive",
            ObsEnum.CUMULATIVE_TESTED : "Cumulative Tested",
            ObsEnum.CUMULATIVE_HOSPITALIZATIONS : "Cumulative Hospitalizations",
            StateEnum.SYMPTOMATIC : "Symptomatic",
            StateEnum.ASYMPTOMATIC : "Asymptomatic",
            StateEnum.EXPOSED : "Exposed",
            StateEnum.SUSCEPTIBLE:"Susceptible",
            StateEnum.RECOVERED: "Recovered",
            StateEnum.HOSPITALIZED: "Hospitalized",
            StateEnum.CRITICAL: "Critical",
            StateEnum.FATALITIES: "Fatalities"
           }

COLORS_DICT = {ObsEnum.NEW_POSITIVE: 'green',
               StateEnum.NEW_POSITIVE:'green',
               StateEnum.NEW_POSITIVE:'green',
               ObsEnum.CUMULATIVE_POSITIVE: "green",
               ObsEnum.NEW_TESTED: 'blue',
               ObsEnum.CUMULATIVE_TESTED: "blue",
               ObsEnum.NEW_IN: 'purple',
               StateEnum.NEW_IN: 'purple',
               ObsEnum.NEW_FATALITIES: 'black',
               StateEnum.SYMPTOMATIC: 'lime',
               StateEnum.ASYMPTOMATIC: 'pink',
               StateEnum.EXPOSED: 'brown',
               StateEnum.SUSCEPTIBLE:'grey',
               StateEnum.RECOVERED: 'cyan',
               StateEnum.NEW_FATALITIES: 'black',
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


_SCIENSANO_URLS = [
    ("https://epistat.sciensano.be/Data/COVID19BE_CASES_MUNI_CUM.csv"),
    ("https://epistat.sciensano.be/Data/COVID19BE_CASES_AGESEX.csv"),
    ("https://epistat.sciensano.be/Data/COVID19BE_CASES_MUNI.csv"),
    ("https://epistat.sciensano.be/Data/COVID19BE_HOSP.csv"),
    ("https://epistat.sciensano.be/Data/COVID19BE_MORT.csv"),
    ("https://epistat.sciensano.be/Data/COVID19BE_tests.csv"),
    ("https://epistat.sciensano.be/Data/COVID19BE_VACC.csv")]


def _read_csv(url) -> pandas.DataFrame:
    """ Read a CSV file at url.

    Will cache a copy so that subsequent runs will reuse the
    copy, avoid HTTP connection and improving load times by
    a factor of 10. The copy is refreshed every day.
    """

    name = url.rsplit('/', 1)[-1]
    date = datetime.date.today().strftime("%Y%m%d")

    fname = f"cache_{date}_{name}"
    fpath = os.path.join(tempfile.gettempdir(), fname)

    if os.path.exists(fpath):
        # print(f"Reading cached file {fpath}")
        with open(fpath, "rb") as fp:
            data = fp.read()
    else:
        # print(f"Loading data from Sciensano {url}")
        with urllib.request.urlopen(url) as fp:
            data = fp.read()

        with open(fpath, "wb") as fp:
            fp.write(data)

    data = data.decode("utf-8")
    if "DATE" in data:
        parse_dates = ['DATE']
    else:
        parse_dates = False

    dtypes = {}
    if "NIS5" in data:
        dtypes['NIS5'] = pd.Int64Dtype()

    csv = pd.read_csv(StringIO(data),
                      dtype=dtypes,
                      parse_dates=parse_dates)

    # csv.info()
    # print(csv)
    return csv


def load_sciensano_data():
    start_time = datetime.datetime.now()
    _csvs = [_read_csv(url) for url in _SCIENSANO_URLS]
    end_time = datetime.datetime.now()
    print(f"Loaded data in {(end_time - start_time).total_seconds():.2f} sec.")

    CASES_MUNI_CUM, CASES_AGESEX, CASES_MUNI, HOSP, MORT, TESTS, VACC = _csvs
    # Fixing data entries equal to "<5"
    CASES_MUNI["CASES"].replace("<5", "2.5", inplace=True)
    # Fixing type
    CASES_MUNI["CASES"] = pd.to_numeric(CASES_MUNI["CASES"])

    return CASES_MUNI_CUM, CASES_AGESEX, CASES_MUNI, HOSP, MORT, TESTS, VACC


def load_model_data():
    # Load sciensano's datasets
    CASES_MUNI_CUM, CASES_AGESEX, CASES_MUNI, HOSP, MORT, TESTS, VACC = load_sciensano_data()

    # Grouping together the dataframe rows when similar dates via a sum operation
    DAILY_TESTS = TESTS.groupby("DATE", as_index = False).sum()
    DAILY_HOSP = HOSP.groupby("DATE", as_index = False).sum()
    DAILY_DEATHS = MORT.groupby("DATE", as_index = False).sum()

    # Selection and renaming of dataframes columns
    DAILY_TESTS = pd.concat([DAILY_TESTS.DATE,
                             DAILY_TESTS.TESTS_ALL_POS.rename("NEW_POSITIVE"),
                             DAILY_TESTS.TESTS_ALL_POS.cumsum().rename("CUMULATIVE_POSITIVE"),
                             DAILY_TESTS.TESTS_ALL.rename("NEW_TESTED"),
                             DAILY_TESTS.TESTS_ALL.cumsum().rename("CUMULATIVE_TESTED")], axis=1)

    DAILY_HOSP = pd.concat([DAILY_HOSP.DATE,
                            DAILY_HOSP.NEW_IN,
                            DAILY_HOSP.NEW_OUT,
                            (DAILY_HOSP.TOTAL_IN - DAILY_HOSP.TOTAL_IN_ICU).rename("NUM_HOSPITALIZED"),
                            DAILY_HOSP.NEW_IN.cumsum().rename("CUMULATIVE_HOSPITALIZATIONS"),
                            DAILY_HOSP.TOTAL_IN_ICU.rename("NUM_CRITICAL")], axis=1)

    DAILY_DEATHS = pd.concat([DAILY_DEATHS.DATE,
                              DAILY_DEATHS.DEATHS.rename("NEW_FATALITIES"),
                              DAILY_DEATHS.DEATHS.cumsum().rename("NUM_FATALITIES")], axis=1)

    # Outer join of the dataframes on column 'DATE'
    df = reduce(lambda left, right: pd.merge(left, right, on=['DATE'], how='outer'),
                [DAILY_TESTS, DAILY_HOSP, DAILY_DEATHS]).fillna(0)

    return df
