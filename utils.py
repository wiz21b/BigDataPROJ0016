from __future__ import annotations
from enum import Enum
import numpy as np
import pandas as pd
import csv
from collections import namedtuple
from functools import reduce
from enum import Enum
import numpy as np
import csv
from collections import namedtuple
import urllib.request
from datetime import date
import os
import datetime
import tempfile
import matplotlib.dates as mdates
from io import StringIO


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

    # Vaccination
    # Cumulative number of people at least partially vaccinated
    VACCINATED_ONCE = 12
    # Cumulative number of people fully vaccinated
    VACCINATED_TWICE = 13

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
    #RSURVIVOR = ObsEnum.RSURVIVOR.value

    def __int__(self):
        return self.value

    def __str__(self):
        return self.name.replace("_", " ").lower()


class StateFitEnum(Enum):
    DHDT = StateEnum.DHDT.value
    DFTD = StateEnum.DFDT.value
    DSPDT = StateEnum.DSPDT.value
    #RSURVIVOR = StateEnum.RSURVIVOR.value

    def __int__(self):
        return self.value

    def __str__(self):
        return self.name.replace("_", " ").lower()


STRINGS = { ObsEnum.NUM_POSITIVE : "Positive / day",
            ObsEnum.NUM_TESTED : "Tested / day",
            ObsEnum.NUM_CRITICAL : "Critical",
            ObsEnum.NUM_HOSPITALIZED : "Hospitalized",
            ObsEnum.NUM_FATALITIES : "Fatalities",
            ObsEnum.CUMULATIVE_TESTED_POSITIVE : "Cumulative tested positive",
            ObsEnum.CUMULATIVE_TESTED : "Cumulative tested",
            ObsEnum.CUMULATIVE_HOSPITALIZATIONS : "Cumulative hospitalizations",
            ObsEnum.DHDT: "Hospitalized / day",
            ObsEnum.DFDT: "Fatalities / day",
            StateEnum.SYMPTOMATIQUE : "Symptomatic",
            StateEnum.ASYMPTOMATIQUE : "Asymptomatic",
            StateEnum.EXPOSED : "Exposed",
            StateEnum.SUSCEPTIBLE:"Susceptible",
            StateEnum.RECOVERED: "Recovered",
            StateEnum.HOSPITALIZED: "Hospitalized",
            StateEnum.CRITICAL: "Critical",
            StateEnum.FATALITIES: "Fatalities",
            StateEnum.DTESTEDDT: "Tested / day",
            StateEnum.DTESTEDPOSDT : "Positive / day",
            StateEnum.DHDT: "Hospitalized / day",
            StateEnum.DFDT: "Fatalities / day"
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
    def __init__(self, stocha = True, errorFct = None, nbExperiments = 100):
        self._stochastic = stocha
        self._ICInitialized = False
        self._paramInitialized = False
        self._fitted = False
        self._initialConditions = {}
        self._currentState = {}
        self._constantParamNames = {}
        self._params = {}
        self._optimalParams = {}
        self._population = 0
        self._nbExperiments = nbExperiments
        self._errorFct = None
        self._data = None
        self._dataLength = 0

        if not(stocha):
            self._errorFct = errorFct

    def fit_parameters(self, method, randomPick):
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

    def population_leave(self, param, population):
        # param : the proportion of population that should
        # leave on average

        if population < 0:
            # Fix for some edge cases
            return 0

        # Part of the population that leaves on average
        #average = param * population

        # Binomial centered on the population part

        # The rounding is important because binomial
        # is for integer number. By using a round we favor
        # sometimes the high limit sometimes the low
        # limit => on average we center. I think np
        # will use "int" instead which always favour
        # the low limit => the distribution is skewed.
        #r = np.random.binomial(round(2 * average), 0.5)
        r = np.random.binomial(population, param)

        return r


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


def _read_csv(url) -> pd.DataFrame:
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
    #print(csv)
    return csv


def load_sciensano_data():
    start_time = datetime.datetime.now()
    _csvs = [_read_csv(url) for url in _SCIENSANO_URLS]
    end_time = datetime.datetime.now()
    #print(f"Loaded data in {(end_time - start_time).total_seconds():.2f} sec.")

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
    DAILY_VACCINES = VACC.groupby(["DATE", "DOSE"], as_index = False).sum()

    fraction_hospitalized_per_day_added = 0.00112927
    fraction_rsurvivor_added = 0.00114305

    # Selection and renaming of dataframes columns
    DAILY_TESTS1 = pd.concat([DAILY_TESTS.DATE,
                              DAILY_TESTS.TESTS_ALL_POS.rename("NUM_POSITIVE"),
                              DAILY_TESTS.TESTS_ALL.rename("NUM_TESTED")], axis=1)

    DAILY_HOSP1 = pd.concat([DAILY_HOSP.DATE,
                             (DAILY_HOSP.TOTAL_IN - DAILY_HOSP.TOTAL_IN_ICU).rename("NUM_HOSPITALIZED"),
                             (DAILY_HOSP.NEW_IN * (1 + fraction_hospitalized_per_day_added)).cumsum().rename("CUMULATIVE_HOSPITALIZATIONS"),
                             DAILY_HOSP.TOTAL_IN_ICU.rename("NUM_CRITICAL")], axis=1)

    DAILY_DEATHS1 = pd.concat([DAILY_DEATHS.DATE,
                               DAILY_DEATHS.DEATHS.cumsum().rename("NUM_FATALITIES")], axis=1)

    DAILY_TESTS2 = pd.concat([DAILY_TESTS.DATE,
                              DAILY_TESTS.TESTS_ALL_POS.cumsum().rename("CUMULATIVE_TESTED_POSITIVE"),
                              DAILY_TESTS.TESTS_ALL.cumsum().rename("CUMULATIVE_TESTED")], axis=1)

    DAILY_HOSP2 = pd.concat([DAILY_HOSP.DATE,
                             (DAILY_HOSP.NEW_OUT * (1 + fraction_rsurvivor_added)).rename("RSURVIVOR"),
                             # STC prefers but still need more test : (DAILY_HOSP.NEW_IN - DAILY_HOSP.NEW_OUT).rename("DHDT")], axis=1)
                             (DAILY_HOSP.NEW_IN * (1 + fraction_hospitalized_per_day_added)).rename("DHDT")], axis=1)

    DAILY_DEATHS2 = pd.concat([DAILY_DEATHS.DATE,
                               DAILY_DEATHS.DEATHS.rename("DFDT")], axis=1)

    DAILY_VACCINES1 = DAILY_VACCINES.loc[DAILY_VACCINES["DOSE"] == 'A', ["DATE", "COUNT"]]
    DAILY_VACCINES1["COUNT"] = DAILY_VACCINES1.COUNT.cumsum()
    DAILY_VACCINES1 = DAILY_VACCINES1.rename(columns = {"COUNT": "VACCINATED_ONCE"})
    DAILY_VACCINES2 = DAILY_VACCINES.loc[DAILY_VACCINES["DOSE"] == 'B', ["DATE", "COUNT"]]
    DAILY_VACCINES2["COUNT"] = DAILY_VACCINES2.COUNT.cumsum()
    DAILY_VACCINES2 = DAILY_VACCINES2.rename(columns = {"COUNT": "VACCINATED_TWICE"})

    # Outer join of the dataframes on column 'DATE'
    # Attention à bien respecter l'ordre de ObsEnum.
    df = reduce(lambda left, right: pd.merge(left, right, on=['DATE'], how='outer'),
                [DAILY_TESTS1, DAILY_HOSP1, DAILY_DEATHS1, DAILY_TESTS2, DAILY_HOSP2, DAILY_DEATHS2, DAILY_VACCINES1, DAILY_VACCINES2]).fillna(0)

    return df

"""
Load the vaccination data and compute the forecast of number of doses adminisitered

Argument:
one_dose_vaccination_forecasts, a dictionary of (date, cumulative number of doses administered) for the first dose of vaccine
one_dose_vaccination_forecasts, a dictionary of (date, cumulative number of doses administered) for the second dose of vaccine

Return:
a pandas dataframe with - the dates since the beginning of sciensano's reporting
                        - the cumulative number of people at least partially vaccinated
                        - the cumulative number of people fully vaccinated
"""
def load_vaccination_data(one_dose_vaccination_forecasts, two_dose_vaccination_forecasts):
    CASES_MUNI_CUM, CASES_AGESEX, CASES_MUNI, HOSP, MORT, TESTS, VACC = load_sciensano_data()
    DATES = pd.DataFrame({"DATE": TESTS.DATE.unique()})
    DAILY_VACCINES = VACC.groupby(["DATE", "DOSE"], as_index = False).sum()
    DAILY_VACCINES1 = DAILY_VACCINES.loc[DAILY_VACCINES["DOSE"] == 'A', ["DATE", "COUNT"]]
    DAILY_VACCINES1["COUNT"] = DAILY_VACCINES1.COUNT.cumsum()
    DAILY_VACCINES1 = DAILY_VACCINES1.rename(columns = {"COUNT": "VACCINATED_ONCE"})
    DAILY_VACCINES2 = DAILY_VACCINES.loc[DAILY_VACCINES["DOSE"] == 'B', ["DATE", "COUNT"]]
    DAILY_VACCINES2["COUNT"] = DAILY_VACCINES2.COUNT.cumsum()
    DAILY_VACCINES2 = DAILY_VACCINES2.rename(columns = {"COUNT": "VACCINATED_TWICE"})

    # ---- Linear forecasts of the vaccination ----
    n_prev_vaccines = DAILY_VACCINES1.VACCINATED_ONCE.iloc[-1]
    prev_date = DAILY_VACCINES1.DATE.iloc[-1].date()
    for date, n_vaccines in one_dose_vaccination_forecasts.items():
        forecasted_n_vaccines = np.linspace(n_prev_vaccines, n_vaccines, (date - prev_date).days, dtype=int)
        dates = pd.date_range(prev_date, date, freq='d')[1:]
        linear_vaccination_forecasts = pd.DataFrame(data={"DATE":dates, "VACCINATED_ONCE":forecasted_n_vaccines})
        DAILY_VACCINES1 = DAILY_VACCINES1.append(linear_vaccination_forecasts)
        prev_date = dates[-1].date()
        n_prev_vaccines = forecasted_n_vaccines[-1]

    n_prev_vaccines = DAILY_VACCINES2.VACCINATED_TWICE.iloc[-1]
    prev_date = DAILY_VACCINES2.DATE.iloc[-1].date()
    for date, n_vaccines in two_dose_vaccination_forecasts.items():
        forecasted_n_vaccines = np.linspace(n_prev_vaccines, n_vaccines, (date - prev_date).days, dtype = int)
        dates = pd.date_range(prev_date, date, freq = 'd')[1:]
        linear_vaccination_forecasts = pd.DataFrame(data = {"DATE": dates, "VACCINATED_TWICE": forecasted_n_vaccines})
        DAILY_VACCINES2 = DAILY_VACCINES2.append(linear_vaccination_forecasts)
        prev_date = dates[-1].date()
        n_prev_vaccines = forecasted_n_vaccines[-1]

    df = reduce(lambda left, right: pd.merge(left, right, on = ['DATE'], how = 'outer'),
                [DATES, DAILY_VACCINES1, DAILY_VACCINES2]).fillna(method='ffill').fillna(0)
    return df


"""
Compute the number of days between 2 date objects
"""
def days_between_dates(date1, date2):
    return (date2 - date1).days

"""
Compute the periods in number of days between all adjacent dates of a list

Argument:
dates, a list of days

Return:
a list of tuples (start, end) describing each period delimited from the start^th day to the end^th day.
A period is computed for each pair of adjacent dates from the given list of dates
"""
def periods_in_days(dates):
    periods_in_days = []
    current_day = 0
    for i in range(len(dates) - 1):
        elapsed_days = days_between_dates(dates[i], dates[i+1])
        periods_in_days.append((current_day, current_day + elapsed_days))
        current_day += elapsed_days

    return periods_in_days

"""
Plot vertical dashed lines to visualize distinctive periods in a plot for a function of time

Argument:
plt, the matplotlib pyplot object to which we should add vertical lines
dates, a list of dates delimiting periods
"""
def plot_periods(plt, dates):
    periods = periods_in_days(dates)
    plt.xticks([start for start, _ in periods] + [periods[-1][-1]], dates, fontsize = 9)
    plt.gcf().autofmt_xdate(rotation=50)
    for start, _ in periods:
        plt.axvline(start, color='black', linestyle='dashed', lw=0.3)
    plt.axvline(periods[-1][-1], color = 'black', linestyle = 'dashed', lw = 0.3)
