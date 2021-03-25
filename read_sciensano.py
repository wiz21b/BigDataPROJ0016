import os
import datetime
import tempfile
import urllib.request
from io import StringIO
import pandas

_URLS = [("https://epistat.sciensano.be/Data/COVID19BE_CASES_MUNI_CUM.csv"),
         ("https://epistat.sciensano.be/Data/COVID19BE_CASES_AGESEX.csv"),
         ("https://epistat.sciensano.be/Data/COVID19BE_CASES_MUNI.csv"),
         ("https://epistat.sciensano.be/Data/COVID19BE_HOSP.csv"),
         ("https://epistat.sciensano.be/Data/COVID19BE_MORT.csv"),
         ("https://epistat.sciensano.be/Data/COVID19BE_tests.csv"),
         ("https://epistat.sciensano.be/Data/COVID19BE_VACC.csv")]


def read_csv(url) -> pandas.DataFrame:
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
        dtypes['NIS5'] = pandas.Int64Dtype()

    csv = pandas.read_csv(StringIO(data),
                          dtype=dtypes,
                          parse_dates=parse_dates)

    # csv.info()
    # print(csv)
    return csv


def _read_data():
    start_time = datetime.datetime.now()
    _csvs = [read_csv(url) for url in _URLS]
    end_time = datetime.datetime.now()
    print(f"Loaded data in {(end_time - start_time).total_seconds():.2f} sec.")
    return _csvs

CASES_MUNI_CUM, CASES_AGESEX, CASES_MUNI, HOSP, MORT, TESTS, VACC = _read_data()
# Fixing data "<5"
CASES_MUNI["CASES"].replace("<5", "2.5", inplace=True)
# Fixing type
CASES_MUNI["CASES"] = pandas.to_numeric(CASES_MUNI["CASES"])
