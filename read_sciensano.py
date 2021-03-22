import os
import datetime
import tempfile
import urllib.request
from io import StringIO
import pandas


def read_csv(url):

    name = url.rsplit('/', 1)[-1]
    date = datetime.date.today().strftime("%Y%m%d")

    fname = f"cache_{date}_{name}"
    fpath = os.path.join(tempfile.gettempdir(), fname)

    if os.path.exists(fpath):
        print(f"Reading cached file {fpath}")
        with open(fpath, "rb") as fp:
            data = fp.read()
    else:
        print(f"Loading data from Sciensano {url}")
        with urllib.request.urlopen(url) as fp:
            data = fp.read()

        with open(fpath, "wb") as fp:
            fp.write(data)

    data = data.decode("utf-8")
    if "DATE" in data:
        parse_dates = ['DATE']
    else:
        parse_dates = False

    dtypes= {}
    if "NIS5" in data:
        dtypes['NIS5'] = pandas.Int64Dtype()

    csv = pandas.read_csv( StringIO(data),
                           dtype = dtypes,
                           parse_dates=parse_dates)

    # csv.info()
    # print(csv)
    return csv


_URLS = [("https://epistat.sciensano.be/Data/COVID19BE_CASES_MUNI_CUM.csv"),
         ("https://epistat.sciensano.be/Data/COVID19BE_CASES_AGESEX.csv"),
         ("https://epistat.sciensano.be/Data/COVID19BE_CASES_MUNI.csv"),
         ("https://epistat.sciensano.be/Data/COVID19BE_HOSP.csv"),
         ("https://epistat.sciensano.be/Data/COVID19BE_MORT.csv"),
         ("https://epistat.sciensano.be/Data/COVID19BE_tests.csv"),
         ("https://epistat.sciensano.be/Data/COVID19BE_VACC.csv")]

start_time = datetime.datetime.now()


csvs = [read_csv(url) for url in _URLS]
CASES_MUNI_CUM, CASES_AGESEX, CASES_MUNI, HOSP, MORT, TESTS, VACC = csvs


end_time = datetime.datetime.now()
delta = end_time - start_time
duration = delta.seconds + delta.microseconds/1000000
print(f"Loaded data in {duration:.2f} sec.")
