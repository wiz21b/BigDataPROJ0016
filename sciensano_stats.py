import matplotlib.pyplot as plt

from read_sciensano import *

# CASES_MUNI_CUM, CASES_AGESEX, CASES_MUNI, HOSP, MORT, TESTS, VACC = _read_data()

cumul_cases = CASES_MUNI[["DATE","CASES"]].groupby("DATE").sum().reset_index()

plt.plot(cumul_cases['DATE'],cumul_cases['CASES'].rolling(window=14).mean(), label="New case/day")

cumul_total_in = HOSP[["DATE","TOTAL_IN","TOTAL_IN_ICU"]].groupby("DATE").sum().reset_index()

plt.plot(cumul_total_in['DATE'], cumul_total_in['TOTAL_IN'].rolling(window=6).mean(), label="In hospital/day")
plt.plot(cumul_total_in['DATE'], cumul_total_in['TOTAL_IN_ICU'], label="In ICU/day")
plt.xlabel("Time")
plt.ylabel("Individuals")

cumul_total_deaths = MORT[["DATE","DEATHS"]].groupby("DATE").sum().reset_index()
plt.plot(cumul_total_deaths['DATE'], cumul_total_deaths['DEATHS'], label="Deaths/day")

cumul_tests = TESTS[["DATE","TESTS_ALL","TESTS_ALL_POS"]].groupby("DATE").sum().reset_index()
plt.plot(cumul_tests['DATE'], cumul_tests['TESTS_ALL'].rolling(window=14).mean(), label="Tests/day")
plt.plot(cumul_tests['DATE'], cumul_tests['TESTS_ALL_POS'].rolling(window=14).mean(), label="Positive/day")
print(TESTS)


plt.title("Sciensano dataset")
plt.legend()
plt.show()
