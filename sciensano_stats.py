import matplotlib.pyplot as plt

from read_sciensano import *

def rolling_mean(df, window):
    return df
    #return df.rolling(window=window).mean()

# CASES_MUNI_CUM, CASES_AGESEX, CASES_MUNI, HOSP, MORT, TESTS, VACC = _read_data()

cumul_cases = CASES_MUNI[["DATE","CASES"]].groupby("DATE").sum().reset_index()

plt.plot(cumul_cases['DATE'], rolling_mean(cumul_cases['CASES'], 14), label="New case/day")

cumul_total_in = HOSP[["DATE","TOTAL_IN","TOTAL_IN_ICU"]].groupby("DATE").sum().reset_index()

plt.plot(cumul_total_in['DATE'], rolling_mean(cumul_total_in['TOTAL_IN'],6), label="In hospital/day")
plt.plot(cumul_total_in['DATE'], cumul_total_in['TOTAL_IN_ICU'], label="In ICU/day")
plt.xlabel("Time")
plt.ylabel("Individuals")

cumul_total_deaths = MORT[["DATE","DEATHS"]].groupby("DATE").sum().reset_index()
plt.plot(cumul_total_deaths['DATE'], cumul_total_deaths['DEATHS'], label="Deaths/day")

cumul_tests = TESTS[["DATE","TESTS_ALL","TESTS_ALL_POS"]].groupby("DATE").sum().reset_index()
plt.plot(cumul_tests['DATE'], rolling_mean(cumul_tests['TESTS_ALL'],14), label="Tests/day")
plt.plot(cumul_tests['DATE'], rolling_mean(cumul_tests['TESTS_ALL_POS'],14), label="Positive/day")

print(VACC)

print(list(VACC.columns.values))
first_dose = VACC[["DATE","DOSE","COUNT"]].query('DOSE == "A"')
cumul_vacc_1st_dose = first_dose.groupby("DATE").sum().reset_index()
second_dose = VACC[["DATE","DOSE","COUNT"]].query('DOSE == "B"')
cumul_vacc_2nd_dose = second_dose.groupby("DATE").sum().reset_index()

plt.plot(cumul_vacc_1st_dose['DATE'], rolling_mean(cumul_vacc_1st_dose['COUNT'],14), label="1st dose/day")
plt.plot(cumul_vacc_2nd_dose['DATE'], rolling_mean(cumul_vacc_2nd_dose['COUNT'],14), label="2nd dose/day")

plt.title("Sciensano dataset")
plt.legend()
plt.show()
