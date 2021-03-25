import matplotlib.pyplot as plt

from read_sciensano import *

# CASES_MUNI_CUM, CASES_AGESEX, CASES_MUNI, HOSP, MORT, TESTS, VACC = _read_data()

cumul_cases = CASES_MUNI[["DATE","CASES"]].groupby("DATE").sum().reset_index()

plt.plot(cumul_cases['DATE'],cumul_cases['CASES'].rolling(window=14).mean(), label="New case per day")

print(HOSP)
cumul_total_in = HOSP[["DATE","TOTAL_IN","TOTAL_IN_ICU"]].groupby("DATE").sum().reset_index()

plt.plot(cumul_total_in['DATE'], cumul_total_in['TOTAL_IN'].rolling(window=6).mean(), label="In hospital")
plt.plot(cumul_total_in['DATE'], cumul_total_in['TOTAL_IN_ICU'], label="In ICU")
plt.xlabel("Time")
plt.ylabel("Individuals")
plt.title("Sciensano dataset")
plt.legend()
plt.show()
