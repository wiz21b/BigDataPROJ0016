import numpy as np
import matplotlib.pyplot as plt

from utils import ObsEnum, Model, residuals_error, load_data

if __name__ == "__main__":
    head, observations, rows = load_data()

    rows = np.array(rows).transpose()

    print(head)
    print(rows)

    x = rows[ObsEnum.DAYS.value]

    for t in [ObsEnum.CUMULATIVE_POSITIVE,
              ObsEnum.CUMULATIVE_TESTED,
              ObsEnum.HOSPITALIZED]:

        plt.plot(np.log(rows[t.value]) / np.log(2), label=f"{t}", color=ObsEnum.color(t))
        y = np.log(rows[t.value]) / np.log(2)
        y[0] = 0
        coef = np.polyfit(x,y,1)
        poly1d_fn = np.poly1d(coef)
        plt.plot(x,poly1d_fn(x),'--', color=ObsEnum.color(t))


    plt.xlabel('Days')
    plt.ylabel(r'$log_2(Individuals)$')
    plt.legend()
    #plt.show()
    plt.savefig('log_plot.pdf')



    # --------------------------------------------------------
    plt.figure()
    for t in [ObsEnum.CUMULATIVE_POSITIVE,
              ObsEnum.CUMULATIVE_TESTED,
              ObsEnum.HOSPITALIZED,
              ObsEnum.FATALITIES]:
        plt.plot(rows[t.value], label=f"{t}")

    plt.xlabel('Days')
    plt.ylabel('Individuals')
    plt.legend()
    #plt.show()
    plt.savefig('linear_plot.pdf')


    # --------------------------------------------------------
    plt.figure()
    plt.plot(rows[ObsEnum.CUMULATIVE_POSITIVE.value] / rows[ObsEnum.CUMULATIVE_TESTED.value], label=f"cumulative positive/tested ratio")
    plt.ylim(0,1)
    plt.xlabel('Days')
    plt.ylabel('Ratio')
    plt.legend()
    plt.savefig('tested_positive_ratio.pdf')


    # --------------------------------------------------------
    plt.figure()
    a = rows[ObsEnum.CUMULATIVE_POSITIVE.value]
    b = rows[ObsEnum.HOSPITALIZED.value]
    plt.plot(a / b, label=f"cumulative positive/hospitalized ratio")
    a = rows[ObsEnum.CUMULATIVE_TESTED.value]
    b = rows[ObsEnum.HOSPITALIZED.value]
    plt.plot(a / b, label=f"cumulative tested/hospitalized ratio")

    plt.plot(rows[ObsEnum.CUMULATIVE_POSITIVE.value] / rows[ObsEnum.CUMULATIVE_TESTED.value], label=f"cumulative positive/tested ratio")

    plt.xlabel('Days')
    plt.ylabel('Ratio')
    plt.legend()
    plt.savefig('hospitalized_ratio.pdf')

    plt.show()
