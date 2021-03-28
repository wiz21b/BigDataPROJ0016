import numpy as np
import matplotlib.pyplot as plt

from utils import ObsEnum, Model, residuals_error, load_data

if __name__ == "__main__":
    head, observations, rows = load_data()

    rows = np.array(rows).transpose()

    print(head)
    print(rows)

    x = rows[ObsEnum.DAYS.value]

    for t in [ObsEnum.CUMULATIVE_TESTED_POSITIVE,
              ObsEnum.CUMULATIVE_TESTED,
              ObsEnum.CUMULATIVE_HOSPITALIZATIONS]:

        y = np.log(rows[t.value]) / np.log(2)
        y = np.where( y == float('-inf'), 0, y)
        plt.plot(x, y, label=f"{t}", color=ObsEnum.color(t))
        coef = np.polyfit(x,y,1)
        poly1d_fn = np.poly1d(coef)
        plt.plot(x,poly1d_fn(x),'--', color=ObsEnum.color(t))


    plt.title("Cumulative observations")
    plt.xlabel('Days')
    plt.ylabel(r'$log_2(Individuals)$')
    plt.legend()
    #plt.show()
    plt.savefig('log_plot.pdf')



    # --------------------------------------------------------
    plt.figure()

    for t in [ObsEnum.TESTED_POSITIVE,
              ObsEnum.TESTED]:
        y = rows[t.value]
        plt.plot(y, label=f"{t}", color=ObsEnum.color(t))
        coef = np.polyfit(x,y,1)
        poly1d_fn = np.poly1d(coef)
        plt.plot(x,poly1d_fn(x),'--', color=ObsEnum.color(t))

    daily_hospitalizations = rows[ObsEnum.CUMULATIVE_HOSPITALIZATIONS.value][1:] - rows[ObsEnum.CUMULATIVE_HOSPITALIZATIONS.value][0:-1]
    daily_hospitalizations = np.insert(daily_hospitalizations, 0, 0)

    t = ObsEnum.DAILY_HOSPITALIZATIONS
    y = daily_hospitalizations
    plt.plot(daily_hospitalizations, label=str(t), color=ObsEnum.color(t))
    coef = np.polyfit(x,y,1)
    poly1d_fn = np.poly1d(coef)
    plt.plot(x,poly1d_fn(x),'--', color=ObsEnum.color(t))

    plt.xlabel('Days')
    plt.ylabel('Individuals')
    plt.legend()
    plt.title("Daily observations")
    #plt.show()
    plt.savefig('linear_plot.pdf')


    # --------------------------------------------------------
    plt.figure()
    y = rows[ObsEnum.TESTED_POSITIVE.value] / rows[ObsEnum.TESTED.value]
    print(y)
    y[0] = np.mean(y[1:])
    plt.plot(y, label=f"Positive/tested ratio")
    coef = np.polyfit(x,y,1)
    poly1d_fn = np.poly1d(coef)
    #plt.plot(x,poly1d_fn(x),'--')

    y = rows[ObsEnum.CUMULATIVE_TESTED_POSITIVE.value] / rows[ObsEnum.CUMULATIVE_TESTED.value]
    print(y)
    y[0] = np.mean(y[1:])
    plt.plot(y, label=f"Cumulative positive/tested ratio")
    coef = np.polyfit(x,y,1)
    poly1d_fn = np.poly1d(coef)
    #plt.plot(x,poly1d_fn(x),'--')

    plt.ylim(0,1)
    plt.xlabel('Days')
    plt.ylabel('Ratio')
    plt.legend()
    plt.savefig('tested_positive_ratio.pdf')


    # --------------------------------------------------------
    plt.figure()
    a = rows[ObsEnum.CUMULATIVE_TESTED_POSITIVE.value]
    b = rows[ObsEnum.HOSPITALIZED.value]
    plt.plot(a / b, label=f"cumulative positive/hospitalized ratio")
    a = rows[ObsEnum.CUMULATIVE_TESTED.value]
    b = rows[ObsEnum.HOSPITALIZED.value]
    plt.plot(a / b, label=f"cumulative tested/hospitalized ratio")

    plt.plot(rows[ObsEnum.CUMULATIVE_TESTED_POSITIVE.value] / rows[ObsEnum.CUMULATIVE_TESTED.value], label=f"cumulative positive/tested ratio")

    plt.xlabel('Days')
    plt.ylabel('Ratio')
    plt.legend()
    plt.savefig('hospitalized_ratio.pdf')

    plt.show()
