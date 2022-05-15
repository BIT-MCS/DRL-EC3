import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.backends.backend_pdf import PdfPages
import os
import numpy as np


def error(input_list):
    input = np.array(input_list)
    input = input.transpose((1, 0))
    error_low = input[0] - input[1]
    error_high = input[2] - input[0]
    error = []
    error.append(error_low)
    error.append(error_high)
    return error


def average(input_list):
    input = np.array(input_list)
    input = input.transpose((1, 0))
    return input[0]


def compare_plot_errorbar(xlabel, ylabel, x, eDivert, woApeX, woRNN, MADDPG):
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.errorbar(x=x, y=average(eDivert), yerr=error(eDivert), fmt='r-o', label='e-Divert', capsize=4)
    plt.errorbar(x=x, y=average(woApeX), yerr=error(woApeX), fmt='g-^', label='e-Divert w/o Ape-X', capsize=4)
    plt.errorbar(x=x, y=average(woRNN), yerr=error(woRNN), fmt='m-<', label='e-Divert w/o RNN', capsize=4)
    plt.errorbar(x=x, y=average(MADDPG), yerr=error(MADDPG), fmt='k-*', label='MADDPG', capsize=4)

    plt.ylim(ymin=0, ymax=1)
    plt.grid(True)
    plt.grid(linestyle='--')
    plt.legend()
    plt.show()


def compare_plot(xlabel, ylabel, x, yrange, eDivert, TSP):
    if os.path.exists('./pdf') is False:
        os.makedirs('./pdf')
    pdf = PdfPages('./pdf/%s-%s.pdf' % (xlabel, ylabel))
    plt.figure(figsize=(13, 13))

    plt.xlabel(xlabel, fontsize=32)
    plt.ylabel(ylabel, fontsize=32)
    plt.xticks(fontsize=32)
    plt.yticks(fontsize=32)
    plt.plot(x, eDivert, color='b', marker='o', label='e-Divert', markersize=26, markeredgewidth=5,
             markerfacecolor='none', linewidth=4)
    plt.plot(x, TSP, color='orange', marker='s', label='GA-based route planning', markersize=26, markeredgewidth=5,
             markerfacecolor='none', linewidth=4)

    # if ylabel == "Energy usage (# of full batteries)":
    #     if xlabel == "No. of vehicles":
    #         plt.plot(x, [3.62, 4.62, 5.62, 6.62, 7.62], color='red', linestyle='--', label="Maximum used energy",
    #                  linewidth=4)
    #     else:
    #         plt.axhline(y=2.83, color='red', linestyle='--', label="Maximum used energy", linewidth=4)
    plt.xticks(x, x)
    plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    plt.ylim(yrange[0], yrange[1] * 1.5)
    plt.grid(True)
    plt.grid(linestyle='--')
    plt.legend(loc='upper left', fontsize=25, ncol=1, markerscale=0.9)
    plt.tight_layout()

    pdf.savefig()
    plt.close()
    pdf.close()


if __name__ == '__main__':
    # collection-range
    compare_plot(xlabel="Sensing range (unit)",
                 ylabel="Data collection ratio",
                 x=[0.6, 0.8, 1.0, 1.2, 1.4],
                 yrange=[0, 0.8],
                 eDivert=[0.704, 0.719, 0.746, 0.88, 0.95],
                 TSP=[0.905, 0.917, 0.930, 0.952, 0.974],
                 )

    # fairness_range
    compare_plot(xlabel="Sensing range (unit)",
                 ylabel="Geographical fairness",
                 x=[0.6, 0.8, 1.0, 1.2, 1.4],
                 yrange=[0, 0.8],
                 eDivert=[0.755, 0.766, 0.801, 0.91, 0.957],
                 TSP=[0.919, 0.935, 0.950, 0.963, 0.980],
                 )
    # #
    # energy_range
    compare_plot(xlabel="Sensing range (unit)",
                 ylabel="Energy usage (# of full batteries)",
                 x=[0.6, 0.8, 1.0, 1.2, 1.4],
                 yrange=[0, 4],
                 eDivert=[1.32, 1.329, 1.459, 1.57, 1.805],
                 TSP=[3.855, 4.219, 4.234, 4.250, 4.270],
                 )

    # efficiency_range
    compare_plot(xlabel="Sensing range (unit)",
                 ylabel="Energy efficiency",
                 x=[0.6, 0.8, 1.0, 1.2, 1.4],
                 yrange=[0, 0.36],
                 eDivert=[0.357, 0.362, 0.371, 0.382, 0.4],
                 TSP=[0.189, 0.178, 0.183, 0.189, 0.196],
                 )

    # TODO
    # collection-range
    compare_plot(xlabel="No. of vehicles",
                 ylabel="Data collection ratio",
                 x=[1, 2, 3, 4, 5],
                 yrange=[0, 0.8],
                 eDivert=[0.841,0.852,0.902,0.942,0.94],
                 TSP=[0.893,0.992,0.999,0.994,0.994],
                 )

    # fairness_range
    compare_plot(xlabel="No. of vehicles",
                 ylabel="Geographical fairness",
                 x=[1, 2, 3, 4, 5],
                 yrange=[0, 0.8],
                 eDivert=[0.862,0.878,0.921,0.943,0.939],
                 TSP=[0.936,0.988,0.991,0.991,0.991],
                 )
    # #
    # energy_range
    compare_plot(xlabel="No. of vehicles",
                 ylabel="Energy usage (# of full batteries)",
                 x=[1, 2, 3, 4, 5],
                 yrange=[0, 6],
                 eDivert=[1.38,1.784,2.01,2.493,2.67],
                 TSP=[3.395,4.324,4.941,7.402,7.996],
                 )

    # efficiency_range
    compare_plot(xlabel="No. of vehicles",
                 ylabel="Energy efficiency",
                 x=[1, 2, 3, 4, 5],
                 yrange=[0, 0.32],
                 eDivert=[0.386,0.371,0.349,0.311,0.302],
                 TSP=[0.213,0.201,0.179,0.118,0.102],
                 )
