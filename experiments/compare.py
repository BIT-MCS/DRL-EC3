import matplotlib.pyplot as plt
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


def compare_plot(xlabel, ylabel, x,yrange, eDivert, woApeX, woRNN, MADDPG):
    plt.figure(figsize=(15, 20))
    plt.xlabel(xlabel,fontsize=32)
    plt.ylabel(ylabel,fontsize=32)
    plt.xticks(fontsize=32)
    plt.yticks(fontsize=32)
    plt.plot(x,eDivert, color='b', marker='o', label='e-Divert',markersize=26,markeredgewidth=5,markerfacecolor='none',linewidth=4)
    plt.plot(x, woApeX, color='g',marker='^', label='e-Divert w/o Ape-X',markersize=26,markeredgewidth=5,markerfacecolor='none',linewidth=4)
    plt.plot(x, woRNN, color='m',marker='d', label='e-Divert w/o RNN',markersize=26,markeredgewidth=5,markerfacecolor='none',linewidth=4)
    plt.plot(x, MADDPG, color='k',marker='s', label='MADDPG',markersize=26,markeredgewidth=5,markerfacecolor='none',linewidth=4)
    # plt.plot(x,[3.62,4.62,5.62,6.62,7.62],color='red',linestyle='--',label="Maximum used energy",linewidth=4)

    plt.xticks(x,x)
    # plt.axhline(y=4.62, color='red', linestyle='--', label="Maximum used energy",linewidth=4)
    plt.ylim(yrange[0],yrange[1])
    plt.grid(True)
    plt.grid(linestyle='--')
    plt.legend(loc='lower right',fontsize=22)
    plt.show()


if __name__ == '__main__':
    # collection-range
    compare_plot(xlabel="Sensing range (unit)",
                 ylabel="Data collection ratio",
                 x=[0.6, 0.8, 1.0, 1.2, 1.4],
                 yrange=[0,1],
                 eDivert=[0.706, 0.874, 0.916, 0.936, 0.952],
                 woApeX=[0.584, 0.70, 0.871, 0.906, 0.949],
                 woRNN=[0.205, 0.41, 0.463, 0.569, 0.722],
                 MADDPG=[0.139, 0.245, 0.323, 0.360, 0.439],
                 )


    # fairness_range
    compare_plot(xlabel="Sensing range (unit)",
                 ylabel="Geographical fairness",
                 x=[0.6, 0.8, 1.0, 1.2, 1.4],
                 yrange=[0,1],
                 eDivert=[0.784,0.909,0.936,0.951,0.970],
                 woApeX=[0.675,0.729,0.903,0.935,0.963],
                 woRNN=[0.294,0.467,0.573,0.650,0.777],
                 MADDPG=[0.168,0.293,0.382,0.409,0.5],
                 )
    # # #
    # # energy_range
    # compare_plot(xlabel="Sensing range (unit)",
    #              ylabel="Energy usage (# of full batteries)",
    #              x=[0.6, 0.8, 1.0, 1.2, 1.4],
    #              yrange=[0,5],
    #              eDivert=[3.45,4.086,3.89,3.918,3.9],
    #              woApeX=[3.39,3.588,4.617,4.43,4.48],
    #              woRNN=[1.395,2.514,3.188,3.113,4.113],
    #              MADDPG=[1.792,2.201,2.545,2.547,3.027],
    #              )

    # efficiency_range
    compare_plot(xlabel="Sensing range (unit)",
                 ylabel="Energy efficiency",
                 x=[0.6, 0.8, 1.0, 1.2, 1.4],
                 yrange=[-0.04,0.2],
                 eDivert=[0.129,0.155,0.179,0.182,0.193],
                 woApeX=[0.092,0.118,0.139,0.153,0.165],
                 woRNN=[0.033,0.062,0.063,0.097,0.108],
                 MADDPG=[0.011,0.027,0.039,0.048,0.058],
                 )

    # collection_uav
    compare_plot(xlabel="No. of vehicles",
                 ylabel="Data collection ratio",
                 x=[1, 2, 3, 4, 5],
                 yrange=[0,1],
                 eDivert=[0.88,0.943,0.916,0.912,0.911],
                 woApeX=[0.769,0.871,0.746,0.738,0.764],
                 woRNN=[0.842,0.722,0.636,0.682,0.772],
                 MADDPG=[0.401,0.383,0.415,0.478,0.269],
                 )

    # fairness_uav
    compare_plot(xlabel="No. of vehicles",
                 ylabel="Geographical fairness",
                 x=[1, 2, 3, 4, 5],
                 yrange=[0,1],
                 eDivert=[0.912,0.958,0.943,0.944,0.935],
                 woApeX=[0.814,0.902,0.795,0.790,0.819],
                 woRNN=[0.874,0.777,0.714,0.732,0.815],
                 MADDPG=[0.500,0.431,0.463,0.537,0.338],
                 )
    # #
    # # energy_uav
    # compare_plot(xlabel="No. of vehicles",
    #              ylabel="Energy usage (# of full batteries)",
    #              x=[1, 2, 3, 4, 5],
    #              yrange=[1,8],
    #              eDivert=[3.576,4,4.004,4.402,4.668],
    #              woApeX=[3.244,4.273,4.562,5.156,5.953],
    #              woRNN=[3.42,4.113,4.496,5.613,6.45],
    #              MADDPG=[1.853,2.695,3.543,4.44,5.08],
    #              )

    # efficiency_uav
    compare_plot(xlabel="No. of vehicles",
                 ylabel="Energy efficiency",
                 x=[1, 2, 3, 4, 5],
                 yrange=[-0.04,0.2],
                 eDivert=[0.182,0.181,0.179,0.158,0.149],
                 woApeX=[0.155,0.150,0.104,0.091,0.083],
                 woRNN=[0.174,0.108,0.080,0.080,0.080],
                 MADDPG=[0.085,0.050,0.045,0.046,0.015],
                 )

    # collection_fill
    compare_plot(xlabel="Charging proportion (%)",
                 ylabel="Data collection ratio",
                 x=[10, 20, 30, 40, 50],
                 yrange=[0,1],
                 eDivert=[0.927,0.911,0.937,0.905,0.939],
                 woApeX=[0.736,0.766,0.761,0.791,0.838],
                 woRNN=[0.638,0.702,0.713,0.680,0.672],
                 MADDPG=[0.305,0.354,0.393,0.392,0.369],
                 )

    # fairness_fill
    compare_plot(xlabel="Charging proportion (%)",
                 ylabel="Geographical fairness",
                 x=[10, 20, 30, 40, 50],
                 yrange=[0,1],
                 eDivert=[0.951,0.935,0.958,0.941,0.959],
                 woApeX=[0.804,0.829,0.821,0.843,0.880],
                 woRNN=[0.704,0.745,0.776,0.722,0.727],
                 MADDPG=[0.360,0.425,0.436,0.421,0.431],
                 )

    # # energy_fill
    # compare_plot(xlabel="Charging proportion (%)",
    #              ylabel="Energy usage (# of full batteries)",
    #              x=[10, 20, 30, 40, 50],
    #              yrange=[0,5],
    #              eDivert=[4.023,3.844,3.926,3.73,4],
    #              woApeX=[3.463,3.771,3.74,3.889,4.348],
    #              woRNN=[2.844,3.184,3.457,3.066,3.064],
    #              MADDPG=[2.15,2.285,2.342,2.3,2.244],
    #              )
    #
    # efficiency_fill
    compare_plot(xlabel="Charging proportion (%)",
                 ylabel="Energy efficiency",
                 x=[10, 20, 30, 40, 50],
                 yrange=[0,0.3],
                 eDivert=[0.180,0.180,0.185,0.185,0.184],
                 woApeX=[0.138,0.136,0.137,0.141,0.139],
                 woRNN=[0.132,0.132,0.131,0.131,0.132],
                 MADDPG=[0.044,0.055,0.059,0.061,0.057],
                 )

    # collection_station
    compare_plot(xlabel="No. of charging stations",
                 ylabel="Data collection ratio",
                 x=[1,2,3,4,5],
                 yrange=[0,1],
                 eDivert=[0.819,0.865,0.911,0.905,0.943],
                 woApeX=[0.461,0.680,0.795,0.874,0.871],
                 woRNN=[0.480,0.684,0.702,0.649,0.688],
                 MADDPG=[0.366,0.366,0.332,0.336,0.371],
                 )

    # fairness_station
    compare_plot(xlabel="No. of charging stations",
                 ylabel="Geographical fairness",
                 x=[1, 2, 3, 4, 5],
                 yrange=[0,1],
                 eDivert=[0.865,0.906,0.935,0.934,0.958],
                 woApeX=[0.526,0.734,0.851,0.903,0.902],
                 woRNN=[0.547,0.710,0.745,0.694,0.758],
                 MADDPG=[0.411,0.415,0.415,0.392,0.423],
                 )
    #
    # # energy_station
    # compare_plot(xlabel="No. of charging stations",
    #              ylabel="Energy usage (# of full batteries)",
    #              x=[1, 2, 3, 4, 5],
    #              yrange=[0,5],
    #              eDivert=[1.993,3.537,3.844,3.773,4],
    #              woApeX=[2.092,3.135,3.855,4.383,4.273],
    #              woRNN=[2.09,3.041,3.184,3.05,3.98],
    #              MADDPG=[2.016,2.203,2.264,2.473,2.693],
    #              )

    # efficiency_station
    compare_plot(xlabel="No. of charging stations",
                 ylabel="Energy efficiency",
                 x=[1, 2, 3, 4, 5],
                 yrange=[-0.04,0.2],
                 eDivert=[0.138,0.177,0.180,0.181,0.181],
                 woApeX=[0.093,0.128,0.142,0.148,0.150],
                 woRNN=[0.101,0.126,0.132,0.119,0.104],
                 MADDPG=[0.063,0.055,0.048,0.047,0.048],
                 )




