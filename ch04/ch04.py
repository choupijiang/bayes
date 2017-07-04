import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
from IPython.core.pylabtools import figsize

plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.dpi'] = 300

# figsize(10, 4)

#
# sample_size = 100000
#
# expected_value = lambda_ = 4.5
# poi = pm.rpoisson
# N_samples = range(1, sample_size, 100)
#
# for k in range(3):
#     samples = poi(lambda_, size=sample_size)
#     partial_average = [samples[:i].mean() for i in N_samples]
#     plt.plot(N_samples, partial_average, lw=1.5, label="average of $n$ samples; seq. %d" % k)
# plt.plot(N_samples, expected_value*np.ones_like(partial_average), ls="--", label="true expected value", c="k")
# plt.ylim(4.35, 6.45)
# plt.title("Convergence of the average of \n random variables to their expected value")
# plt.xlabel("Average of $n$ samples")
# plt.ylabel("Number of samples, $n$")
# plt.legend()
# plt.show()
#
# figsize(12.5, 4)
# N_Y = 250
# N_array = np.arange(1000, 50000, 2500)
# D_N_results = np.zeros(len(N_array))
# lambda_ = 4.5
# expected_value = lambda_
# poi = pm.rpoisson
#
# def D_N(n):
#     Z = poi(lambda_, size=(n, N_Y))
#     average_Z = Z.mean(axis=0)
#     return np.sqrt(((average_Z - expected_value)**2).mean())
#
# for i,n in enumerate(N_array):
#     D_N_results[i] = D_N(n)
#
# plt.xlabel('$N$')
# plt.ylabel("Expected squared-distance from the value")
# plt.plot(N_array, D_N_results, lw=3, label="expected value and average of $N$ randfom variables")
# plt.plot(N_array, np.sqrt(expected_value)/np.sqrt(N_array), lw=2, ls="--", label=r"$\frac{\sqrt{\lambda}}{\sqrt{N}}$")
# plt.legend()
# plt.title("How 'quickly' is the same average converging ")
# plt.show()
#
# figsize(12.5, 4)
# std_height = 15
# mean_height = 150
# n_counties = 5000
# pop_generator = pm.rdiscrete_uniform
# norm = pm.rnormal
#
# population = pop_generator(100, 1500, size=n_counties)
#
# average_across_county = np.zeros(n_counties)
#
# for i in range(n_counties):
#     average_across_county[i] = norm(mean_height, 1./std_height**2, size=population[i]).mean()
#
# i_min = np.argmin(average_across_county)
# i_max = np.argmax(average_across_county)
#
# plt.scatter(population, average_across_county, alpha=0.5, c="#7A68A6", s=30)
# plt.scatter([population[i_min], population[i_max]],
#             [average_across_county[i_min], average_across_county[i_max]],
#             s=30,
#             marker="o",
#             facecolors="none",
#             edgecolors="#A60628",
#             linewidths=1.5,
#             label="extreme heights"
#             )
#
# plt.xlim(100, 1500)
# plt.title("Average height versus county population")
# plt.xlabel("Country population")
# plt.ylabel("Average height in county")
# plt.plot([100, 1500], [150, 150], color="k", label="true expected height", ls="--")
# plt.legend(scatterpoints = 1)
# plt.show()
#
# figsize(12.5, 6.5)
# data = np.genfromtxt("census_data.csv", skip_header=1, delimiter=",")
# plt.scatter(data[:, 1], data[:, 0], alpha=.5, c="#7A68A6")
# plt.title("census mail-back rate versus population")
# plt.ylabel("Mail-back rate")
# plt.xlabel("Population of block group")
# plt.xlim(-100, 15e3)
# plt.ylim(-5, 105)
# i_min = np.argmin(data[:, 0])
# i_max = np.argmax(data[:, 0])
#
# plt.scatter(
#     [data[i_min, 1], data[i_max, 1]],
#     [data[i_min, 0], data[i_max, 0]],
#     s=60,
#     marker="o",
#     facecolors="none",
#     edgecolors="#A60628",
#     linewidths=1.5,
#     label="most extreme points"
# )
# plt.legend(scatterpoints = 1)
# plt.show()

n_comments = len(contents)