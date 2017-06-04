# -*- coding:utf-8 -*-
import sys

reload(sys)
sys.setdefaultencoding('utf-8')

import pymc as pm
from scipy import stats
import numpy as np
from IPython.core.pylabtools import figsize
from matplotlib import pyplot as plt
#
# lambda_ = pm.Exponential("poisson_param", 1)
# data_generator = pm.Poisson("data_generator", lambda_)
# data_plus_one = data_generator + 1
#
# print "Children of 'lambda_':"
# print lambda_.children
#
# print "\n Parents of 'data_generator':"
# print data_generator.parents
#
# print "\n Children of 'data_generator':"
# print data_generator.children
#
# print "lambda_.value =", lambda_.value
# print "data_generator.value=", data_generator.value
# print "data_plus_one.value=", data_plus_one.value
#
# lambda_1 = pm.Exponential("lambda_1", 1)
# lambda_2 = pm.Exponential("lambda_2", 1)
# tau = pm.DiscreteUniform("tau", lower=0, upper=10)
#
# print "Initialized value..."
# print "lambda_1.value: %.3f" % lambda_1.value
# print "lambda_2.value: %.3f" % lambda_2.value
# print "tau.value: %.3f" % tau.value
# print
#
# lambda_1.random(), lambda_2.random(), tau.random()
#
# print "After calling random() on the variables..."
# print "lambda_1.value: %.3f" % lambda_1.value
# print "lambda_2.value: %.3f" % lambda_2.value
# print "tau.value: %.3f" % tau.value
#
# import numpy as np
#
# n_data_points = 5
#
#
# @pm.deterministic
# def lambda_(tau=tau, lambda_1=lambda_1, lambda_2=lambda_2):
#     out = np.zeros(n_data_points)
#     out[:tau] = lambda_1
#     out[tau:] = lambda_2
#     return out
#
#
#
# figsize(12.5, 4)
# plt.rcParams['savefig.dpi'] = 100
# plt.rcParams['figure.dpi'] = 100
# samples = [lambda_1.random() for i in range(20000)]
# plt.hist(samples, bins=70, normed=True, histtype="stepfilled")
# plt.title("Prior distribution for $\lambda_1$")
# plt.xlabel("Value")
# plt.ylabel("Density")
# plt.xlim(0, 8)
# plt.show()
#
# data = np.array([10, 5])
# fixed_variable = pm.Poisson("fxd", 1, value=data, observed=True)
# print "value:", fixed_variable.value
# print "calling .random()"
# fixed_variable.random()
# print "value:", fixed_variable.value
#
# data = np.array([10, 25, 15, 20, 35])
# obs = pm.Poisson("obs", lambda_, value=data, observed=True)
# print obs.value
# model = pm.Model([obs, lambda_, lambda_1, lambda_2, tau])


###=============================================================######

# tau = pm.rdiscrete_uniform(0, 80)
# print tau
#
# alpha = 1./20.
# lambda_1, lambda_2 = pm.rexponential(alpha, 2)
# print lambda_1, lambda_2
# lambda_ = np.r_[lambda_1*np.ones(tau), lambda_2*np.ones(80-tau)]
# print lambda_
# data = pm.rpoisson(lambda_)
# print data




# def plot_artificial_sms_dataset():
#     tau = pm.rdiscrete_uniform(0, 80)
#     alpha = 1. / 20.
#     lambda_1, lambda_2 = pm.rexponential(alpha, 2)
#     data = np.r_[pm.rpoisson(lambda_1, tau), pm.rpoisson(lambda_2, 80 - tau)]
#     plt.bar(np.arange(80), data, color="#348ABD")
#     plt.bar(tau - 1, data[tau - 1], color="r", label="user behavior changed")
#     # plt.xlabel("Time (days)")
#     # plt.ylabel("Text message received")
#
# figsize(12.5, 5)
# plt.title("Artificial dataset from simulating the model")
#
# for i in range(4):
#     plt.subplot(4, 1, i+1)
#     plot_artificial_sms_dataset()
#
# plt.xlabel("Time (days)")
# plt.ylabel("Text messages received")
#
# plt.show()


###=============================================================######
#
# p = pm.Uniform('p', lower=0, upper=1)
# p_true = 0.05
# N = 1500
# occurrences = pm.rbernoulli(p_true, N)
# print occurrences
# print occurrences.sum()
#
# # Occurrences.mean()  is equal to n/N
# print "What is the observed freq in Group A? %.4f"   % occurrences.mean()
#
# print "Does the observed freq equal the true freq? %s" % (occurrences.mean() == p_true)
#
# obs = pm.Bernoulli("obs", p, value=occurrences, observed=True)
#
# mcmc = pm.MCMC([p, obs])
# mcmc.sample(20000, 1000)
#
# figsize(12.5, 4)
# plt.title("Posterior distribution of $p_A$, the true effectiveness of site A")
# plt.vlines(p_true, 0, 90, linestyles="--", label="true $p_A$ (unknown)")
# plt.hist(mcmc.trace("p")[:], bins=35, histtype="stepfilled", normed=True)
# plt.xlabel("Value of $p_A$")
# plt.ylabel("Density")
# plt.legend()
# plt.show()
#
#
# figsize(12, 4)
# true_p_A = 0.05
# true_p_B = 0.04
# N_A = 1500
# N_B = 750
# observations_A = pm.rbernoulli(true_p_A, N_A)
# observations_B = pm.rbernoulli(true_p_B, N_B)
# print "Obs from site A:", observations_A[:30].astype(int), "..."
# print "Obs from site B:", observations_B[:30].astype(int), "..."
# print observations_A.mean()
# print observations_B.mean()
# p_A = pm.Uniform("p_A", 0, 1)
# p_B = pm.Uniform("p_B", 0, 1)
#
# @pm.deterministic
# def delta(p_A=p_A, p_B=p_B):
#     return p_A - p_B
#
# obs_A = pm.Bernoulli("obs_A", p_A, value=observations_A, observed=True)
# obs_B = pm.Bernoulli("obs_B", p_B, value=observations_B, observed=True)
#
# mcmc = pm.MCMC([p_A, p_B, delta, obs_A, obs_B])
# mcmc.sample(25000, 5000)
#
# p_A_samples = mcmc.trace("p_A")[:]
# p_B_samples = mcmc.trace("p_B")[:]
# delta_samples = mcmc.trace("delta")[:]
#
# figsize(12.5, 10)
# ax = plt.subplot(311)
# plt.xlim(0, 0.1)
# plt.hist(p_A_samples, histtype="stepfilled", bins=30, alpha=0.85, label="posterior pf $p_A$", color="#A60628", normed=True)
# plt.vlines(true_p_A, 0, 80, linestyles="--", label="true $p_A$ (unknown)")
# plt.legend(loc="upper right")
# plt.title("Posterior distributions of $p_A$, $p_B$, and delta unkowns")
# plt.ylim(0, 80)
#
# ax = plt.subplot(312)
# plt.xlim(0, 0.1)
# plt.hist(p_B_samples, histtype="stepfilled", bins=30, alpha=0.85, label="posterior pf $p_B$", color="#467821", normed=True)
# plt.vlines(true_p_A, 0, 80, linestyles="--", label="true $p_B$ (unknown)")
# plt.legend(loc="upper right")
# plt.ylim(0, 80)
#
# ax = plt.subplot(313)
# plt.xlim(-0.1, 0.1)
# plt.hist(delta_samples, histtype="stepfilled", bins=30, alpha=0.85, label="posterior pf delta", color="#7A68A6", normed=True)
# plt.vlines(true_p_A - true_p_B, 0, 80, linestyles="--", label="true delta (unknown)")
# plt.vlines(0, 0, 60, colors="black", alpha=0.2)
# plt.legend(loc="upper right")
# plt.ylim(0, 80)
#
# plt.show()
#
# figsize(12.5, 3)
# plt.xlim(0, 0.1)
# plt.hist(p_A_samples, histtype="stepfilled", bins=30, alpha=0.85, label="posterior pf $p_A$", color="#A60628", normed=True)
# plt.hist(p_B_samples, histtype="stepfilled", bins=30, alpha=0.85, label="posterior pf $p_B$", color="#467821", normed=True)
# plt.legend(loc="upper right")
# plt.xlabel("Value")
# plt.ylabel("Density")
# plt.title("Posterior distributions of $p_A$, $p_B$")
# plt.ylim(0, 80)
# plt.show()

# ###====================================================================##
#
# figsize(12.5, 4)
# binomial = stats.binom
# params = [(1000, .4), (1000, .9)]
# colors = ["#348ABD", "#A60628"]
#
# for i in range(2):
#     N, p = params[i]
#     _x = np.arange(N + 1)
#     plt.bar(_x - 0.5, binomial.pmf(_x, N, p), color=colors[i], edgecolor=colors[i], alpha=.6, label="$N$: %d, $p$: %.1f" % (N, p), linewidth=3)
#
# plt.legend(loc="upper left")
# plt.xlim(0, 1000.5)
# plt.xlabel("$k$")
# plt.ylabel("$P(X=k)$")
# plt.title("Probability mass distributions of binomial random variables")
# plt.show()

###====================================================================##

# N = 100
# p = pm.Uniform("freq_cheating", 0, 1)
# true_answers = pm.Bernoulli("truths", p, size=N)
# first_coin_flips = pm.Bernoulli("first_flips", .5, size=N)
# # print first_coin_flips.value
#
# second_coin_flips = pm.Bernoulli("second_flips", .5, size=N)
#
# @pm.deterministic
# def observed_proportion(t_a=true_answers, fc=first_coin_flips, sc=second_coin_flips):
#
#     observed = fc * t_a + (1 - fc) * sc
#     return observed.sum() / float(N)
#
# print observed_proportion.value
#
# X = 35
# observations = pm.Binomial("obs", N, observed_proportion, observed=True, value=X)
# print observations.value
#
# model = pm.Model([p, true_answers, first_coin_flips, second_coin_flips, observed_proportion, observations])
# mcmc = pm.MCMC(model)
# mcmc.sample(40000,15000)
# figsize(12.5, 3)
# p_trace = mcmc.trace("freq_cheating")[:]
# print p_trace, p_trace.size
# plt.hist(p_trace, histtype="stepfilled", normed=True, alpha=.85, bins=30, label="posterior distribution", color="#348ABD")
# plt.vlines([.05, .35], [0, 0], [5, 5], alpha=.3)
# plt.xlim(0, 1)
# plt.xlabel("Value of $p$")
# plt.ylabel("Density")
# plt.title("Posterior distribution of parameter $p$")
# plt.legend()
# plt.show()


#--------------------------------------------#
#
# p = pm.Uniform("freq_cheating", 0, 1)
#
# @pm.deterministic
# def p_skewed(p=p):
#     return .5*p + .25
#
# yes_responses = pm.Binomial("number_cheaters", 100, p_skewed, value=35, observed=True)
# print yes_responses.value
# model = pm.Model([p, p_skewed, yes_responses])
#
# mcmc = pm.MCMC(model)
#
# mcmc.sample(25000, 2500)
# figsize(12.5, 3)
# p_trace = mcmc.trace("freq_cheating")[:]
# print p_trace, p_trace.size
# plt.hist(p_trace, histtype="stepfilled", normed=True, alpha=.85, bins=30, label="posterior distribution", color="#348ABD")
# plt.vlines([.05, .35], [0, 0], [5, 5], alpha=.3)
# plt.xlim(0, 1)
# plt.xlabel("Value of $p$")
# plt.ylabel("Density")
# plt.title("Posterior distribution of parameter $p$")
# plt.legend()
# plt.show()

###====================================================================##


def logistic(x, beta, alpha=0):
    return 1.0 / (1.0 + np.exp(np.dot(beta, x) + alpha))


np.set_printoptions(precision=3, suppress=True)
challenger_data = np.genfromtxt("challenger_data.csv", skip_header=1, usecols=[1, 2], missing_values="NA", delimiter=",")

challenger_data= challenger_data[~np.isnan(challenger_data[:, 1])]
# print challenger_data

# figsize(12.5, 3.5)
# plt.scatter(challenger_data[:, 0], challenger_data[:, 1], s=75, color="k", alpha=.75)
# plt.yticks([0, 1])
# plt.ylabel("Damage incident")
# plt.xlabel("Outside temp ")
# plt.title("Defects of the space shuttle O-rings versus temp")
# plt.show()

import pymc as pm
temprature = challenger_data[:, 0]
D = challenger_data[:, 1]

beta = pm.Normal("beta", 0, 0.001, value=0)
alpha = pm.Normal("alpha", 0, 0.001, value=0)


@pm.deterministic
def p(t=temprature, alpha=alpha, beta=beta):
    return 1.0/(1. + np.exp(beta*t + alpha))
# print p.value

observed = pm.Bernoulli("bernoulli_obs", p, value=D, observed=True)
model = pm.Model([observed, beta, alpha])
# print model
map_ = pm.MAP(model)
map_.fit()
mcmc = pm.MCMC(model)
mcmc.sample(120000, 100000, 2)

alpha_samples = mcmc.trace('alpha')[:, None]
# print alpha_samples
beta_samples = mcmc.trace('beta')[:, None]
# print beta_samples

# figsize(12.5, 6)
#
# plt.subplot(211)
# plt.title("Posterior distributions of the model parameters  alpha , beta")
# plt.hist(beta_samples, histtype="stepfilled", bins=35, alpha=.85, label="posterior of  beta  " , color="#7A68A6", normed=True)
# plt.legend()
#
# plt.subplot(212)
# plt.hist(alpha_samples, histtype="stepfilled", bins=35, alpha=.85, label="posterior of alpha ", color="#A60638", normed=True)
# plt.xlabel("Value pf params")
# plt.ylabel("Density")
# plt.legend()
# plt.show()

t = np.linspace(temprature.min() - 5, temprature.max() + 5, 50)[:, None]
p_t = logistic(t.T, beta_samples, alpha_samples)
mean_prob_t = p_t.mean(axis=0)
# print mean_prob_t.shape, mean_prob_t
# figsize(12.5, 4)
# plt.plot(t, mean_prob_t, lw=3, label="average posterior \nprobability of defect")
# plt.plot(t, p_t[0, :], ls="--", label="realization from posterior")
# plt.plot(t, p_t[-2, :], ls="--", label="realization from posterior")
# plt.scatter(temprature, D, color="k", s=50, alpha=.5)
# plt.title("Posterior expected value of the prob of defect, including two realizations")
# plt.legend(loc="upper left")
# plt.ylim(-0.1, 1.1)
# plt.xlim(t.min(), t.max())
# plt.ylabel("Prob")
# plt.xlabel("Temperature")
# plt.show()

# from scipy.stats.mstats import mquantiles
# qs = mquantiles(p_t, [0.025, 0.975], axis=0)
# plt.fill_between(t[:, 0], *qs, alpha=0.7, color="#7A68A6")
# plt.plot(t, qs[0], label="95% CI", color="#7A68A6", alpha=.7)
# plt.plot(t, mean_prob_t, lw=1, ls="--", color="k", label="avaerage posterior probability of defect")
# plt.xlim(t.min(), t.max())
# plt.ylim(-0.02, 1.02)
# plt.legend(loc="lower left")
# plt.scatter(temprature, D, color="k", s=50, alpha=.5)
# plt.ylabel("Prob")
# plt.xlabel("Temperature")
# plt.title("Posterior prob of estimates, given temperature $t$")

# figsize(12.5, 2.5)
# prob_31 = logistic(31, beta_samples, alpha_samples)
# plt.xlim(0.995, 1)
#
# plt.hist(prob_31, bins=1000, normed=True, histtype="stepfilled")
# plt.title("posterior distribution of probability of defect, given $t = 31$")
# plt.ylabel("Density")
# plt.xlabel("Prob of defect occurring in O-ring")
# plt.show()
###################################################################

###模拟
simulated_data = pm.Bernoulli("simulation_data", p)

simulated = pm.Bernoulli("bernoulli_sim", p)
N = 10000
mcmc = pm.MCMC([simulated, alpha, beta, observed])
mcmc.sample(N)

figsize(12.5, 5)
simulations = mcmc.trace("bernoulli_sim")[:].astype(int)
print "Shape of simulations array:", simulations.shape
# figsize(12.5, 6)
# for i in range(4):
#     ax = plt.subplot(4, 1, i+1)
#     print simulations[1000*i, :]
#     plt.scatter(temprature, simulations[1000*i, :], color="k", s=50, alpha=.6)
# plt.show()

###########################################################################################

##分离图

posterior_probability = simulations.mean(axis=0)
print posterior_probability

print "Obs.| Array of simulated Defects     | Posterior Prob of Defect | Realized Defect."

for i in range(len(D)):
    print "%s  | %s |  %.2f               | %d  " %(str(i).zfill(2), str(simulations[:10, i])[:-1] + "...]".ljust(12), posterior_probability[i] ,D[i])


ix = np.argsort(posterior_probability)
print ix
print "Posterior Probability of Defect | Realized Defect"
for i in range(len(D)):
    print "%.2f                            |   %d" % (posterior_probability[ix[i]], D[ix[i]])

