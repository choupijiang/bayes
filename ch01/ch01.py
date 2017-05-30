# -*- coding:utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

from IPython.core.pylabtools import figsize
import numpy as np
from matplotlib import pyplot as plt


## first case of bayesian
# figsize(12.5, 4)
# plt.rcParams['savefig.dpi'] = 300
# plt.rcParams['figure.dpi'] = 300
#
# colors = ["#348ABD", "#A60628"]
# prior = [1/21., 20/21.]
# posterior = [0.087, 1-0.087]
# plt.bar([0, .7], prior, alpha=0.7, width=0.25, color=colors[0], label="prior distribution", lw="3", edgecolor="#348ABD")
# plt.bar([0+0.25, .7+0.25], posterior, alpha=0.7, width=0.25, color=colors[1], label="posterior distribution", lw="3", edgecolor="#A60628")
#
# plt.xticks([0.20, 0.95], ["Libratian", "Farmer"])
# plt.title("Prior and posterior probabilities of Steve's occupation")
#
# plt.ylabel("Probability")
# plt.legend(loc="upper left")
# plt.show()

# #poisson distribution
# figsize(12.5, 4)
# import scipy.stats as stats
#
# a = np.arange(20)
# poi = stats.poisson
#
# lambda_ = [1.5, 4.25]
# colors = ["#348ABD", "#A60628"]
#
# plt.bar(a, poi.pmf(a, lambda_[0]), color=colors[0], label="$\lambda = %.1f$" % lambda_[0], alpha=0.6, edgecolor=colors[0], lw="3")
# plt.bar(a, poi.pmf(a, lambda_[1]), color=colors[1], label="$\lambda = %.1f$" % lambda_[1], alpha=0.6, edgecolor=colors[1], lw="3")
#
# plt.xticks(a+0.4, a)
# plt.legend()
# plt.ylabel("Probability of $k$")
# plt.xlabel("$k$")
# plt.title("Probability nass function of a Poisson random variable, differing $\lambda$ values")
# plt.show()


figsize(12.5, 3.5)
count_data = np.loadtxt("txtdata.csv")
n_count_data = len(count_data)
# plt.bar(np.arange(n_count_data), count_data, color="#348ABD")
# plt.xlabel("Time (days)")
# plt.ylabel("Text messages received")
# plt.title("Did the user's texting habits change over time?")
# plt.xlim(0, n_count_data)
# plt.show()

import pymc as pm
alpha = 1.0/count_data.mean()

lambda_1 = pm.Exponential("lambda_1", alpha)
lambda_2 = pm.Exponential("lambda_2", alpha)
print lambda_1, lambda_2
tau = pm.DiscreteUniform("tau", lower=0, upper=n_count_data)
print "Random output:", tau.random(), tau.random(), tau.random()

@pm.deterministic
def lambda_(tau=tau, lambda_1=lambda_1, lambda_2=lambda_2):
    out = np.zeros(n_count_data)
    out[:tau] = lambda_1
    out[tau:] = lambda_2
    return out

print type(lambda_1), type(lambda_)
observation = pm.Poisson("obs", lambda_, value=count_data, observed=True)
model = pm.Model([observation, lambda_1, lambda_2, tau])
print model

mcmc = pm.MCMC(model)
mcmc.sample(40000, 10000)

lambda_1_samples = mcmc.trace("lambda_1")[:]
lambda_2_samples = mcmc.trace("lambda_2")[:]
tau_samples = mcmc.trace("tau")[:]

print "----",lambda_1_samples.size, lambda_2_samples.size, tau_samples.size

figsize(14.5, 10)
ax = plt.subplot(311)
ax.set_autoscaley_on(False)
plt.hist(lambda_1_samples, histtype="stepfilled", bins=30, alpha=.85, label="posterior of $\lambda_1$", color="#A60628", normed=True)
plt.legend(loc="upper left")
plt.title(r"""Posterior distributions of the parameters $\lambda_1,\; \lambda_2,\; \tau$""")
plt.xlim([15, 30])
plt.xlabel("$\lambda_1$ value")
plt.ylabel("Density")

ax = plt.subplot(312)
ax.set_autoscaley_on(False)
plt.hist(lambda_2_samples, histtype="stepfilled", bins=30, alpha=.85, label="posterior of $\lambda_2$", color="#7A68A6", normed=True)
plt.legend(loc="upper left")
plt.xlim([15, 30])
plt.xlabel("$\lambda_2$ value")
plt.ylabel("Density")

plt.subplot(313)
w = 1.0 / tau_samples.shape[0] * np.ones_like(tau_samples)
plt.hist(tau_samples, bins=n_count_data, alpha=1, label=r"posterior of $\tau$", color="#467821", weights=w, rwidth=2)
plt.xticks(np.arange(n_count_data))
plt.legend(loc="upper left")
plt.ylim([0, .75])
plt.xlim([35, len(count_data) - 20])
plt.xlabel(r"$\tau$ (in days)")
plt.ylabel("Probability")

plt.show()