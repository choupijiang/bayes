# -*- coding:utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import pymc as pm
lambda_ = pm.Exponential("poisson_param", 1)
data_generator = pm.Poisson("data_generator", lambda_)
data_plus_one = data_generator + 1

print "Children of 'lambda_':"
print lambda_.children

print "\n Parents of 'data_generator':"
print data_generator.parents

print "\n Children of 'data_generator':"
print data_generator.children

print "lambda_.value =", lambda_.value
print "data_generator.value=", data_generator.value
print "data_plus_one.value=", data_plus_one.value


lambda_1 = pm.Exponential("lambda_1", 1)
lambda_2 = pm.Exponential("lambda_2", 2)
tau = pm.DiscreteUniform("tau", lower=0, upper=10)

print "Initialized value..."
print "lambda_1.value: %.3f" % lambda_1.value
print "lambda_2.value: %.3f" % lambda_2.value
print "tau.value: %.3f" % tau.value
print

lambda_1.random(), lambda_2.random(), tau.random()

print "After calling random() on the variables..."
print "lambda_1.value: %.3f" % lambda_1.value
print "lambda_2.value: %.3f" % lambda_2.value
print "tau.value: %.3f" % tau.value

import numpy as np

n_data_points = 5

@pm.deterministic
def lambda_(tau=tau, lambda_1=lambda_1, lambda_2=lambda_2):
    out = np.zeros(n_data_points)
    out[:tau] = lambda_1
    out[tau:] = lambda_2
    return out

from IPython.core.pylabtools import figsize
from matplotlib import pyplot as plt
figsize(12.5, 4)
plt.rcParams['savefig.dpi'] = 100
plt.rcParams['figure.dpi'] = 100
samples = [lambda_1.random() for i in range(20000)]
plt.hist(samples, bins=70, normed=True, histtype="stepfilled")
plt.title("Prior distribution for $\lambda_1$")
plt.xlabel("Value")
plt.ylabel("Density")
plt.xlim(0, 8)
plt.show()