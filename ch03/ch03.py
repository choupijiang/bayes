# -*- coding:utf-8 -*-
import sys

reload(sys)
sys.setdefaultencoding('utf-8')

import scipy.stats as stats
from matplotlib import pyplot as plt
from IPython.core.pylabtools import figsize
import numpy as np

##################################################################
## 均匀分布

figsize(12.5, 4)
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.dpi'] = 300

from mpl_toolkits.mplot3d import Axes3D
#
# jet = plt.cm.jet
# fig = plt.figure()
#
# x = y = np.linspace(0, 5, 100)
# X, Y = np.meshgrid(x, y)
# plt.subplot(121)
#
# uni_x = stats.uniform.pdf(x, loc=0, scale=5)
# uni_y = stats.uniform.pdf(y, loc=0, scale=5)
#
# # print uni_x[:, None], uni_y[None, :]
# M = np.dot(uni_x[:, None], uni_y[None, :])
# im = plt.imshow(M, interpolation='none', origin='lower', cmap=jet, vmax=1, vmin=-.15, extent=(0, 5, 0, 5))
# plt.xlim(0, 5)
# plt.ylim(0, 5)
# plt.title("Overhead view of landscape formed by Uniform priors")
#
# ax = fig.add_subplot(122, projection='3d')
# ax.plot_surface(X, Y, M, cmap=plt.cm.jet, vmax=1, vmin=-0.15)
# ax.view_init(azim=390)
# ax.set_xlabel('Value of $p_1$')
# ax.set_ylabel('Value of $p_2$')
# ax.set_zlabel('Density')
# plt.title("Alternate view of landscape formed by Uniform priors")
# plt.show()

##################################################################

# figsize(12.5, 5)
# fig = plt.figure()
# plt.subplot(121)
#
# x = y = np.linspace(0, 5, 100)
# X, Y = np.meshgrid(x, y)
# exp_x = stats.expon.pdf(x, scale=3)
# exp_y = stats.expon.pdf(y, scale=10)
#
# M = np.dot(exp_x[:, None], exp_y[None, :])
# CS = plt.contour(X, Y, M)
#
# im = plt.imshow(M, interpolation='none', origin='lower', cmap=plt.cm.jet, extent=(0, 5, 0, 5))
# plt.title("Overhead view of landscape formed by $Exp(3), Exp(10)$ priors")
#
# ax = fig.add_subplot(122, projection='3d')
# ax.plot_surface(X, Y, M, cmap=plt.cm.jet)
# ax.view_init(azim=390)
# ax.set_xlabel("Value of $p_1$")
# ax.set_ylabel("Value pf $p_2$")
# ax.set_zlabel("Density")
# plt.title("Alternative view of landscape formed by $Exp(3), Exp(10)$ priors")
# plt.show()


##################################################################
#
# N = 1
# lambda_1_true = 1
# lambda_2_true = 3
#
# data = np.concatenate([
#         stats.poisson.rvs(lambda_1_true, size=(N, 1)),
#         stats.poisson.rvs(lambda_2_true, size=(N, 1)),
#     ], axis=1
# )
#
# print "observed (2-dim, sample size=%d):" % N, data
#
# x = y = np.linspace(.01, 5, 100)
# likelihood_x = np.array([stats.poisson.pmf(data[:, 0], _x) for _x in x]).prod(axis=1)
# likelihood_y = np.array([stats.poisson.pmf(data[:, 1], _y) for _y in y]).prod(axis=1)
# print likelihood_x[:, None], likelihood_y[None, :]
# L = np.dot(likelihood_x[:, None], likelihood_y[None, :])
# print L.shape#, L[:10,:10]
#
# figsize(12, 8)
# plt.subplot(221)
# uni_x = stats.uniform.pdf(x, loc=0, scale=5)
# uni_y = stats.uniform.pdf(y, loc=0, scale=5)
#
# M = np.dot(uni_x[:, None], uni_y[None, :])
# print uni_x[:10], M
# im = plt.imshow(M, interpolation='none', origin="lower", cmap=plt.cm.jet, vmax=1, vmin=-.15, extent=(0, 5, 0, 5))
# plt.scatter(lambda_2_true, lambda_1_true, c="k", s=50, edgecolors="none")
# plt.xlim(0, 5)
# plt.ylim(0, 5)
# # plt.title("Landscape formed by uniform \n priors on $p_1,p_2$")
#
# plt.subplot(223)
# plt.contour(x, y, M * L)
# uni_x = stats.uniform.pdf(x, loc=0, scale=5)
# uni_y = stats.uniform.pdf(y, loc=0, scale=5)
# M = np.dot(uni_x[:, None], uni_y[None, :])
# im = plt.imshow(M * L , interpolation='none', origin="lower", cmap=plt.cm.jet,  extent=(0, 5, 0, 5))
# plt.scatter(lambda_2_true, lambda_1_true, c="k", s=50, edgecolors="none")
# plt.xlim(0, 5)
# plt.ylim(0, 5)
# # plt.title("Landscape warped by uniform \n priors on $p_1,p_2$")
#
#
# plt.subplot(222)
# exp_x = stats.expon.pdf(x, loc=0, scale=3)
# exp_y = stats.expon.pdf(y, loc=0, scale=10)
# M = np.dot(exp_x[:, None], exp_y[None, :])
#
# plt.contour(x, y, M)
# im = plt.imshow(M, interpolation='none', origin="lower", cmap=plt.cm.jet, extent=(0, 5, 0, 5))
# plt.scatter(lambda_2_true, lambda_1_true, c="k", s=50, edgecolors="none")
# plt.xlim(0, 5)
# plt.ylim(0, 5)
# # plt.title("Landscape formed by Exp \n priors on $p_1,p_2$")
#
# plt.subplot(224)
#
# plt.contour(x, y, M * L)
# im = plt.imshow(M * L , interpolation='none', origin="lower", cmap=plt.cm.jet,  extent=(0, 5, 0, 5))
# plt.scatter(lambda_2_true, lambda_1_true, c="k", s=50, edgecolors="none")
# plt.xlim(0, 5)
# plt.ylim(0, 5)
# # plt.title("Landscape warped by exp \n priors on $p_1,p_2$")
# plt.show()

##################################################################


figsize(12.5, 5)
data = np.loadtxt("mixture_data.csv", delimiter=",")
print data.shape[0], data[:3], "..."
# plt.hist(data, bins=20, color="k", histtype="stepfilled", alpha=.8)
# plt.title("Histogram of the dataset")
# plt.ylim([0, None])
# plt.xlabel("Value")
# plt.ylabel("Count")
# plt.show()

import pymc as pm
p = pm.Uniform("p", 0., 1.)
# print p.value

assignment = pm.Categorical("assignment", [p, 1-p], size=data.shape[0])
print "prior assignment, with p = %.2f:" % p.value
print assignment.value[:10], "..."
taus = 1.0 / pm.Uniform("stds", 0, 100, size=2) ** 2
print taus.value

taus = 1.0 / pm.Uniform("stds", 0, 33, size=2) ** 2
centers = pm.Normal("centers", [120, 190], [0.01, 0.01], size=2)
print "centers:", centers.value
@pm.deterministic
def center_i(assignment=assignment, centers=centers):
    return centers[assignment]

@pm.deterministic
def tau_i(assignment=assignment, taus=taus):
    return taus[assignment]

print "Random assignment: ", assignment.value[:5], "..."
print "Assigned center: ", center_i.value[:4], "..."
print "Assigned precision: ", tau_i.value[:4], "..."


observations = pm.Normal("obs", center_i, tau_i, value=data, observed=True)
model = pm.Model([p, assignment, taus, centers])
mcmc = pm.MCMC(model)

mcmc.sample(50000)


colors = ["#348ABD", "#A60628"]
center_trace = mcmc.trace("centers")[:]
if center_trace[-1, 0] < center_trace[-1, 1]:
    colors = ["#A60628", "#348ABD"]
# figsize(12.5, 9)
# plt.subplot(311)
# line_width = 1
# plt.plot(center_trace[:, 0], label="trace of center 0", c=colors[0], lw=line_width)
# plt.plot(center_trace[:, 1], label="trace of center 1", c=colors[1], lw=line_width)
# plt.title("Traces of unknown parameters")
# leg = plt.legend(loc="upper right")
# leg.get_frame().set_alpha(.7)
#
# plt.subplot(312)
std_trace = mcmc.trace('stds')[:]
# plt.plot(std_trace[:, 0], label="trace of standard deviation of cluster 0", c=colors[0], lw=line_width)
# plt.plot(std_trace[:, 1], label="trace of standard deviation of cluster 1", c=colors[1], lw=line_width)
# plt.legend(loc="upper left")
#
# plt.subplot(313)
# p_trace = mcmc.trace("p")[:]
# plt.plot(p_trace, label="$p$: frequency of assignment to cluster 0", color="#467821", lw=line_width)
# plt.xlabel("Steps")
# plt.ylim(0, 1)
# plt.ylabel("Value")
# plt.legend()
# plt.show()

# mcmc.sample(100000)
# figsize(12.5, 4)
# center_trace = mcmc.trace("centers", chain=1)[:]
# prev_center_trace = mcmc.trace("centers", chain=0)[:]
# line_width = 1
# x = np.arange(50000)
# plt.plot(x, prev_center_trace[:, 0], label="prev trace of center 0", lw=line_width, alpha=.4, c=colors[1])
# plt.plot(x, prev_center_trace[:, 1], label="prev trace of center 1", lw=line_width, alpha=.4, c=colors[0])
#
# x = np.arange(50000, 150000)
# plt.plot(x, center_trace[:, 0], label="new trace of center 0", lw=line_width, alpha=.7, c=colors[1])
# plt.plot(x, center_trace[:, 1], label="new trace of center 1", lw=line_width, alpha=.7, c=colors[0])
#
# plt.title("Traces of unknown center parameters after sampling 100,000 more times")
# leg = plt.legend(loc="upper right")
# leg.get_frame().set_alpha(.8)
# plt.ylabel("Value")
# plt.xlabel("Steps")
# plt.show()

# figsize(11.0, 4)
# std_trace = mcmc.trace('stds')[:]
#
# _i = [1, 2, 3, 4]
# for i in range(2):
#     plt.subplot(2, 2, _i[2*i])
#     # plt.title("Posterior distribution of center of cluster %d" % i)
#     plt.hist(center_trace[:, i], color=colors[i], bins=30, histtype="stepfilled")
#
#     plt.subplot(2, 2, _i[2*i + 1])
#     # plt.title("Posterior distribution of center of standard deviation of cluster %d" % i)
#     plt.hist(std_trace[:, i], color=colors[i], bins=30, histtype="stepfilled")
#     plt.ylabel("Density")
#     plt.xlabel("Value")
# plt.tight_layout()
# plt.show()
#
import matplotlib as mpl
figsize(12.5, 4.5)
# plt.cmap = mpl.colors.ListedColormap(colors)
# print mcmc.trace("assignment")[::400, np.argsort(data)]
# plt.imshow(mcmc.trace("assignment")[::400, np.argsort(data)], cmap=plt.cmap, aspect=.4, alpha=.9)
# plt.xticks(np.arange(0, data.shape[0], 40), ["%.2f" % s for s in np.sort(data)[::40]])
# plt.ylabel("Posterior sample")
# plt.xlabel("Value of $i$th data point")
# plt.title("Posterior labels of data points")
# plt.show()

# cmap = mpl.colors.LinearSegmentedColormap.from_list("BMH", colors)
# assign_trace = mcmc.trace("assignment")[:]
#
# plt.scatter(data, 1 - assign_trace.mean(axis=0), cmap=cmap, c=assign_trace.mean(axis=0), s=1)
# plt.ylim(-0.05, 1.05)
# plt.xlim(35, 300)
# plt.title("Prob of data point belonging to cluster 0")
# plt.ylabel("Prob")
# plt.xlabel("Value of data point")
# plt.show()

norm = stats.norm
x = np.linspace(20, 300, 500)
posterior_center_means = center_trace.mean(axis=0)
posterior_std_means = std_trace.mean(axis=0 )
posterior_p_means = mcmc.trace("p")[:].mean()

plt.hist(data, bins=20, histtype="step", normed=True, color="k", lw=2, label="Histogram of data")

y = posterior_p_means*norm.pdf(x, loc=posterior_center_means[0], scale=posterior_std_means[0])
plt.plot(x, y, label="cluster 0", lw=3)
plt.fill_between(x, y, color=colors[0], alpha=.3)

y = (1- posterior_p_means)*norm.pdf(x, loc=posterior_center_means[1], scale=posterior_std_means[1])
plt.plot(x, y, label="cluster 1", lw=3)
plt.fill_between(x, y, color=colors[1], alpha=.3)

plt.legend(loc="upper left")
plt.title("Visualizing clusters using posterior mean parameters")
# plt.show()

from pymc.Matplot import plot as mcplot
mcmc.sample(25000, 0, 10)
mcplot(mcmc.trace("centers", 2), common_scale=False)