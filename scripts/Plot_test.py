# coding: utf-8

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

#STAT_DIR = '../stats/'
#models = [f[:-4] for f in os.listdir(STAT_DIR) if f.endswith(".npy")]
STAT_DIR = '../scripts/'
models = ['test_L5U150', 'test_L4U200', 'test_L3U150']
num_steps = 75
colors = ['#204594', '#95B333', '#FD9D59', '#F585A5', '#CDB460', '#3AB5D4', '#B72220', '#FEE30E', '#319848', '#5f1250', '#e7221a']

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111)
for modelname, colorname in zip(models, colors):
    stats = np.load(STAT_DIR+modelname+".npy")
    ax.plot(range(num_steps), stats[:,0], label=modelname+"_loss(train)", color=colorname, alpha=0.5, linestyle='--')
    ax.plot(range(num_steps), stats[:,3], label=modelname+"_loss(test)", color=colorname)
lgd = ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=2, mode="expand", borderaxespad=0)
ax.set_xlim(0, num_steps)
ax.set_xlabel("step")
ax.set_ylabel("loss value")
ax.grid()
fig.tight_layout()
fig.savefig("loss_test.pdf", bbox_extra_artists=(fig,), bbox_inches='tight')

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111)
for modelname, colorname in zip(models, colors):
    stats = np.load(STAT_DIR+modelname+".npy")
    ax.plot(range(num_steps), stats[:,1], label=modelname+"_acc(train)", color=colorname, alpha=0.5, linestyle='--')
    ax.plot(range(num_steps), stats[:,4], label=modelname+"_acc(test)", color=colorname)
lgd = ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=2, mode="expand", borderaxespad=0)
ax.set_xlim(0, num_steps)
ax.set_xlabel("step")
ax.set_ylabel("accurucy value")
ax.grid()
fig.tight_layout()
fig.savefig("acc_test.pdf", bbox_extra_artists=(fig,), bbox_inches='tight')

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111)
for modelname, colorname in zip(models, colors):
    stats = np.load(STAT_DIR+modelname+".npy")
    ax.plot(range(num_steps), stats[:,2], label=modelname+"_f1(train)", color=colorname, alpha=0.5, linestyle='--')
    ax.plot(range(num_steps), stats[:,5], label=modelname+"_f1(test)", color=colorname)
lgd = ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=2, mode="expand", borderaxespad=0)
ax.set_xlim(0, num_steps)
ax.set_xlabel("step")
ax.set_ylabel("f1 score")
ax.grid()
fig.tight_layout()
fig.savefig("f1_test.pdf", bbox_extra_artists=(fig,), bbox_inches='tight')
