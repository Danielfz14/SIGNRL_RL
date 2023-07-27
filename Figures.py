# -*- coding: utf-8 -*-
"""Figures.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1j82enTO7G9izLtHUOlyTnhsvpJ9SSEvE
"""

!pip install SciencePlots

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math
import pickle
from sklearn.metrics import mean_squared_error as MSE
from scipy.io import savemat
import scipy.stats as st
import texttable as tt
import latextable as lt
import scienceplots

plt.style.use(['ieee','no-latex'])

def cum_mean(arr):
    cum_sum = np.cumsum(arr, axis=0)
    for i in range(cum_sum.shape[0]):
        if i == 0:
            continue
        #print(cum_sum[i] / (i + 1))
        cum_sum[i] =  cum_sum[i] / (i + 1)
    return cum_sum

## Cumulative Reward Figure

rec2 = np.zeros([50, 500])
prom2 = np.zeros([50, 500])

for j in range(50):

  with open('recom_Carpole_'+str(j), "rb") as fp:
    recom=pickle.load(fp)
  rec=np.array(recom)
  m = np.argsort(np.mean(rec,axis=0))
  prom = np.zeros([30, 500])
  gra = np.mean(rec,axis=1)
  for i in range(30):
    prom[i] = cum_mean(rec[:,i])

  agente = np.argmin(np.min(prom,axis=1))
  rec2[j,:]=rec[:,agente]
  prom2[j,:]=prom[agente,:]


maxi = np.max(-1*rec2, axis=0)
mini = np.min(-1*rec2, axis=0)

with plt.style.context(['ieee','no-latex']):
    fig, ax1 = plt.subplots(1, 1, figsize = (4,2))
    x = np.arange(0, 500, 1)
    ax1.fill_between(x, maxi, mini, color='green', alpha=0.3)
    ax1.axis([0,500,-150,230])
    ax1.set_ylabel(r'$G_{\pi_{\psi}}$', fontsize=12)
    ax1.set_xlabel('Episode', fontsize=12)
    ax1.plot(-1*np.mean(rec2,axis=0),color='blue')
    fig.tight_layout()
    fig.subplots_adjust(left=0.15,
                    bottom=0.22,
                    right=0.95,
                    top=0.94,
                    wspace=0.05,
                    hspace=0.1)
    fig.savefig('cum_reward.pdf')

## Average Cumulative Reward Figure

maxi = np.max(-1*prom2, axis=0)
mini = np.min(-1*prom2, axis=0)

fig, ax1 = plt.subplots(1, 1, figsize = (4,2))
x = np.arange(0, 500, 1)
ax1.fill_between(x, maxi, mini, color='green', alpha=0.3)
ax1.axis([0,500,-150,230])
ax1.set_ylabel('ACR', fontsize=12)
ax1.set_xlabel('Episode', fontsize=12)
ax1.plot(-1*np.mean(prom2,axis=0), color='blue')

fig.tight_layout()
fig.subplots_adjust(left=0.15,
                    bottom=0.22,
                    right=0.95,
                    top=0.94,
                    wspace=0.05,
                    hspace=0.1)
fig.savefig('avg_reward.pdf')

# Best Agent Performance in Long Episode with Disturbances

agent='3'
with open('posicion_'+agent, "rb") as fp:
  pos=pickle.load(fp)
with open('angulo_'+agent, "rb") as fp:
  ang=pickle.load(fp)
with open('acciones_'+agent, "rb") as fp:
  acc=pickle.load(fp)
with open('perturbacion_'+agent, "rb") as fp:
  pert=pickle.load(fp)
fig, axs = plt.subplots(4, figsize=(4,5))

axs[0].axis([0,2000,-0.2,0.2])
axs[0].axes.xaxis.set_ticklabels([])
axs[0].set_ylabel(r'$x$ [m]', fontsize=12)
axs[1].axis([0,2000,-0.1,0.1])
axs[1].axes.xaxis.set_ticklabels([])
axs[1].set_ylabel(r'$\theta$ [rad]', fontsize=12)
axs[2].axis([0,2000,-20,20])
axs[2].axes.xaxis.set_ticklabels([])
axs[2].set_ylabel(r'$a$ [N]', fontsize=12)
axs[3].plot(pert,linewidth=0.5)
axs[3].axis([0,2000,-20,20])
axs[3].set_xlabel('Time-step', fontsize=12)
axs[3].set_ylabel(r'$df$ [N]', fontsize=12)
axs[3].ticklabel_format(axis='x', style='sci', scilimits=(3,3), useMathText=True)

for i in range(50):
  if (i==49):
    axs[0].plot(pos[i],linewidth=0.5,color='blue', alpha=0.9)
    axs[1].plot(ang[i],linewidth=0.5,color='blue', alpha=0.9)
    axs[2].plot(acc[i],linewidth=0.5,color='blue', alpha=0.9)
  else:
    axs[0].plot(pos[i],linewidth=0.5, alpha=0.9)
    axs[1].plot(ang[i],linewidth=0.5, alpha=0.9)
    axs[2].plot(acc[i],linewidth=0.5, alpha=0.9)

fig.tight_layout()
fig.subplots_adjust(left=0.16,
                    bottom=0.095,
                    right=0.95,
                    top=0.98,
                    wspace=0.1,
                    hspace=0.2)
fig.savefig('performance_conpert.pdf')