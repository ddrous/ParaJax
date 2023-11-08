

#%%
import jax

# import os
# jax.config.update("jax_platform_name", "cpu")
# os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=4'
print("Available devices:", jax.devices())

from jax import config
config.update("jax_debug_nans", True)

import jax.numpy as jnp
import numpy as np
import equinox as eqx

from parajax.utils import *
from parajax.integrators import *

import optax
from functools import partial
import time

#%%

SEED = 27

## Dataset hps
window_size = 100

## Plotting hps 
plt_hor = 1000


#%%


sp = jnp.load('data/simple_pendulum.npz')
X1 = jnp.concatenate([sp['X'].T, sp['t'][:, None]], axis=-1)

ip = jnp.load('data/inverted_pendulum.npz')
X2 = jnp.concatenate([ip['X'].T, ip['t'][:, None]], axis=-1)

print("Datasets sizes:", X1.shape, X2.shape)

# %%


sp_to_plot = X1_raw[:plt_hor]
ax = sbplot(sp_to_plot[:, -1], sp_to_plot[:, 0], "--", x_label='Time', label=r'$\theta$', title='Simple Pendulum')
ax = sbplot(sp_to_plot[:, -1], sp_to_plot[:, 1], "--", x_label='Time', label=r'$\dot \theta$', ax=ax)

ip_to_plot = X2_raw[:plt_hor]
ax = sbplot(ip_to_plot[:, -1], ip_to_plot[:, 2], "--", x_label='Time', label=r'$\theta$', title='Inverted Pendulum')
ax = sbplot(ip_to_plot[:, -1], ip_to_plot[:, 3], "--", x_label='Time', label=r'$\dot \theta$', ax=ax)
ax = sbplot(ip_to_plot[:, -1], ip_to_plot[:, 0], "--", x_label='Time', label=r'$x$', ax=ax)
ax = sbplot(ip_to_plot[:, -1], ip_to_plot[:, 1], "--", x_label='Time', label=r'$\dot x$', ax=ax)



#%%
from pydmd import BOPDMD
from pydmd.plotter import plot_summary

# Build a bagging, optimized DMD (BOP-DMD) model.
dmd = BOPDMD(
    svd_rank=15,  # rank of the DMD fit
    num_trials=100,  # number of bagging trials to perform
    trial_size=0.5,  # use 50% of the total number of snapshots per trial
    eig_constraints={"imag", "conjugate_pairs"},  # constrain the eigenvalue structure
    varpro_opts_dict={"tol":0.2, "verbose":True},  # set variable projection parameters
)

# Fit the DMD model.
# X = (n, m) numpy array of time-varying snapshot data
# t = (m,) numpy array of times of data collection
X = X1[:, :-1].T
t = X1[:, -1]
dmd.fit(X, t)

# Display a summary of the DMD results.
plot_summary(dmd)
# %%

vars(dmd)

# %%
