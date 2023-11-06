

#%%[markdown]
# # Neural ODEs to learn multiple ODEs simulatnuously
#
# We have one encoder, multiple processors, one decoder. The processors work as follows:
#
# $$  \frac{dY}{dt} = \alpha_1 F_{\theta_1} + \alpha_2 F_{\theta_2} + ...+ \alpha_K F_{\theta_K} $$

#%%
import jax

# import os
# jax.config.update("jax_platform_name", "cpu")
# os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=4'
print("Available devices:", jax.devices())

import jax.numpy as jnp
import numpy as np

import pickle

from parajax.utils import sbplot


#%%

## Import pkl data from the /data folder

# Load data from a .pkl file
with open('data/ostrich2d_ref.pkl', 'rb') as file:
# with open('data/walker2d_reg_expert.pkl', 'rb') as file:
    loaded_data = pickle.load(file)

# Now you can use the loaded_data in your code
print(loaded_data.shape)


# %%

for i in range(loaded_data.shape[1]):
    sbplot(loaded_data[:,i], loaded_data[:,0], x_label='Time', y_label='Position', title='Ostrich 2D')

# %%
