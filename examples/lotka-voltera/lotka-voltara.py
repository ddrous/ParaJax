# %%

import torch


## Load the tensor in data/lotka-voltera.pt
data = torch.load("data/model_ind.pt", map_location=torch.device('cpu'))

data
# %%
