#%%
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Define the Dahlquist ODE
def dahlquist(t, y):
    return -5 * y

# Set the initial conditions
t_span = (0, 1)
y0 = [1.0]
t_eval = np.linspace(0, 1, 10001)

# Solve the ODE using solve_ivp
sol = solve_ivp(dahlquist, t_span, y0, method='RK45', t_eval=t_eval)

# Plot the solution
plt.plot(sol.t, sol.y[0])
plt.xlabel('Time')
plt.ylabel('y(t)')
plt.title('Dahlquist Test Problem')
plt.show()

## Save t_eval and the solutuin to a npz file
np.savez('data/dahlquist_n5.npz', t=t_eval, X=sol.y)

# %%
