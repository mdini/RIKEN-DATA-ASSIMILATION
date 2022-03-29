import numpy as np
from scipy.integrate import odeint, solve_ivp
from random import random,seed


tmax, n = 100, 10000
# tmax, n = 100, 1000


def lorenz(x, t, sigma, beta, rho):
    """The Lorenz equations
    """
    u, v, w = x
    up = - sigma * (u - v)
    vp = rho * u - v - u * w
    wp = - beta * w + u * v
    return up, vp, wp
    
    
sigma, beta, rho = 10, 2.667, 28
seed(200)

for i in range(1000000):
        u0, v0, w0=random(),random(),random()
        print(i,u0, v0, w0)
        # Integrate the Lorenz equations on the time grid t
        t = np.linspace(0, tmax, n)
        f = odeint(lorenz, (u0, v0, w0), t, args=(sigma, beta, rho))
        np.save('data/{}.npy'.format(i),f)