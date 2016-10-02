# imports
import numpy as np
import matplotlib.pyplot as plt


""" Implementation of the Heaviside step Function
    Defined as the integral of the dirac delta function."""
def _unit_step(n):
    return 0 if n < 0 else 1

# vectorize function for increased performance
unit_step = np.vectorize(_unit_step)

# define input vector
n = np.arange(-10, 11, 1)

u = unit_step(n)

# graph the unit step function
plt.figure()
plt.plot(n, u)
plt.xlim(-12, 12)
plt.ylim(-1, 2)

plt.figure()
plt.stem(n, u)
plt.xlim(-12, 12)
plt.ylim(-1, 2)

plt.figure()
plt.step(n, u)
plt.xlim(-12, 12)
plt.ylim(-1, 2)

plt.show()
