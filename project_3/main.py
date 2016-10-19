""" File to graph several signals using the Fourier Transform
    and Fourier Coefficients
    Author: Zander Blasingame
    Course: EE 321 """

import numpy as np
from scipy import signal
import plotly.plotly as py
import plotly.graph_objs as go

""" Function Definitions """

# Constants
SIGNAL_RANGE = 40
RESOLUTION = 100E-3  # 100ms
N = 2000

# define discrete unit impulse
d = lambda x: 1 if x == 0 else 0


# Construct signal from fourier coefficients
def construct_signal(X_k, period=1, N=N):
    return lambda x: np.sum(X_k(k) * np.exp(2j*np.pi*k*x/period)
                            for k in range(-N, N))


def mean_squared_error(signal_1, signal_2, N=N):
    return 1/N * np.sum(np.abs(signal_1 - signal_2)**2)

# """ Problem 1a """
# X_k = lambda k: d(k) + 0.25*d(k-1) + 0.25*(k+1) - 0.5j*d(k-2) + 0.5j*d(k+2)

# # Construct and plot signal
# x = construct_signal(X_k)
# t = np.arange(-SIGNAL_RANGE/2, SIGNAL_RANGE/2, RESOLUTION)

# data = [go.Scatter(x=t, y=[x(time).real for time in t])]
# layout = go.Layout(xaxis=dict(title='Time (s)', showticklabels=True,
#                               tickmode='linear',
#                               tickangle=0,
#                               dtick=5),
#                    yaxis=dict(title='Amplitude x(t)'),
#                    title='Signal 1a')

# fig = go.Figure(data=data, layout=layout)

# py.image.save_as(fig, filename='signal_1.png')

# """ Problem 1b """
# X_k = lambda k: 1j*k if abs(k) < 3 else 0

# # Construct and plot signal
# x = construct_signal(X_k)
# t = np.arange(-SIGNAL_RANGE/2, SIGNAL_RANGE/2, RESOLUTION)

# data = [go.Scatter(x=t, y=[x(time).real for time in t])]
# layout = go.Layout(xaxis=dict(title='Time (s)', showticklabels=True,
#                               tickmode='linear',
#                               tickangle=0,
#                               dtick=5),
#                    yaxis=dict(title='Amplitude x(t)'),
#                    title='Signal 1b')

# fig = go.Figure(data=data, layout=layout)

# py.image.save_as(fig, filename='signal_2.png')"""


# """ Problem 2a """
# # added term to avoid dividing by zero
# X_k = lambda k: 1j/(2*np.pi*k + 1E-9) * (np.exp(-1j*4*np.pi*k/3) - 1)

# # Construct and plot signal
# x = construct_signal(X_k, 3)
# t = np.arange(-SIGNAL_RANGE/2, SIGNAL_RANGE/2, RESOLUTION)

# data = [go.Scatter(x=t, y=[x(time).real for time in t])]
# layout = go.Layout(xaxis=dict(title='Time (s)', showticklabels=True,
#                               tickmode='linear',
#                               tickangle=0,
#                               dtick=5),
#                    yaxis=dict(title='Amplitude x(t)'),
#                    title='Signal 2a')

# fig = go.Figure(data=data, layout=layout)

# py.image.save_as(fig, filename='signal_3.png')


""" Problem 2b """    
X_k = lambda k: (-1 + (-1)**k)/(np.pi**2 * (k+1E-9)**2)

# Construct and plot signal
x = construct_signal(lambda k: X_k(k) + 0.5*d(k), 2)
t = np.arange(0.001, 2, 0.01)

data = [go.Scatter(x=t, y=[x(time).real for time in t])]
layout = go.Layout(xaxis=dict(title='Time (s)', showticklabels=True,
                              tickmode='linear',
                              tickangle=0,
                              dtick=5),
                   yaxis=dict(title='Amplitude x(t)'),
                   title='Signal 2b')

fig = go.Figure(data=data, layout=layout)


py.iplot(fig, filename='signal_4.png')

# ideal signal
x_ideal = lambda t: 2 * np.abs(0.5*t - np.floor(0.5*t + 0.5))

t = np.arange(0.001, 2, 0.01)

num_components = np.arange(1,10000,500)
errors = [mean_squared_error(construct_signal(lambda k: X_k(k) + 0.5*d(k), 2, N=n)(t), x_ideal(t), N=n)
          for n in num_components]

data = [go.Scatter(x=num_components, y=errors)]
layout = go.Layout(xaxis=dict(title='N', showticklabels=True,
                              tickmode='linear',
                              tickangle=0,
                              dtick=1000),
                   yaxis=dict(title='Error',
                              type='log'),
                   title='Reconstruction Error 2b')



fig = go.Figure(data=data, layout=layout)
py.image.save_as(fig, filename='rec_err_2.png')
