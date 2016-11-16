""" Project 4 for System and Signals
    Author: Zander W. Blasingame """

import numpy as np
import plotly.plotly as py
import plotly.graph_objs as go
from plotly import tools


# Function definitions
def _u(n):
    return 1 if n > 0 else 0


def _d(n):
    return 1 if n == 0 else 0


# Plot convolution
def plot_conv(n, input_data, y, title):
    inputs = [go.Scatter(
        x=n,
        y=_input['y'],
        name='${}$'.format(_input['name'])
    ) for _input in input_data]

    name = ''.join([input_data[i]['name'] + '*'
                    if i+1 < len(input_data) else input_data[i]['name']
                    for i in range(len(input_data))])

    output = go.Scatter(
        x=n,
        y=y,
        name='${}$'.format(name)
    )

    fig = tools.make_subplots(rows=1, cols=len(input_data)+1)

    i = 1
    for _input in inputs:
        fig.append_trace(_input, 1, i)
        i += 1

    fig.append_trace(output, 1, i)

    fig['layout'].update(title='Convolution of Signals')

    py.plot(fig, filename='ee-321-proj4-{}'.format(title))


def recursive_conv(input_data, n):
    assert len(input_data) > 1

    y = d(n)  # identity array

    for i in range(len(input_data)):
        y = np.convolve(y, input_data[i]['y'], mode='same')

    return y

u = np.vectorize(_u)
d = np.vectorize(_d)

# Create and plot sequences
# P1
n = np.linspace(-10, 10, 101)
input_data = [dict(y=u(n)-u(n-2), name='x(n)'),
              dict(y=d(n)-d(n-1), name='h(n)')]
y = recursive_conv(input_data, n)

plot_conv(n, input_data, y, title='conv-1')

# P2
n = np.linspace(-10, 10, 101)
input_data = [dict(y=2*d(n)+6*d(n-1)-d(n-2), name='x(n)'),
              dict(y=u(n-2)-u(n-10), name='h(n)')]
y = recursive_conv(input_data, n)

plot_conv(n, input_data, y, title='conv-2')

# P1
n = np.linspace(-10, 10, 101)
input_data = [dict(y=u(n)-u(n-2), name='x_1(n)'),
              dict(y=u(n-1)-u(n-4), name='x_2(n)'),
              dict(y=d(n)+d(n-2), name='x_3(n)')]
y = recursive_conv(input_data, n)

plot_conv(n, input_data, y, title='conv-3')
