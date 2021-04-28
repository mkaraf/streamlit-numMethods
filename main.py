import numMethods
import numpy as np
import streamlit as st
import pandas as pd
import altair as alt
from handcalcs.decorator import handcalc
from scipy.integrate import odeint
from functools import wraps
from math import sqrt


def flip(func):
    'Create a new function from the original with the arguments reversed'

    @wraps(func)
    def newfunc(*args):
        return func(*args[::-1])

    return newfunc

# ORDERS = ["first", "second", "third"]
# EQUATIONS_1 = ["y", "-x * y^2", "x^2 - y","Cos(x) * y", "y^2(x - x^3)", "-20y + 20x + 21"]
EQUATIONS = ["y' = cy", "y' = c2y + c1x + c0"]
# EQUATIONS_2 = ["0", "1", "2", "3"]
# EQUATIONS_3 = ["00", "11", "22", "33"]

c = 1.0

f1 = lambda x, y: y * c
f2 = lambda x, y: x * y
f3 = lambda x, y: -20 * y + 20 * x + 21


def model_f1(x,y):
    dydx = c * y
    return dydx

def getDataFrame(f, a, b, n, y0):
    # Calculate ODE via methods
    analytical = odeint(flip(f1),y0,x)
    euler = numMethods.Euler(f, y0, x, h)
    heun = numMethods.Heun(f, y0, x, h)
    rk4 = numMethods.Rk4(f, y0, x, h)
    # DATAFRAME
    df = pd.DataFrame({
        #'x':x,
        'Analytical': analytical[:,0],
        'Euler': euler,
        'Heun': heun,
        'RK4': rk4
    }
        ,index=x
    )
    df.index.name = 'x'
    return df

def getChart(df):
    source = df.reset_index().melt('x', var_name='method', value_name='y')

    #selector for number steps
    #slider = alt.binding_range(min=0, max=100, step=1, name='Steps: ')
    #selector = alt.selection_single(name="SelectorName", fields=['Steps: '],
    #                                bind=slider, init={'Steps: ' : 10})


    # The basic line
    line = alt.Chart().mark_line(point=True).encode(
        x='x:Q',
        y='y:Q',
        color='method:N'
    )

    # Put the five layers into a chart and bind the data
    chart = alt.layer(line,
                data=source).interactive().properties(width=800, height=400)
    return chart


header = st.beta_container()
sidebar = st.beta_container()
features = st.beta_container()
interactive = st.beta_container()

with header:
    st.title('Numerical Methods for solving ODE')
    st.text('Interactive application for solving Ordinary Differentiation Equations...')

with sidebar:
    st.sidebar.header("User Input Parameters")
    equation = st.sidebar.selectbox("Select equation:", (EQUATIONS))

    #n = st.sidebar.slider("Number of steps:", 5, 100, 10)
    #a, b = st.sidebar.slider("Interval:", -10.0, 10.0, (0.0, 5.0), 0.1)
    c = st.sidebar.number_input("c0 = ", 1.0)
    y0 = st.sidebar.number_input("y0 = ", 0.0)

    f = model_f1
    # h = (b - a) / n
    # x = np.linspace(a, b, n + 1)

with features:
    st.write("Equation:")
    st.latex(equation.replace('c',str(c)))
    #st.write("Used parameters:")
    n = st.slider("Number of steps:", 5, 100, 10)
    a, b = st.slider("Interval:", -10.0, 10.0, (0.0, 5.0), 0.1)
    #st.write("Interval: ", (a, b))
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    st.write("Size of the step: ", h)
    st.write("y0 = ", y0)

with interactive:
    df = getDataFrame(f, a, b, n, y0)
    chart = getChart(df)
    st.altair_chart(chart)
    st.table(df)