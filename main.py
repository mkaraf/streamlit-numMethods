import numMethods
import numpy as np
import streamlit as st
import pandas as pd
import altair as alt
from time import perf_counter
from scipy.integrate import odeint
from functools import wraps


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
    analytic = odeint(flip(f), y0, x)

    start_time_p1 = perf_counter()
    euler = numMethods.euler(f, y0, x, h)
    eul_exec_time = perf_counter() - start_time_p1

    start_time_p2 = perf_counter()
    heun = numMethods.heun(f, y0, x, h)
    heun_exec_time = perf_counter() - start_time_p2

    start_time_p3 = perf_counter()
    rk2 = numMethods.rk2(f, y0, x, h)
    rk2_exec_time = perf_counter() - start_time_p3

    start_time_p4 = perf_counter()
    rk3 = numMethods.rk3(f, y0, x, h)
    rk3_exec_time = perf_counter() - start_time_p4

    start_time_p5 = perf_counter()
    rk4 = numMethods.rk4(f, y0, x, h)
    rk4_exec_time = perf_counter() - start_time_p5

    start_time_p6 = perf_counter()
    ab2 = numMethods.ab2(f, y0, x, h)
    ab2_exec_time = perf_counter() - start_time_p6

    start_time_p7 = perf_counter()
    ab3 = numMethods.ab3(f, y0, x, h)
    ab3_exec_time = perf_counter() - start_time_p7

    start_time_p8 = perf_counter()
    ab4 = numMethods.ab4(f, y0, x, h)
    ab4_exec_time = perf_counter() - start_time_p8

    start_time_p9 = perf_counter()
    rkf_x, rkf_y, e = numMethods.rkf45(f, y0, a, b, 1e-5, 0.25, 0.01)
    rkf45_exec_time = perf_counter() - start_time_p9

    start_time_p10 = perf_counter()
    abm4_pc = numMethods.ab4(f, y0, x, h)
    stop_time = perf_counter()
    abm4_exec_time = stop_time - start_time_p10

    max_time = ((stop_time - start_time_p1) / 100)

    df_et = pd.DataFrame()
    df_et['ET'] = [eul_exec_time / max_time, heun_exec_time / max_time, rk2_exec_time / max_time,
                             rk3_exec_time / max_time, rk4_exec_time / max_time, ab2_exec_time / max_time,
                             ab3_exec_time / max_time, ab4_exec_time / max_time, rkf45_exec_time / max_time,
                             abm4_exec_time / max_time]
    df_et.index = ['Euler', 'Heun', 'RK2', 'RK3', 'RK4', 'AB2', 'AB3', 'AB4', 'RKF45','ABM4_PC']
    df_et['method'] = df_et.index

    # DATAFRAME
    df = pd.DataFrame({
        'Analytical': analytic[:, 0],
        'Euler': euler,
        'Heun': heun,
        'RK2': rk2,
        'RK3': rk3,
        'RK4': rk4,
        'AB2': ab2,
        'AB3': ab3,
        'AB4': ab4,
        'ABM4_PC': abm4_pc
    },
        index=x
    )
    df.index.name = 'x'
    return df, df_et


def getChart(df):
    source = df.reset_index().melt('x', var_name='method', value_name='y')

    # The basic line
    line = alt.Chart().mark_line(point=True).encode(
        x='x:Q',
        y='y:Q',
        color='method:N'
    )

    # Put the five layers into a chart and bind the data
    chart = alt.layer(line, data=source).interactive().properties(width=800, height=400)

    return chart


def getChart2(df_et):
    chart2 = alt.Chart(df_et).mark_bar().encode(
        x=alt.X('ET', axis=alt.Axis(title='Elapsed Time [%]')),
        y=alt.Y('method',
                sort=alt.EncodingSortField(field='ET', order='ascending', op='sum'),
                axis=alt.Axis(title='Method')
                ),
        color=alt.Color('method:N')
    ).properties(width=800, height=400)
    return chart2


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
    st.sidebar.markdown("***")

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
    st.markdown("***")
    #st.write("Interval: ", (a, b))
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    st.write("Size of the step: ", h)
    st.write("y0 = ", y0)

with interactive:
    df, df_et = getDataFrame(f, a, b, n, y0)
    chart = getChart(df)
    chart2 = getChart2(df_et)
    st.altair_chart(chart)
    st.table(df)
    st.altair_chart(chart2)
