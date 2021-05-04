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


f1 = lambda x, y: y
s1 = "y' - y = 0"
f2 = lambda x, y: -y
s2 = "y' + y = 0"
f3 = lambda x, y: -x * y**2
s3 = "y' + x * y^2 = 0"
f4 = lambda x, y: x ** 2 - y
s4 = "y' + y = x^2"
f5 = lambda x, y: -20*y + 20*x + 21
s5 = "y' + 20y -21 = -20x"
f6 = lambda x, y: 1 + x**2
s6 = "y' - 1 = x^2"
f7 = lambda x, y: x*y
s7 = "y' xy = 0"
f8 = lambda x, y: y - x**2 + 1
s8 = "y' - y = -x^2 +1"
f9 = lambda x, y: y + x
s9 = "y' - y = x"


EQUATIONS = [s1, s2, s3, s4, s5, s6, s7, s8, s9]
MODELS = [f1, f2, f3, f4, f5, f6, f7, f8, f9]
# METHODS = ["Euer","Heun","RK2","RK3","RK4","AB2","AB3","AB4","ABM PRED-COR","RKF45"]


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
    rk2 = np.array(numMethods.rkx(f, [y0], x, h, 2)).flatten()
    rk2_exec_time = perf_counter() - start_time_p3

    start_time_p4 = perf_counter()
    rk3 = np.array(numMethods.rkx(f, [y0], x, h, 3)).flatten()
    rk3_exec_time = perf_counter() - start_time_p4

    start_time_p5 = perf_counter()
    rk4 = np.array(numMethods.rkx(f, [y0], x, h, 4)).flatten()
    rk4_exec_time = perf_counter() - start_time_p5

    start_time_p6 = perf_counter()
    ab2 = numMethods.abx(f, y0, x, h, 2)
    ab2_exec_time = perf_counter() - start_time_p6

    start_time_p7 = perf_counter()
    ab3 = numMethods.abx(f, y0, x, h, 3)
    ab3_exec_time = perf_counter() - start_time_p7

    start_time_p8 = perf_counter()
    ab4 = numMethods.abx(f, y0, x, h, 4)
    ab4_exec_time = perf_counter() - start_time_p8

    start_time_p9 = perf_counter()
    rkf_x, rkf_y, rkf45_status = numMethods.rkf45(f, y0, a, b, 1*10**tol, h_max, h_min)
    rkf45_exec_time = perf_counter() - start_time_p9

    start_time_p10 = perf_counter()
    abm4_pc = numMethods.abm4_pc(f, y0, x, h)
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

    if 0 == rkf45_status:
        df_rkf45 = pd.DataFrame({
                'RKF45': rkf_y
            },
            index=rkf_x
        )
        df_rkf45.index.name = 'x'
    else:
        df_rkf45 = 0


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
    return df, df_rkf45, df_et, rkf45_status


def getChart(df, df_rkf45, rkf45_status):
    source_1 = df.reset_index().melt('x', var_name='method', value_name='y')
    if 0 == rkf45_status:
        source_rkf45 = df_rkf45.reset_index().melt('x', var_name='method', value_name='y')

        frames = [source_1, source_rkf45]
        source = pd.concat(frames)
    else:
        source = source_1


    selection = alt.selection_multi(fields=['method'])
    color = alt.condition(selection,
                          alt.Color('method:N',legend=None),
                          alt.value('lightgray'))

    line_chart = alt.Chart(source,title="Results").mark_line(point=True).encode(
        x=alt.X('x:Q', axis=alt.Axis(title='x [-]')),
        y=alt.Y('y:Q', axis=alt.Axis(title='y [-]')),
        color=color,
        tooltip='Name:N'
    ).properties(width=700, height=400).transform_filter(selection)#.interactive()

    make_selector = alt.Chart(source).mark_rect().encode(
        y=alt.Y('method:N', axis=alt.Axis(orient='right')),
        color=color
    ).add_selection(
        selection
    )

    return line_chart | make_selector


def getChart2(df_et):
    chart2 = alt.Chart(df_et, title="Performance comparison").mark_bar().encode(
        x=alt.X('ET', axis=alt.Axis(title='Elapsed Time [%]')),
        y=alt.Y('method',
                sort=alt.EncodingSortField(field='ET', order='ascending', op='sum'),
                axis=alt.Axis()
                ),
        color=alt.Color('method:N')
    ).properties(width=800, height=400)
    return chart2


header = st.beta_container()
sidebar = st.beta_container()
features = st.beta_container()
interactive = st.beta_container()


with header:
    st.title("Numerical Methods for solving ODE")
    st.text("Interactive application for solving Ordinary Differentiation Equations...")


with sidebar:
    st.sidebar.header("User Input Parameters")
    equation = st.sidebar.selectbox("Select equation:", EQUATIONS)


    st.sidebar.markdown("***")
    st.sidebar.markdown("Initial conditions")
    y0 = st.sidebar.number_input("y0 = ", 0.0)

    # check_boxes = [st.sidebar.checkbox(method, key=method) for method in METHODS]

    st.sidebar.markdown("***")
    st.sidebar.markdown("RKF45")
    h_min, h_max = st.sidebar.slider("step interval:", 0.001, 1.0, (0.001, 0.75), 0.001)
    tol = st.sidebar.slider("Tolerance: (1e(x)):", -12, -5, -7)


with features:
    st.write("Equation:")
    #st.latex(equation.replace('c', str(c)))
    st.latex(equation)

    # st.write("Analytical solution:")
    # x = np.linspace(-1000, 1000, n + 1)

    n = st.slider("Number of steps:", 5, 100, 10)
    a, b = st.slider("Interval:", -10.0, 10.0, (0.0, 5.0), 0.1)

    st.markdown("***")
    id_mod = EQUATIONS.index(equation)
    f = MODELS[id_mod]
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)

    st.write("Size of the step: ", h)
    st.write("y0 = ", y0)
    st.markdown("***")


with interactive:
    df, df_rkf45, df_et, rkf45_status = getDataFrame(f, a, b, n, y0)
    chart = getChart(df, df_rkf45, rkf45_status)
    chart2 = getChart2(df_et)

    if -1 == rkf45_status:
        st.markdown("Warning:")
        st.markdown("RKF45 can't be solved with choose parameters, try change minimum of the interval or tolerance.")

    # <span styl="color:red"> TEST </span>
    # st.markdown(f'<span styl=color:red> TEST </span>',unsafe_allow_html=True)
    st.altair_chart(chart)
    st.markdown("Note: By holding shift you can pick multiple choices in the legend")
    st.table(df)
    st.altair_chart(chart2)
