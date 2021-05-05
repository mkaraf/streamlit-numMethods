import numMethods
import numpy as np
import streamlit as st
import pandas as pd
import altair as alt
from time import perf_counter
from scipy.integrate import odeint
from functools import wraps
from math import sin, cos


def flip(func):
    'Create a new function from the original with the arguments reversed'
    @wraps(func)
    def newfunc(*args):
        return func(*args[::-1])

    return newfunc


f1 = lambda x, y: y[0]
s1 = "y' - y = 0"
f2 = lambda x, y: -y[0]
s2 = "y' + y = 0"
f3 = lambda x, y: -x * y[0]**2
s3 = "y' + x * y^2 = 0"
f4 = lambda x, y: x ** 2 - y[0]
s4 = "y' + y = x^2"
# f5 = lambda x, y: -20*y + 20*x + 21
def f5(x, y):
    return -20*y[0] + 20*x + 21
s5 = "y' + 20y -21 = -20x"
# f6 = lambda x, y: 1 + x**2
def f6(x, y):
    return (y[0] + x**2)
s6 = "y' - 1 = x^2"
f7 = lambda x, y: x*y[0]
s7 = "y' xy = 0"
f8 = lambda x, y: y[0] - x**2 + 1
s8 = "y' - y = -x^2 +1"
f9 = lambda x, y: y[0] + x
s9 = "y' - y = x"


# y'' + y = 4x + 10*sin(x)
# y' = y'
# y'' = 4x + 10sin(x) - y
s10 = "y'' + y = 4x + 10*sin(x)"
def f10(x, y):
    return (y[1], 4 * x + 10 * sin(x) - y[0])

# y''' + 4y'' + 2y' - y = 0
# y' = y'
# y'' = y''
# y''' = -4y'' -2y' +y
s11 = "y''' = -4y'' -2y' +y"
def f11(x, y):
    return (y[2], y[1], - 4 * y[2] - 2 * y[1] - y[0])


EQUATIONS = [s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11]
MODELS = [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11]
# METHODS = ["Euler","Heun","RK2","RK3","RK4","AB2","AB3","AB4","ABM PRED-COR","RKF45"]


def getDataFrame(f, a, b, n, y0):
    analytic = odeint(flip(f), y0, x)

    start_time_p0 = perf_counter()
    euler = np.array(numMethods.euler(f, y0, x, h))[:, 0]

    start_time_p1 = perf_counter()
    heun = np.array(numMethods.heun(f, y0, x, h))[:, 0]

    start_time_p2 = perf_counter()
    rk2 = np.array(numMethods.rkx(f, y0, x, h, 2))[:, 0]

    start_time_p3 = perf_counter()
    rk3 = np.array(numMethods.rkx(f, y0, x, h, 3))[:, 0]

    start_time_p4 = perf_counter()
    rk4 = np.array(numMethods.rkx(f, y0, x, h, 4))[:, 0]

    start_time_p5 = perf_counter()
    ab2 = np.array(numMethods.abx(f, y0, x, h, 2))[:, 0]

    start_time_p6 = perf_counter()
    ab3 = np.array(numMethods.abx(f, y0, x, h, 3))[:, 0]

    start_time_p7 = perf_counter()
    ab4 = np.array(numMethods.abx(f, y0, x, h, 4))[:, 0]

    start_time_p8 = perf_counter()
    abm4_pc = np.array(numMethods.abm4_pc(f, y0, x, h))[:, 0]

    start_time_p9 = perf_counter()
    rkf_x, rkf_y, rkf45_status = np.array(numMethods.rkf45(f, y0, a, b, 1*10**tol, h_max, h_min))
    rkf_y = np.array(rkf_y)[:, 0]
    start_time_p10 = perf_counter()

    total_time = ((start_time_p10 - start_time_p0) / 100)


    # df_et = pd.DataFrame()
    # df_et['ET'] = [eul_exec_time / max_time, heun_exec_time / max_time, rk2_exec_time / max_time,
    #                          rk3_exec_time / max_time, rk4_exec_time / max_time, ab2_exec_time / max_time,
    #                          ab3_exec_time / max_time, ab4_exec_time / max_time, rkf45_exec_time / max_time,
    #                          abm4_exec_time / max_time]
    # df_et.index = ['Euler', 'Heun', 'RK2', 'RK3', 'RK4', 'AB2', 'AB3', 'AB4', 'ABM4_PC', 'RKF45']
    # df_et['method'] = df_et.index

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

    return df, df_rkf45, rkf45_status #, df_et


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

    st.altair_chart(line_chart | make_selector)
    return


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
    id_mod = EQUATIONS.index(equation)


    st.sidebar.markdown("***")
    st.sidebar.markdown("Initial conditions")

    if id_mod < 8:
        y0 = [st.sidebar.number_input("y0 = ", 0.0)]
    elif id_mod == 9:
        y0 = [st.sidebar.number_input("y0 = ", 0.0),
                st.sidebar.number_input("z0 = ", 0.0),
                ]
    elif id_mod == 10:
        y0 = [st.sidebar.number_input("y0 = ", 0.0),
                st.sidebar.number_input("z0 = ", 0.0),
                st.sidebar.number_input("u0 = ", 0.0),
                ]

    st.sidebar.markdown("***")
    st.sidebar.markdown("RKF45")
    h_min, h_max = st.sidebar.slider("Size of the step:", 0.001, 1.000, (0.100, 0.750), 0.001)
    tol = st.sidebar.slider("Tolerance: (1e(x)):", -12, -2, -5)


with features:
    st.write("Equation:")
    #st.latex(equation.replace('c', str(c)))
    st.latex(equation)

    # st.write("Analytical solution:")
    # x = np.linspace(-1000, 1000, n + 1)

    n = st.slider("Number of steps:", 5, 100, 10)
    a, b = st.slider("Interval:", -10.0, 10.0, (0.0, 5.0), 0.1)

    st.markdown("***")
    f = MODELS[id_mod]
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)

    st.write("Size of the step: ", h)
    st.write("Initial conditions = ", y0)
    st.markdown("***")


with interactive:
    df, df_rkf45, rkf45_status = getDataFrame(f, a, b, n, y0)

    if -1 == rkf45_status:
        st.markdown("Warning:")
        st.markdown("RKF45 could not converge to the required tolerance with chose minimum step size")

    st.text("Note: You can select methods in legend (for multiple select hold SHIFT and click)")
    getChart(df, df_rkf45, rkf45_status)

    # chart = getChart(df, df_rkf45, rkf45_status)
    # chart2 = getChart2(df_et)

    # <span styl="color:red"> TEST </span>
    # st.markdown(f'<span styl=color:red> TEST </span>',unsafe_allow_html=True)
    # st.altair_chart(chart)
    st.markdown("***")
    st.table(df)
    # st.altair_chart(chart2)
