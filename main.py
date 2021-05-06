import numMethods
import models
import numpy as np
import streamlit as st
import pandas as pd
import altair as alt
from time import perf_counter
from scipy.integrate import odeint
from functools import wraps


# EQUATIONS = [s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11]
# MODELS = [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11]
METHODS = ["NONE", "Euler", "Heun","Runge-Kutta 2", "Runge-Kutta 3","Runge-Kutta 4","Adams-Basforth 2",
           "Adams-Basforth 3", "Adams-Basforth 4", "Adams-Basforth-Moulton", "Runge-Kutta Fehlberg 45"]
EQUATIONS = models.get_model_str()
MODELS = [models.f0, models.f1, models.f2, models.f3, models.f4, models.f5]


def flip(func):
    'Create a new function from the original with the arguments reversed'
    @wraps(func)
    def newfunc(*args):
        return func(*args[::-1])
    return newfunc


def get_latex_method(method):
    if method == "Euler":
        show_euler()
    elif method == "Heun":
        show_heun()
    elif method == "Runge-Kutta 2":
        show_rk2()
    elif method == "Runge-Kutta 3":
        show_rk3()
    elif method == "Runge-Kutta 4":
        show_rk4()
    elif method == "Adams-Basforth 2":
        show_ab2()
    elif method == "Adams-Basforth 3":
        show_ab3()
    elif method == "Adams-Basforth 4":
        show_ab4()
    elif method == "Adams-Basforth-Moulton":
        show_abm4()
    elif method == "Runge-Kutta Fehlberg 45":
        show_rkf45()
    return


def show_euler():
    st.latex("y_{(i+1)} = y_{(i)}  + h f(x_{(i)}, y_{(i)})")
    return


def show_heun():
    st.latex("y_{pred} = y_{(i)}  + h f(x_{(i)}, y_{(i)})")
    st.latex("y_{(i+1)} = y_{(i)} + \\frac {1}{2} h (f(x_{(i)}, y_{(i)}) + f(x_{(i+1)}, y_{pred}))")
    return


def show_rk2():
    st.latex('C = \\begin{pmatrix} 0 \\\\\\ 1  \end{pmatrix}'
             '\\quad'
             '\ A = \\begin{pmatrix} 0 & 0 \\\\ \\ 1/2 & 0 \end{pmatrix}'
             '\\quad'
             '\\ B = \\begin{pmatrix} 0 & 1/2  \end{pmatrix}')

    st.latex("k_1 = f(x_{(i)}, y_{(i)})")
    st.latex("k_2 = f(x_{(i)} + h C_2, y_{(i)} + h(A_{21}k_1))")
    st.latex("y_{(i+1)} = y_{(i)}  + h \displaystyle\sum_{j=1}^2 B_jk_j")
    return


def show_rk3():
    st.latex('C = \\begin{pmatrix} 0 \\\\\\ 1/2 \\\\\\ 1 \end{pmatrix}'
             '\\quad'
             '\ A = \\begin{pmatrix} 0 & 0 & 0 \\\\\\ 1/2 & 0 & 0 \\\\\\ -1 & 2 & 0 \end{pmatrix}'
             '\\quad'
             '\\ B = \\begin{pmatrix} 1/6 & 2/3 & 1/6  \end{pmatrix}')

    st.latex("k_1 = f(x_{(i)}, y_{(i)})")
    st.latex("k_2 = f(x_{(i)} + h C_2, y_{(i)} + h(A_{21}k_1))")
    st.latex("k_3 = f(x_{(i)} + h C_3, y_{(i)} + h(A_{31}k_1 + A_{32}k_2))")
    st.latex("y_{(i+1)} = y_{(i)}  + h \displaystyle\sum_{j=1}^3 B_jk_j")
    return


def show_rk4():
    st.latex('C = \\begin{pmatrix} 1/6 \\\\\\ 1/3 \\\\\\ 1/3 \\\\\\ 1/6 \end{pmatrix}'
             '\\quad'
             '\ A = \\begin{pmatrix} 0 & 0 & 0 & 0 \\\\\\ 1/2 & 0 & 0 & 0 \\\\\\ 0 & 1/2 & 0 & 0 '
             '\\\\\\ 0 & 0 & 1 & 0 \end{pmatrix}'
             '\\quad'
             '\\ B = \\begin{pmatrix} 0 & 1/2 & 1/2 & 1 \end{pmatrix}')

    st.latex("k_1 = f(x_{(i)}, y_{(i)})")
    st.latex("k_2 = f(x_{(i)} + h C_2, y_{(i)} + h(A_{21}k_1))")
    st.latex("k_3 = f(x_{(i)} + h C_3, y_{(i)} + h(A_{31}k_1) + h(A_{32}k_2))")
    st.latex("k_4 = f(x_{(i)} + h C_4, y_{(i)} + h(A_{41}k_1 + A_{42}k_2 + A_{43}k_3))")
    st.latex("y_{(i+1)} = y_{(i)}  + h \displaystyle\sum_{j=1}^4 B_jk_j")
    return


def show_ab2():
    st.latex("y_{(i+1)} = y_{(i)} + \\frac{1}{2} h (3f(x_{(i)},y_{(i)})-f(x_{(i-1)},y_{(i-1)}))")
    return


def show_ab3():
    st.latex("y_{(i+1)} = y_{(i)} + \\frac{1}{12} h (23f(x_{(i)},y_{(i)})-16f(x_{(i-1)},y_{(i-1)})"
             "+5f(x_{(i-2)},y_{(i-2)}))")
    return


def show_ab4():
    st.latex("y_{(i+1)} = y_{(i)} + \\frac{1}{24} h (55f(x_{(i)},y_{(i)}) - 59f(x_{(i-1)},y_{(i-1)})"
             "+37f(x_{(i-2)},y_{(i-2)}) - 9f(x_{(i-3)},y_{(i-3)}))")
    return

def show_abm4():
    st.latex("y_{pred} = y_{(i)} + \\frac{1}{24} h (55f(x_{(i)},y_{(i)}) - 59f(x_{(i-1)},y_{(i-1)})"
             "+37f(x_{(i-2)},y_{(i-2)}) - 9f(x_{(i-3)},y_{(i-3)}))")

    st.latex("y_{(i+1)} = y_{(i)} + \\frac{1}{24} h (9f(x_{(i+1)},y_{pred}) + 19f(x_{(i)},y_{(i)})"
             "-5f(x_{(i-1)},y_{(i-1)}) + f(x_{(i-2)},y_{(i-2)}))")
    return

def show_rkf45():
    st.latex('C = \\begin{pmatrix} 0 \\\\ 1/4 \\\\ 3/8 \\\\ 12/13 \\\\ 1 \\\\ 1/2 \end{pmatrix}'
             '\\quad'
             '\\ A = \\begin{pmatrix} '
             '\\ 0 & 0 & 0 & 0 & 0 & 0 \\\\'
             '\\ 1/4 & 0 & 0 & 0 & 0 & 0 \\\\'
             '\\ 3/32 & 9/32 & 0 & 0 & 0 & 0 \\\\'
             '\\ 1932/2197 & -7200/2197 & 7296/2197 & 0 & 0 & 0 \\\\'
             '\\ 439/216 & -8 & 3680/513 & -845/4104 & 0 & 0 \\\\'
             '\\ -8/27 & 2 & -3544/2565 & 1859/4104 & -11/40 & 0'
             '\end{pmatrix}')
    st.latex('B^* = \\begin{pmatrix} 25/216 & 0 & 1408/2565 & 2197/4104 & -1/5 & 0 \end{pmatrix}')
    st.latex('B = \\begin{pmatrix} 16/135 & 0 & 6656/12825 & 28561/56430 & -9/50 & 2/55\end{pmatrix}')

    st.latex("k_1 = f(x_{(i)}, y_{(i)})")
    st.latex("k_2 = f(x_{(i)} + h C_2, y_{(i)} + h(A_{21}k_1))")
    st.latex("k_3 = f(x_{(i)} + h C_3, y_{(i)} + h(A_{31}k_1 + A_{32}k_2))")
    st.latex("k_4 = f(x_{(i)} + h C_4, y_{(i)} + h(A_{41}k_1 + A_{42}k_2 + A_{43}k_3))")
    st.latex("k_5 = f(x_{(i)} + h C_3, y_{(i)} + h(A_{51}k_1 + A_{52}k_2 + A_{53}k_3 + A_{54}k_4))")
    st.latex("k_6 = f(x_{(i)} + h C_4, y_{(i)} + h(A_{61}k_1 + A_{62}k_2 + A_{63}k_3 + A_{64}k_4 + A_{65}k_5))")

    st.latex("R = \\frac{1}{h} \\lvert 1/360k_1 - 128/4275k_3 - 2167/752460k_4 + 1/50k_5 + 2/55k_6 \\rvert")
    st.latex("if\\;R\\le tolerance,\\;then\\;approximation\\;is\\;accepted:")
    # st.write("if R <= tol then approximation is accepted, if R > tol then skip to the adjustation of step")
    st.latex("y_{(i+1)}^{(4)} = y_{(i)}^{(4)}  + h \displaystyle\sum_{j=1}^5 B_j^*k_j")
    st.latex("x_{i+1} = x_{i} + h")
    # st.latex("y_{(i+1)}^{(5)} = y_{(i)}^{(4)}  + h \displaystyle\sum_{j=1}^6 B_jk_j")

    # st.latex("R = \\frac{1}{h} \\lvert y_{(i)}^{(4)} - y_{(i)}^{(5)} \\rvert")
    st.latex("Adjustation\\;of\\;the\\;step")
    # st.write("Continue with adjustation of step")
    st.latex("h_{adj} = 0,84 \\lparen \\frac{\\varepsilon*h}{R}\\rparen ^{0,25}")
    st.latex("if\\;h_{adj}>h_{max}\\;then:")
    st.latex("\\;h=h_{max}")
    st.latex("else\\; if\\; h_{adj}>h_{min}\\; then:")
    st.latex("\\;h=h_{adj}")
    st.latex("else\\; stop\\;the\\;algorithm\\;because")
    st.latex("converge\\;to\\;the\\;required\\;tolerance\\;with\\;chosen\\;minimum\\;step\\;size\\;is\\;not\\;possible")

    return


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
    ).properties(width=700, height=400).transform_filter(selection)

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
methods = st.beta_container()
sidebar = st.beta_container()
features = st.beta_container()
interactive = st.beta_container()


with header:
    st.title("Numerical Methods for solving ODE")
    st.text("Interactive application for solving Ordinary Differentiation Equations...")

with methods:
    show = st.selectbox("Show me the algorithm: ", METHODS)
    get_latex_method(show)
    st.markdown("***")


with sidebar:
    st.sidebar.header("User Input Parameters")
    equation = st.sidebar.selectbox("Select equation:", EQUATIONS)
    id_mod = EQUATIONS.index(equation)

    st.sidebar.markdown("***")
    st.sidebar.markdown("Initial conditions")

    if id_mod < 4:
        y0 = [st.sidebar.number_input("y(0) = ", 0.0)]
        init_cond = ("y_{(0)}="+str(y0[0]))
    elif id_mod == 4:
        y0 = [st.sidebar.number_input("y'(0) = ", 0.0),
                st.sidebar.number_input("y(0) = ", 0.0),
                ]
        init_cond = ("y'_{(0)}=" + str(y0[0]),",\\;y_{(0)}=" + str(y0[1]))
    elif id_mod == 5:
        y0 = [st.sidebar.number_input("y''(0) = ", 0.0),
                st.sidebar.number_input("y'(0) = ", 0.0),
                st.sidebar.number_input("y(0) = ", 0.0),
                ]
        init_cond = ("y''_{(0)}=" + str(y0[0]) + ",\\;y'_{(0)}=" + str(y0[1]) + ",\\;y_{(0)}=" + str(y0[2]))

    st.sidebar.markdown("***")
    st.sidebar.markdown("RKF45")
    h_min, h_max = st.sidebar.slider("Size of the step:", 0.001, 1.000, (0.100, 0.750), 0.001)
    tol = st.sidebar.slider("Tolerance: (1e(x)):", -12, -2, -5)


with features:
    st.write("Selected ordinary differential equation:")

    st.latex(equation)
    st.latex(init_cond)

    # st.write("Analytical solution:")
    # x = np.linspace(-1000, 1000, n + 1)
    # x_analytic = np.linspace(0, 1000, 1)
    # analytic = odeint(flip(f), y0, x)

    n = st.slider("Number of steps:", 5, 100, 10)
    a, b = st.slider("Interval x:", -10.0, 10.0, (0.0, 5.0), 0.1)

    st.markdown("***")
    f = MODELS[id_mod]
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)

    st.write("Size of the step: ", h)
    st.markdown("***")


with interactive:
    df, df_rkf45, rkf45_status = getDataFrame(f, a, b, n, y0)

    if -1 == rkf45_status:
        st.markdown("Warning:")
        st.markdown("RKF45 could not converge to the required tolerance with chose minimum step size, please adjust"
                    " the parameters.")

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
