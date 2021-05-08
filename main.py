import numMethods
import models
import latex_methods as lm
import numpy as np
import streamlit as st
import pandas as pd
import altair as alt
from scipy.integrate import odeint
from functools import wraps


METHODS = ["NONE", "Euler", "Heun", "Runge-Kutta 2", "Runge-Kutta 3", "Runge-Kutta 4", "Adams-Bashforth 2",
           "Adams-Bashforth 3", "Adams-Bashforth 4", "Adams-Bashforth-Moulton", "Runge-Kutta Fehlberg 45"]
MODELS = [models.f0, models.f1, models.f2, models.f3, models.f4, models.f5]
EQUATIONS = models.get_model_str()

st.set_page_config(layout="wide")

analytic = pd.DataFrame(columns=['Analytical'])
numeric = pd.DataFrame(columns=['Euler', 'Heun', 'RK2', 'RK3', 'RK4', 'AB2', 'AB3', 'AB4', 'ABM4_PC'])
rkf45 = pd.DataFrame(columns=['RKF45'])
analytic.index.name = 'x'
numeric.index.name = 'x'
rkf45.index.name = 'x'
first_run = True


def flip(func):
    # Create a new function from the original with the arguments reversed
    @wraps(func)
    def newfunc(*args):
        return func(*args[::-1])
    return newfunc


def get_latex_method(method):
    formulas = []
    if method == "Euler":
        formulas = lm.get_euler_latex()
    elif method == "Heun":
        formulas = lm.get_heun_latex()
    elif method == "Runge-Kutta 2":
        formulas = lm.get_rk2_latex()
    elif method == "Runge-Kutta 3":
        formulas = lm.get_rk3_latex()
    elif method == "Runge-Kutta 4":
        formulas = lm.get_rk4_latex()
    elif method == "Adams-Bashforth 2":
        formulas = lm.get_ab2_latex()
    elif method == "Adams-Bashforth 3":
        formulas = lm.get_ab3_latex()
    elif method == "Adams-Bashforth 4":
        formulas = lm.get_ab4_latex()
    elif method == "Adams-Bashforth-Moulton":
        formulas = lm.get_abm4_latex()
    elif method == "Runge-Kutta Fehlberg 45":
        formulas = lm.get_rkf45_latex()

    if method != "NONE":
        for formula in formulas:
            st.latex(formula)
    return


def get_chart(df, df_rkf45, rkf45_stat, data, requested_update):
    source_1 = df.reset_index().melt('x', var_name='method', value_name='y')
    if 0 == rkf45_stat:
        source_rkf45 = df_rkf45.reset_index().melt('x', var_name='method', value_name='y')

        frames = [source_1, source_rkf45]
        source = pd.concat(frames)
    else:
        source = source_1

    selection = alt.selection_multi(fields=['method'])
    color = alt.condition(selection,
                          alt.Color('method:N', legend=None),
                          alt.value('lightgray'))

    line_chart = alt.Chart(source, title="Results").mark_line(point=True).encode(
        x=alt.X('x:Q', axis=alt.Axis(title='x [-]')),
        y=alt.Y('y:Q', axis=alt.Axis(title='y [-]')),
        color=color,
        tooltip=[alt.Tooltip('method:N'),
                 alt.Tooltip('y:N'),
                 alt.Tooltip('x:N')]
    ).properties(width=800, height=400).transform_filter(selection)

    data.index.name = 'x'
    source_data = data.reset_index().melt('x', var_name='method', value_name='y')
    chart = alt.Chart(source_data.reset_index()).mark_line(color='#FF0000', point=False).encode(
        x='x:Q',
        y='y:Q',
    ).properties(width=800, height=400)

    make_selector = alt.Chart(source).mark_rect().encode(
        y=alt.Y('method:N', axis=alt.Axis(orient='right')),
        color=color
    ).add_selection(
        selection
    )
    if requested_update:
        test = alt.layer(chart, line_chart)
    else:
        test = chart
    st.altair_chart(test | make_selector)
    return


def calc_numerical(model, a_min, b_max, init_con, num_steps, step_size, step_size_min, step_size_max, tolerance):
    x = np.linspace(a_min, b_max, num_steps + 1)

    euler = np.array(numMethods.euler(model, init_con, x, step_size))[:, 0]
    heun = np.array(numMethods.heun(model, init_con, x, step_size))[:, 0]
    rk2 = np.array(numMethods.rkx(model, init_con, x, step_size, 2))[:, 0]
    rk3 = np.array(numMethods.rkx(model, init_con, x, step_size, 3))[:, 0]
    rk4 = np.array(numMethods.rkx(model, init_con, x, step_size, 4))[:, 0]
    ab2 = np.array(numMethods.abx(model, init_con, x, step_size, 2))[:, 0]
    ab3 = np.array(numMethods.abx(model, init_con, x, step_size, 3))[:, 0]
    ab4 = np.array(numMethods.abx(model, init_con, x, step_size, 4))[:, 0]
    abm4_pc = np.array(numMethods.abm4_pc(model, init_con, x, step_size))[:, 0]
    rkf_x, rkf_y, rkf45_stat = np.array(numMethods.rkf45(model, init_con, a, b, 1 * 10 ** tolerance,
                                                         step_size_max, step_size_min))
    rkf_y = np.array(rkf_y)[:, 0]

    # DATAFRAME
    df_num_sol = pd.DataFrame({
        # 'Analytical': analytic[:, 0],
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
    df_num_sol.index.name = 'x'

    if 0 == rkf45_status:
        df_rkf45 = pd.DataFrame({
            'RKF45': rkf_y
        },
            index=rkf_x
        )
        df_rkf45.index.name = 'x'
    else:
        df_rkf45 = 0

    return df_num_sol, df_rkf45, rkf45_status


def calc_analytical(a_min, b_max, model, init_con):
    number_of_steps = (b_max - a_min) / 0.01
    x = np.linspace(a_min, b_max, round(number_of_steps + 1.0))
    solution = odeint(flip(model), init_con, x)
    pd_analytic = pd.DataFrame({'value': solution[:, 0]}, index=x)
    return pd_analytic


header = st.beta_container()
sidebar = st.beta_container()
equations = st.beta_container()
left_ode, right_analytic = st.beta_columns(2)

control_panel = st.beta_container()
start_point_left, start_point_right = st.beta_columns(2)
request_analytical, request_numerical = st.beta_columns(2)
calculations = st.spinner('Calculation in progress...')

# features = st.beta_container()
# interactive = st.beta_container()
# table = st.beta_expander("Show me the data in the table")
# about = st.beta_container()


with header:
    st.title("Numerical Methods for solving ODE")
    st.text("Interactive application for solving Ordinary Differential Equations...")
    show = st.selectbox("Show me the basic principle of the calculation (select the algorithm) :", METHODS)
    get_latex_method(show)

with sidebar:
    st.sidebar.header("User Input Parameters")
    equation = st.sidebar.selectbox("Select equation:", EQUATIONS)
    id_mod = EQUATIONS.index(equation)
    f = MODELS[id_mod]

    # st.sidebar.markdown("***")
    # st.sidebar.markdown("Constansts")
    # c = [
    #     st.sidebar.number_input("a0 = ", 0.0),
    #     st.sidebar.number_input("a1 = ", 0.0),
    #     st.sidebar.number_input("a3 = ", 0.0),
    #     st.sidebar.number_input("a2 = ", 0.0)
    # ]
    # if id_mod < 4:
    #     c = [
    #         st.sidebar.number_input("a0 = ", 0.0),
    #         st.sidebar.number_input("a1 = ", 0.0)
    #     ]
    # elif id_mod == 4:
    #     c = [
    #         st.sidebar.number_input("a0 = ", 0.0),
    #         st.sidebar.number_input("a1 = ", 0.0),
    #         st.sidebar.number_input("a2 = ", 0.0)
    #     ]
    # elif id_mod == 5:
    #     c = [
    #         st.sidebar.number_input("a0 = ", 0.0),
    #         st.sidebar.number_input("a1 = ", 0.0),
    #         st.sidebar.number_input("a2 = ", 0.0),
    #         st.sidebar.number_input("a3 = ", 0.0)
    #     ]

    # equation = equation.replace("a0", str(c[0]))
    # equation = equation.replace("a1", str(c[1]))
    # equation = equation.replace("a2", str(c[2]))
    # equation = equation.replace("a3", str(c[3]))

    st.sidebar.markdown("***")
    st.sidebar.markdown("Initial conditions")
    if id_mod < 4:
        y0 = [st.sidebar.number_input("y(0) = ", value=0.0)]
        init_cond = ("y_{(0)}="+str(y0[0]))
    elif id_mod == 4:
        y0 = [st.sidebar.number_input("y'(0) = ", value=0.0),
              st.sidebar.number_input("y(0) = ", value=0.0)]
        init_cond = ("y'_{(0)}=" + str(y0[0]) + ",\\;y_{(0)}=" + str(y0[1]))
    elif id_mod == 5:
        y0 = [st.sidebar.number_input("y''(0) = ", value=0.0),
              st.sidebar.number_input("y'(0) = ", value=0.0),
              st.sidebar.number_input("y(0) = ", value=0.0)]
        init_cond = ("y''_{(0)}=" + str(y0[0]) + ",\\;y'_{(0)}=" + str(y0[1]) + ",\\;y_{(0)}=" + str(y0[2]))

    st.sidebar.markdown("***")
    st.sidebar.markdown("RKF45")
    h_min, h_max = st.sidebar.slider("Size of the step:", 0.0001, 1.000, (0.100, 0.750), 0.001)
    tol = st.sidebar.slider("Tolerance: (1e(x)):", -12, -2, -5)

with equations:
    st.subheader("Selected Equations")
    left_ode.write("Selected ordinary differential equation:")
    left_ode.latex(equation)
    left_ode.latex(init_cond)
    right_analytic.write("Analytic solution of selected ODE")

a = start_point_left.number_input("Start of the interval:", value=0)
b = start_point_right.number_input("End of the interval:", value=10)

# butt_request_a = request_analytical.button('Calculate analytically')
# butt_request_num = request_numerical.button('Calculate by numerical methods')

with calculations:
    n = st.slider("Number of steps:", 5, 100, 10)
    h = (b - a) / n
    st.write('Step size: ', h)

    request_analytical = False
    request_numerical = False
    update_requested = False
    rkf45_status = -1

    analytic = calc_analytical(a, b, f, y0)

    if st.button('Calculate by numerical methods'):
        numeric, rkf45, rkf45_status = calc_numerical(f, a, b, y0, n, h, h_min, h_max, tol)
        update_requested = True

    get_chart(numeric, rkf45, rkf45_status, analytic, update_requested)

    st.text("Note: You can select methods in the legend (for multiple selection hold SHIFT and click)")
    if -1 == rkf45_status:
        st.warning("RKF45 could not converge to the required tolerance with chose minimum step size, please adjust"
                   " the parameters.")

    with st.beta_expander('Show me the values in the table'):
        st.write(numeric)

# with about:
#     st.__version__
