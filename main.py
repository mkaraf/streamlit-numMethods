import numMethods
import models
import latex_methods as lm
import numpy as np
import streamlit as st
import pandas as pd
import altair as alt
from functools import wraps
# from scipy.integrate import odeint
# from functools import wraps


st.set_page_config(layout="wide")
link = '[ODE Math Laboratory](http://calculuslab.deltacollege.edu/ODE/ODE-h.html)'

METHODS = ["NONE", "Euler", "Heun", "Runge-Kutta 2", "Runge-Kutta 3", "Runge-Kutta 4", "Adams-Bashforth 2",
           "Adams-Bashforth 3", "Adams-Bashforth 4", "Adams-Bashforth-Moulton", "Runge-Kutta Fehlberg 45"]

EQUATIONS = models.get_system_str()
MODELS = [models.model_math_pendulum, models.model_do]
ANALYTIC = [models.analytic_math_pendulum, models.get_analytical_real]


analytic = pd.DataFrame(columns=['Analytical'])
table = pd.DataFrame(columns=['Analytical','Euler', 'Heun', 'RK2', 'RK3', 'RK4', 'AB2', 'AB3', 'AB4', 'ABM4_PC'])
numeric = pd.DataFrame(columns=['Euler', 'Heun', 'RK2', 'RK3', 'RK4', 'AB2', 'AB3', 'AB4', 'ABM4_PC'])
rkf45 = pd.DataFrame(columns=['RKF45'])
analytic.index.name = 'x'
numeric.index.name = 'x'
table.index.name = 'x'
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
    rkf_x, rkf_y, rkf45_stat = np.array(numMethods.rkf45(model, init_con, a, b, tolerance,
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


def get_table(num, an):
    result = pd.concat([an, num], axis=1)
    return result


def calc_analytical_chart(a_min, b_max, analytic_formula, init_con, step):
    number_of_steps = round((b_max - a_min) / step) + 1
    x = np.linspace(a_min, b_max, number_of_steps)
    # solution = odeint(flip(model), init_con, x)
    solution = [0] * number_of_steps
    for i in range(number_of_steps):
        solution[i] = analytic_formula(x[i], init_con)
    pd_analytic = pd.DataFrame({'Analytical': solution}, index=x)
    return pd_analytic


def calc_truncation(vals):
    last_row = vals.iloc[[-1]]
    # st.write(last_row)

    # DATAFRAME
    df_global_trunc = pd.DataFrame({
        'Euler': abs(last_row['Analytical'] - last_row['Euler']),
        'Heun': abs(last_row['Analytical'] - last_row['Heun']),
        'RK2': abs(last_row['Analytical'] - last_row['RK2']),
        'RK3': abs(last_row['Analytical'] - last_row['RK3']),
        'RK4': abs(last_row['Analytical'] - last_row['RK4']),
        'AB2': abs(last_row['Analytical'] - last_row['AB2']),
        'AB3': abs(last_row['Analytical'] - last_row['AB3']),
        'AB4': abs(last_row['Analytical'] - last_row['AB4']),
        'ABM4_PC': abs(last_row['Analytical'] - last_row['ABM4_PC'])
    },
        index=last_row.index
    )
    df_global_trunc.index.name = 'x'

    bchart_source = df_global_trunc.reset_index().melt('x', var_name='method', value_name='y')
    bars = alt.Chart(bchart_source, title='Global Truncation').mark_bar().encode(
        x='y:Q',
        y=alt.Y('method:N',sort=alt.EncodingSortField(field='y', order='ascending', op='sum'),),
        color=alt.Color('method', legend=None)
    ).properties(width=800, height=400)

    st.altair_chart(bars)
    return


header = st.beta_container()
sidebar = st.beta_container()
equations = st.beta_container()
left_ode, right_analytic = st.beta_columns(2)
init_conditions = st.beta_container()

control_panel = st.beta_container()
start_point_left, start_point_right = st.beta_columns(2)
request_analytical, request_numerical = st.beta_columns(2)
calculations = st.beta_container()
about = st.beta_container()
footer = st.beta_container()


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
    general_solution = ANALYTIC[id_mod]

    if id_mod == 0:
        l = st.sidebar.number_input("l = ", value=1.5, min_value=0.1)
        models.set_var_math_pendulum(l)

    if id_mod == 1:
        do_m = st.sidebar.number_input("m = ", value=1.0, min_value=0.1, key='kg')
        do_b = st.sidebar.number_input("damping force = ", value=14.0, min_value=0.1, key='N')
        do_l = st.sidebar.number_input("streched spring = ", value=0.2, min_value=0.1, key='m')
        general_solution, s = models.get_damp_osc_analytic(do_m, do_b, do_l)
        st.sidebar.write(s)

    st.sidebar.markdown("***")
    st.sidebar.markdown("Initial conditions")
    if id_mod < -1:
        y0 = [st.sidebar.number_input("y(0) = ", value=1.0)]
        init_cond = ("y_{(0)}="+str(y0[0]))
    elif id_mod == 0:
        y0 = [np.radians(st.sidebar.number_input("C1 = ", value=1.0, help="input in °")),
              np.radians(st.sidebar.number_input("C2= ", value=1.0, help="input in °/s"))]
        init_cond = ("C1=" + str(y0[0]) + "\\;radians" + ",\\;C2=" + str(y0[1]) + "\\;radians/s")
        model = models.get_model_str(id_mod)
    elif id_mod == 1:
        y0 = [st.sidebar.number_input("C1 = ", value=1.0),
              st.sidebar.number_input("C2 = ", value=1.0)]
        init_cond = ("C1=" + str(y0[0]) + ",\\;C2=" + str(y0[1]))
        model = models.get_model_str(id_mod)

    st.sidebar.markdown("***")
    st.sidebar.markdown("RKF45")
    h_min, h_max = st.sidebar.slider("Size of the step:", 0.1, 1000.0, (100.0, 700.0), 0.1,
                                     help="Value is divided by 1000")
    h_min = h_min / 1000
    h_max = h_max / 1000
    tol = st.sidebar.slider("Tolerance:", -5, 10, 4, help="Computation method: 1e(-value)")
    tol = 1 * 10 ** (-tol)
    st.sidebar.write('step_interval = ' + str([h_min, h_max]))
    st.sidebar.write('tolerance = ', tol)

with equations:
    st.subheader("Selected Equations")
    left_ode.write("Selected ordinary differential equation:")
    left_ode.latex(model)
    right_analytic.write("Analytic solution of selected ODE:")
    # right_analytic.latex(sp.latex("y[1] * cos((mp_g / mp_l) ** (1 / 2) * x) + y[0] * sin((mp_g / mp_l) ** (1 / 2) * x)"))
    right_analytic.latex('y = C1 \\cos(\\sqrt{\\frac{g}{l}}x) + C2 \\sin(\\sqrt{\\frac{g}{l}}x)')

    left_ode.latex(init_cond)
    left_ode.latex(models.get_const_math_pendulum())

    # l = right_analytic.number_input("l = ", value=1.5, min_value=0.1)
    # models.set_var_math_pendulum(l)


a = start_point_left.number_input("Start of the interval:", value=0)
b = start_point_right.number_input("End of the interval:", value=10)

with calculations:
    with st.spinner('Calculation in progress...'):
        if b <= a:
            st.error('End of the interval has to be higher than start')
        else:
            n = st.slider("Number of steps:", 5, 100, 30)
            h = (b - a) / n
            st.write('Step size: ', h)

            update_requested = False
            rkf45_status = -1

            analytic = calc_analytical_chart(a, b, general_solution, y0, 0.01)

            if st.button('Calculate by numerical methods'):
                numeric, rkf45, rkf45_status = calc_numerical(f, a, b, y0, n, h, h_min, h_max, tol)
                update_requested = True

            get_chart(numeric, rkf45, rkf45_status, analytic, update_requested)

            st.text("Note: You can select methods in the legend (for multiple selection hold SHIFT and click)")
            if -1 == rkf45_status:
                st.warning("RKF45 could not converge to the required tolerance with chose minimum step size, please adjust"
                           " the parameters.")

            table = get_table(numeric, calc_analytical_chart(a, b, general_solution, y0, h))

            if update_requested:
                st.markdown("***")
                st.write(table)
                st.subheader("Global Truncation")
                calc_truncation(table)


with footer:
    st.markdown('***')
    st.subheader("Useful links:")
    st.markdown(link, unsafe_allow_html=True)
    st.write("Streamlit version: " + st.__version__)
