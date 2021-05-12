import base64
import numMethods
import models
import latex_methods as lm
import numpy as np
import streamlit as st
import pandas as pd
import altair as alt
from functools import wraps
from scipy.integrate import odeint
# from functools import wraps


st.set_page_config(layout="wide")
pdf = '[Description of ODE](https://github.com/mkaraf/streamlit-numMethods/blob/4261033118c6848c392ed7cae4c2e86b7829a3b4/populacni_model.pdf)'
link = '[ODE Math Laboratory](http://calculuslab.deltacollege.edu/ODE/ODE-h.html)'

METHODS = ["NONE", "Euler", "Heun", "Runge-Kutta 2", "Runge-Kutta 3", "Runge-Kutta 4", "Adams-Bashforth 2",
           "Adams-Bashforth 3", "Adams-Bashforth 4", "Adams-Bashforth-Moulton", "Runge-Kutta Fehlberg 45"]

EQUATIONS = models.get_system_str()
MODELS = [models.model_math_pendulum, models.model_do, models.model_population, models.model_so]
ANALYTIC = [models.analytic_math_pendulum, models.get_analytical_real, models.analytic_population_model]


analytic = pd.DataFrame(columns=['Analytical'])
table = pd.DataFrame(columns=['Analytical','Euler', 'Heun', 'RK2', 'RK3', 'RK4', 'AB2', 'AB3', 'AB4', 'ABM4_PC'])
numeric = pd.DataFrame(columns=['Euler', 'Heun', 'RK2', 'RK3', 'RK4', 'AB2', 'AB3', 'AB4', 'ABM4_PC'])
rkf45 = pd.DataFrame(columns=['RKF45'])
analytic.index.name = 'x'
numeric.index.name = 'x'
table.index.name = 'x'
rkf45.index.name = 'x'
first_run = True
rkf45_status = 0


def flip(func):
    # Create a new function from the original with the arguments reversed
    @wraps(func)
    def newfunc(*args):
        return func(*args[::-1])
    return newfunc


def get_table_download_link_csv(df):
    csv = df.to_csv(index=False)
    csv = df.to_csv().encode()
    #b64 = base64.b64encode(csv.encode()).decode()
    b64 = base64.b64encode(csv).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="dataframe.csv" target="_blank">Download csv file</a>'
    return href


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
        combine = alt.layer(chart, line_chart)
    else:
        combine = chart
    st.altair_chart(combine | make_selector)
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
    df_an = pd.DataFrame({
        'Analytical': an,
    },
        index=num.index
    )
    result = pd.concat([df_an, num], axis=1)
    return result


def calc_analytical_chart(a_min, b_max, init_con, step):
    # number_of_steps = round((b_max - a_min) / step) + 1
    x = np.linspace(a_min, b_max, n * 100 + 1)
    # x = np.linspace(a_min, b_max, number_of_steps * 10)
    odeintern = odeint(flip(f), init_con, x)[:, 0]
    odeinter_subset = odeintern[0::100]
    # solution = [0] * number_of_steps
    # for i in range(number_of_steps):
    #     solution[i] = analytic_formula(x[i], init_con)
    pd_analytic = pd.DataFrame({'Analytical': odeintern}
                               , index=x)
    return pd_analytic, odeinter_subset


def calc_truncation(df_qdeviation):
    size = len(df_qdeviation)

    # DATAFRAME
    df_global_trunc = pd.DataFrame({
        'Euler': (df_qdeviation['Analytical'] - df_qdeviation['Euler']) ** 2,
        'Heun': (df_qdeviation['Analytical'] - df_qdeviation['Heun']) ** 2,
        'RK2': (df_qdeviation['Analytical'] - df_qdeviation['RK2']) ** 2,
        'RK3': (df_qdeviation['Analytical'] - df_qdeviation['RK3']) ** 2,
        'RK4': (df_qdeviation['Analytical'] - df_qdeviation['RK4']) ** 2,
        'AB2': (df_qdeviation['Analytical'] - df_qdeviation['AB2']) ** 2,
        'AB3': (df_qdeviation['Analytical'] - df_qdeviation['AB3']) ** 2,
        'AB4': (df_qdeviation['Analytical'] - df_qdeviation['AB4']) ** 2,
        'ABM4_PC': (df_qdeviation['Analytical'] - df_qdeviation['ABM4_PC']) ** 2
    },
        index=df_qdeviation.index
    )
    df_global_trunc = df_global_trunc.sum()
    max = df_global_trunc.max()
    df_global_trunc = df_global_trunc / max
    df_global_trunc.index.name = 'x'
    bchart_source = df_global_trunc.reset_index().melt('x', var_name='method', value_name='y')

    selection = alt.selection_multi(fields=['x'])
    color = alt.condition(selection,
                          alt.Color('x:N', legend=None),
                          alt.value('lightgray'))

    bars = alt.Chart(bchart_source, title='Sum of the quadratic deviations').mark_bar().encode(
        x='y:Q',
        y=alt.Y('x:N', sort=alt.EncodingSortField(field='y', order='ascending')),
        color=color,
        tooltip=[alt.Tooltip('x:N'),
                 alt.Tooltip('y:Q')]
    ).properties(width=800, height=400).transform_filter(selection)

    make_selector = alt.Chart(bchart_source).mark_rect().encode(
        y=alt.Y('x:N', axis=alt.Axis(orient='right')),
        color=color
    ).add_selection(
        selection
    )

    st.altair_chart(bars | make_selector)
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
    model_str = ""
    params = ""

    if id_mod == 0:
        l = st.sidebar.number_input("l = ", value=1.5, min_value=0.1)
        models.set_var_math_pendulum(l)
        general_solution = ANALYTIC[id_mod]
    elif id_mod == 1:
        do_m = st.sidebar.number_input("m = ", value=1.0, min_value=0.1, key='kg')
        do_b = st.sidebar.number_input("damping force = ", value=14.0, min_value=0.1, key='N')
        do_l = st.sidebar.number_input("streched spring = ", value=0.2, min_value=0.1, key='m')
        general_solution, info = models.get_damp_osc_analytic(do_m, do_b, do_l)     ## TEST INFO REMOVE IT
    elif id_mod == 2:
        pm_m= st.sidebar.number_input("M = ", value=1000.0, min_value=0.1)
        pm_a = st.sidebar.number_input("a = ", value=2.1)
        models.set_pop_model_const(pm_a, pm_m)
        general_solution = ANALYTIC[id_mod]
    elif id_mod == 3:
        so_a = st.sidebar.number_input("a = ", value=1.0)
        so_b = st.sidebar.number_input("b = ", value=1.0)
        so_c = st.sidebar.number_input("c = ", value=1.0)
        so_d = st.sidebar.number_input("d = ", value=1.0)
        models.set_const_so(so_a, so_b, so_c,so_d)
        # general_solution = ANALYTIC[id_mod]

    st.sidebar.markdown("***")
    st.sidebar.markdown("Initial conditions")
    if id_mod == 0:
        y0 = [st.sidebar.number_input("C1 = ", value=1.0),
               st.sidebar.number_input("C2= ", value=1.0)]
        init_cond = ("C1=" + str(y0[0]) + ",\\;C2=" + str(y0[1]))
        params = models.get_const_math_pendulum()
        model_str, analytic_formula = models.get_math_form_analytic(y0)
    elif id_mod == 1:
        y0 = [st.sidebar.number_input("C1 = ", value=1.0),
              st.sidebar.number_input("C2 = ", value=1.0)]
        init_cond = ("C1=" + str(y0[0]) + ",\\;C2=" + str(y0[1]))
        params = models.get_const_do()
        model_str, analytic_formula = models.get_do_form_analytic(y0)
    elif id_mod == 2:
        y0 = [st.sidebar.number_input("C1 = ", value=1.0)]
        init_cond = ("C1=" + str(y0[0]))
        params = models.get_pop_model_const()
        model_str, analytic_formula = models.get_pop_model_formulas(y0)
    elif id_mod == 3:
        y0 = [st.sidebar.number_input("C1 = ", value=1.0),
              st.sidebar.number_input("C2 = ", value=1.0)]
        init_cond = ("C1=" + str(y0[0]) + ",\\;C2=" + str(y0[1]))
        params = models.get_const_so()
        model_str, analytic_formula = models.get_formulas_so(y0)

    st.sidebar.markdown("***")
    st.sidebar.markdown("RKF45")
    h_min, h_max = st.sidebar.slider("Size of the step:", 0.1, 1000.0, (100.0, 700.0), 0.1,
                                     help="Value is divided by 1000")
    h_min = h_min / 1000
    h_max = h_max / 1000
    st.sidebar.write('Minimal step = ', h_min)
    st.sidebar.write('Maximal step = ', h_max)

    tol = st.sidebar.slider("Tolerance:", -5, 10, 4, help="Computation method: 1e(-value)")
    tol = 1 * 10 ** (-tol)
    st.sidebar.write('Tolerance = ', tol)

with equations:
    st.subheader("Selected Equations")
    left_ode.write("Selected ordinary differential equation:")
    left_ode.latex(model_str)
    right_analytic.write("General solution of selected ODE:")
    right_analytic.latex(analytic_formula)
    left_ode.write('Initial conditions:')
    left_ode.latex(init_cond)
    right_analytic.write('Parameters:')
    right_analytic.latex(params)

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

            analytic, odeinter_subset = calc_analytical_chart(a, b, y0, 0.01)

            if st.button('Calculate by numerical methods'):
                numeric, rkf45, rkf45_status = calc_numerical(f, a, b, y0, n, h, h_min, h_max, tol)
                update_requested = True
            get_chart(numeric, rkf45, rkf45_status, analytic, update_requested)

            st.text("Note: You can select methods in the legend (for multiple selection hold SHIFT and click)")
            if -1 == rkf45_status:
                st.warning("RKF45 could not converge to the required tolerance with chose minimum step size,"
                           " please adjust the parameters.")

            if update_requested:
                st.markdown("***")
                table = get_table(numeric, odeinter_subset)
                st.write(table)
                st.markdown(get_table_download_link_csv(table), unsafe_allow_html=True)
                st.subheader("Global Truncation")
                calc_truncation(table)

with footer:
    st.markdown('***')
    st.subheader("Useful links:")
    st.markdown(link, unsafe_allow_html=True)
    st.markdown(pdf, unsafe_allow_html=True)
    st.write("Streamlit version: " + st.__version__)
