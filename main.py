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

st.set_page_config(
    page_title="Numeric Methods",
    initial_sidebar_state="auto",
    layout='wide'
)

pdf = '[Explanation of exercises](https://github.com/mkaraf/streamlit-numMethods/blob/4261033118c6848c392ed7cae4c2e86b7829a3b4/populacni_model.pdf)'
link = '[ODE Math Laboratory](http://calculuslab.deltacollege.edu/ODE/ODE-h.html)'

METHODS = ["NONE", "Euler", "Heun", "Runge-Kutta 2", "Runge-Kutta 3", "Runge-Kutta 4", "Adams-Bashforth 2",
           "Adams-Bashforth 3", "Adams-Bashforth 4", "Adams-Bashforth-Moulton", "Runge-Kutta Fehlberg 45"]

EQUATIONS = models.get_system_str()

analytic = pd.DataFrame(columns=['Analytical'])
table = pd.DataFrame(columns=['Analytical','Euler', 'Heun', 'RK2', 'RK3', 'RK4', 'AB2', 'AB3', 'AB4', 'ABM4_PC'])
numeric_df = pd.DataFrame(columns=['Euler', 'Heun', 'RK2', 'RK3', 'RK4', 'AB2', 'AB3', 'AB4', 'ABM4_PC'])
rkf45 = pd.DataFrame(columns=['RKF45'])
analytic.index.name = 'x'
numeric_df.index.name = 'x'
table.index.name = 'x'
rkf45.index.name = 'x'



def flip(func):
    # Create a new function from the original with the arguments reversed
    @wraps(func)
    def newfunc(*args):
        return func(*args[::-1])
    return newfunc


def get_table_download_link_csv(df):
    # csv = df.to_csv(index=False)
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


def calc_numerical(requested):
    if requested:
        x = np.linspace(a, b, n + 1)

        euler = np.array(numMethods.euler(numeric_f, y0, x, h))[:, 0]
        heun = np.array(numMethods.heun(numeric_f, y0, x, h))[:, 0]
        rk2 = np.array(numMethods.rkx(numeric_f, y0, x, h, 2))[:, 0]
        rk3 = np.array(numMethods.rkx(numeric_f, y0, x, h, 3))[:, 0]
        rk4 = np.array(numMethods.rkx(numeric_f, y0, x, h, 4))[:, 0]
        ab2 = np.array(numMethods.abx(numeric_f, y0, x, h, 2))[:, 0]
        ab3 = np.array(numMethods.abx(numeric_f, y0, x, h, 3))[:, 0]
        ab4 = np.array(numMethods.abx(numeric_f, y0, x, h, 4))[:, 0]
        abm4_pc = np.array(numMethods.abm4_pc(numeric_f, y0, x, h))[:, 0]
        rkf_x, rkf_y, rkf45_stat = numMethods.rkf45(numeric_f, y0, a, b, tol, h_max, h_min)
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

        if 0 == rkf45_stat:
            df_rkf45 = pd.DataFrame({
                'RKF45': rkf_y
            },
                index=rkf_x
            )
            df_rkf45.index.name = 'x'
        else:
            df_rkf45 = 0
    else:
        df_num_sol = numeric_df
        df_rkf45 = 0
        rkf45_stat = -2

    return df_num_sol, df_rkf45, rkf45_stat


def get_table(num, an):
    df_an = pd.DataFrame({
        'Analytical': an,
    },
        index=num.index
    )
    result = pd.concat([df_an, num], axis=1)
    return result


def calc_analytical_chart():
    x = np.linspace(a, b, n * 100 + 1)
    result_all = odeint(flip(numeric_f), y0, x)[:, 0]
    result_subset = result_all[0::100]
    pd_result_all = pd.DataFrame({'Analytical': result_all},
                                 index=x)
    return pd_result_all, result_subset


def do_comparison(df_comparison):
    # DATAFRAME
    df_global_trunc = pd.DataFrame({
        'Euler': (df_comparison['Analytical'] - df_comparison['Euler']) ** 2,
        'Heun': (df_comparison['Analytical'] - df_comparison['Heun']) ** 2,
        'RK2': (df_comparison['Analytical'] - df_comparison['RK2']) ** 2,
        'RK3': (df_comparison['Analytical'] - df_comparison['RK3']) ** 2,
        'RK4': (df_comparison['Analytical'] - df_comparison['RK4']) ** 2,
        'AB2': (df_comparison['Analytical'] - df_comparison['AB2']) ** 2,
        'AB3': (df_comparison['Analytical'] - df_comparison['AB3']) ** 2,
        'AB4': (df_comparison['Analytical'] - df_comparison['AB4']) ** 2,
        'ABM4_PC': (df_comparison['Analytical'] - df_comparison['ABM4_PC']) ** 2
    },
        index=df_comparison.index
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


# PENDULUM
def set_inputs_pendulum():
    models.set_params_pendulum(mp_l)

    # f_analytic = models.
    f_numeric = models.model_pendulum

    str_init_cond = ("y(0) =" + str(y0[0]) + ",\\;y'(0)=" + str(y0[1]))
    str_params = models.get_params_pendulum()
    str_ode, str_general = models.get_formulas_pendulum()

    return f_numeric, str_init_cond, str_params, str_ode, str_general


# DAMPED PENDULUM
def set_inputs_model_oscillations():
    models.set_params_damped_oscillations(do_m, do_b, do_l)

    f_numeric = models.model_damped_oscillations

    str_init_cond = ("y(0) =" + str(y0[0]) + ",\\;y'(0)=" + str(y0[1]))
    str_params = models.get_params_damped_oscillations()
    str_ode, str_general = models.get_formulas_damped_oscillations(y0)

    return f_numeric, str_init_cond, str_params, str_ode, str_general


# POPULATION MODEL
def set_inputs_model_population():
    models.set_params_population(pm_a, pm_m)

    f_numeric = models.model_population

    str_init_cond = ("y(0)=" + str(y0[0]))
    str_params = models.get_params_population()
    str_ode, str_general = models.get_formulas_population(y0)

    return f_numeric, str_init_cond, str_params, str_ode, str_general


# GENERAL SECOND ORDER
def set_inputs_second_order():
    models.set_params_second_order(so_a, so_b, so_c, so_d)

    f_numeric = models.model_second_order

    str_init_cond = ("y(0) =" + str(y0[0]) + ",\\;y'(0)=" + str(y0[1]))
    str_params = models.get_params_second_order()
    str_ode, str_general = models.get_formulas_second_order(y0)

    return f_numeric, str_init_cond, str_params, str_ode, str_general


def set_inputs_third_order():
    models.set_params_third_order(o3_a, o3_b, o3_c, o3_d, o3_e)

    f_numeric = models.model_third_order

    str_init_cond = ("y(0) =" + str(y0[0]) + ",\\;y'(0)=" + str(y0[1])+ ",\\;y''(0)=" + str(y0[2]))
    str_params = models.get_params_third_order()
    str_ode, str_general = models.get_formulas_third_order(y0)

    return f_numeric, str_init_cond, str_params, str_ode, str_general


def set_inputs_skydiver():
    models.set_params_skydiver(sk_m)

    f_numeric = models.model_skydiver

    str_init_cond = ("y(0) =" + str(y0[0]) + ",\\;y'(0)=" + str(y0[1]))
    str_params = models.get_params_skydiver()
    str_ode, str_general = models.get_formulas_skydiver(y0)
    return f_numeric, str_init_cond, str_params, str_ode, str_general


def set_inputs_manometer():
    models.set_params_manometer(mn_l)

    f_numeric = models.model_manometer

    str_init_cond = ("y(0) =" + str(y0[0]) + ",\\;y'(0)=" + str(y0[1]))
    str_params = models.get_params_manometer()
    str_ode, str_general = models.get_formulas_manometer(y0)
    return f_numeric, str_init_cond, str_params, str_ode, str_general


header = st.beta_container()
sidebar = st.beta_container()
equations_cont = st.beta_container()
equations_left_cont, equations_right_cont = st.beta_columns(2)
results_cont = st.spinner('Calculation in progress...')
footer_cont = st.beta_container()


with header:
    st.title("Numerical Methods for solving ODE")
    st.text("Interactive application for solving Ordinary Differential Equations...")
    show = st.selectbox("Show me the basic principle of the calculation (select the algorithm) :", METHODS)
    get_latex_method(show)

with sidebar:
    st.sidebar.header("User Input Parameters")
    st.sidebar.markdown(pdf, unsafe_allow_html=True)
    equation_select = st.sidebar.selectbox("Select equation:", EQUATIONS)

    st.sidebar.markdown("Parameters:")

    if equation_select == "Mathematical pendulum":
        mp_l = st.sidebar.number_input("l = ", value=1.5, min_value=0.1)
        st.sidebar.markdown("Initial conditions:")
        y0 = [st.sidebar.number_input("y(0) = ", value=1.0),
              st.sidebar.number_input("y'(0) = ", value=1.0)]
        numeric_f, init_cond, params, ode_str, general_str = set_inputs_pendulum()

    elif equation_select == "Mass-Damper-Spring":
        do_m = st.sidebar.number_input("mass = ", value=1.0, min_value=0.1, key='kg')
        do_b = st.sidebar.number_input("damping force = ", value=14.0, min_value=0.1, key='N')
        do_l = st.sidebar.number_input("stretched spring = ", value=0.2, min_value=0.1, key='m')
        st.sidebar.markdown("Initial conditions:")
        y0 = [np.radians(st.sidebar.number_input("y(0) = ", value=1.0)),
              st.sidebar.number_input("y'(0) = ", value=1.0)]
        numeric_f, init_cond, params, ode_str, general_str = set_inputs_model_oscillations()

    elif equation_select == "Population model":
        pm_m = st.sidebar.number_input("M = ", value=1000.0, min_value=0.1)
        pm_a = st.sidebar.number_input("a = ", value=2.1)
        st.sidebar.markdown("Initial conditions:")
        y0 = [st.sidebar.number_input("y(0) = ", value=1.0)]
        numeric_f, init_cond, params, ode_str, general_str = set_inputs_model_population()

    elif equation_select == "ay''(x) * by'(x) + cy(x) = d":
        so_a = st.sidebar.number_input("a = ", value=1.0, min_value=0.1)
        so_b = st.sidebar.number_input("b = ", value=1.0)
        so_c = st.sidebar.number_input("c = ", value=1.0)
        so_d = st.sidebar.number_input("d = ", value=1.0)
        st.sidebar.markdown("Initial conditions:")
        y0 = [st.sidebar.number_input("y(0) = ", value=1.0),
              st.sidebar.number_input("y'(0) = ", value=1.0)]
        numeric_f, init_cond, params, ode_str, general_str = set_inputs_second_order()

    elif equation_select == 'Skydiver':
        sk_m = st.sidebar.number_input("mass = ", value=90.0, min_value=1.0, key='kg')
        st.sidebar.markdown("Initial conditions:")
        y0 = [st.sidebar.number_input("y(0) = ", value=1.0),
              st.sidebar.number_input("y'(0) = ", value=1.0)]
        numeric_f, init_cond, params, ode_str, general_str = set_inputs_skydiver()

    elif equation_select == 'Manometer':
        mn_l = st.sidebar.number_input("length = ", value=0.5, min_value=0.1, key='m')
        st.sidebar.markdown("Initial conditions:")
        y0 = [st.sidebar.number_input("y(0) = ", value=0.2),
              st.sidebar.number_input("y'(0) = ", value=0.0)]
        numeric_f, init_cond, params, ode_str, general_str = set_inputs_manometer()

    elif equation_select == "ay'''(x) * by''(x) + cy'(x) + dy(x)= e":
        o3_a = st.sidebar.number_input("a = ", value=1.0, min_value=0.1)
        o3_b = st.sidebar.number_input("b = ", value=1.0)
        o3_c = st.sidebar.number_input("c = ", value=1.0)
        o3_d = st.sidebar.number_input("d = ", value=1.0)
        o3_e = st.sidebar.number_input("e = ", value=1.0)
        st.sidebar.markdown("Initial conditions:")
        y0 = [st.sidebar.number_input("y(0) = ", value=1.0),
              st.sidebar.number_input("y'(0) = ", value=1.0),
              st.sidebar.number_input("y''(0) = ", value=1.0)]
        numeric_f, init_cond, params, ode_str, general_str = set_inputs_third_order()

    st.sidebar.markdown("***")
    st.sidebar.markdown("RKF45")
    h_min, h_max = st.sidebar.slider("Size of the step:", 0.1, 10000.0, (1.0, 7000.0), 0.1,
                                     help="Value is divided by 1000")
    tol = st.sidebar.slider("Tolerance:", 1, 10, 3,
                            help="Higher value means higher precision,"
                                 "Computation method: 1e(-value)")
    h_min = h_min / 10000
    h_max = h_max / 10000
    tol = 1 * 10 ** (-tol)
    st.sidebar.write('Minimal step = ', h_min)
    st.sidebar.write('Maximal step = ', h_max)
    st.sidebar.write('Tolerance = ', tol)

with equations_cont:
    with st.spinner('Calculation in progress...'):
        st.subheader("Selected Equations")
        equations_left_cont.write("Selected ordinary differential equation:")
        equations_left_cont.latex(ode_str)
        equations_right_cont.write("General solution of selected ODE:")
        equations_right_cont.latex(general_str)
        equations_left_cont.write('Initial conditions:')
        equations_left_cont.latex(init_cond)
        equations_right_cont.write('Parameters:')
        equations_right_cont.latex(params)
        a = equations_left_cont.number_input("Start of the interval:", value=0)
        b = equations_right_cont.number_input("End of the interval:", value=10)

with results_cont:
    with st.spinner('Calculation in progress...'):
        if b <= a:
            st.error('End of the interval has to be higher than start')
        else:
            n = st.slider("Number of steps:", 5, 100, 30)
            h = (b - a) / n
            st.write('Step size: ', h)

            update_requested = False

            analytic_ch, analytic_t = calc_analytical_chart()

            if st.button('Calculate by numerical methods'):
                update_requested = True

            numeric_df, rkf45, rkf45_status = calc_numerical(update_requested)
            if -1 == rkf45_status:
                st.warning("RKF45 could not converge to the required tolerance with chose minimum step size,"
                           " please adjust the parameters.")

            st.subheader("Results")
            get_chart(numeric_df, rkf45, rkf45_status, analytic_ch, update_requested)
            st.text("Note: You can select methods in the legend (for multiple selection hold SHIFT and click)")

            if update_requested:
                st.markdown("***")
                table = get_table(numeric_df, analytic_t)
                st.write(table)
                st.markdown(get_table_download_link_csv(table), unsafe_allow_html=True)
                st.subheader("Comparison of results")
                do_comparison(table)

            st.markdown('***')
            st.subheader("Useful links:")
            st.markdown(link, unsafe_allow_html=True)
            st.write("Streamlit version: " + st.__version__)
