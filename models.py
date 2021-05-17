import sympy as sp
from math import sin

#   purpose of this module is provide defined models of the systems of Ordinary Differential Equations

# global for Population Model
lm_a = 2.5
lm_m = 1000

# global vars for mathematical pendulum
mp_g = 9.81
mp_l = 1.5

# global vars for dumped oscillations
do_g = 9.81
do_m = 1   # mass
do_b = 14   # damping force
do_l = 0.20   # spring [m]

# global vars for skydiver
sk_c = 0.2
sk_m = 90
sk_g = 9.81

# manometer
mn_g = 9.81
mn_l = 0.3

# general second order
so_a = 1
so_b = 1
so_c = 1
so_d = 1

# general third order
o3_a = 1
o3_b = 1
o3_c = 1
o3_d = 1
o3_e = 1


def get_system_str():
    model = [""] * 7
    model[0] = "Population model"
    model[1] = "Mathematical pendulum"
    model[2] = "Mass-Damper-Spring"
    model[3] = "Skydiver"
    model[4] = "Manometer"
    model[5] = "ay''(x) * by'(x) + cy(x) = d"
    model[6] = "ay'''(x) * by''(x) + cy'(x) + dy(x)= e"
    return model


# --------------- MODELS ---------------  #
# MODEL FOR POPULATION MODEL
# 'y = a * y - (a * y**2) / M
def model_population(x, y):
    global lm_a
    global lm_m
    return lm_a * y[0] - (lm_a * y[0]**2) / lm_m


# FUNCTIONS FOR MATHEMATICAL PENDULUM
def model_pendulum(x, y):
    global mp_g
    global mp_l
    return y[1], -(mp_g / mp_l) * sin(y[0])


# FUNCTIONS FOR DAMPED OSCILLATION
# my'' + do_b y' + gm/l'y = 0
def model_damped_oscillations(x, y):
    global do_l
    global do_m
    global do_g
    global do_b
    return y[1], (-y[1] * do_b - (do_m * do_g)/do_l * y[0]) / do_m


# y'' = g - (c/m)y'^2
def model_skydiver(x, y):
    global sk_m
    global sk_c
    global sk_g
    return y[1], -sk_g + (sk_c / sk_m) * y[1]**2


# y'' + (2g/l) * y
def model_manometer(x, y):
    global mn_g
    global mn_l
    return y[1], -(2 * mn_g / mn_l) * y[0]


# GENERAL SECOND ORDER
# a y'' + b 'y + c y = d
def model_second_order(x, y):
    global so_a
    global so_b
    global so_c
    global so_d
    return y[1], (so_d - so_c * y[0] - so_b * y[1]) / so_a


# GENERAL THIRD ORDER
# a y''' + b y'' + c y' + d y = e
def model_third_order(x, y):
    global o3_a
    global o3_b
    global o3_c
    global o3_d
    global o3_e
    return y[2], y[1], (o3_e * sin(x) - o3_d * y[0] - o3_c * y[1] - o3_b * y[2]) / o3_a

# --------------- SETTERS ---------------  #
def set_params_population(a, m):
    global lm_a
    global lm_m
    lm_a = a
    lm_m = m
    return


def set_params_pendulum(l_var):
    global mp_l
    mp_l = l_var
    return


def set_params_damped_oscillations(param_m, param_b, param_l):
    global do_m
    global do_b
    global do_l
    do_m = param_m
    do_b = param_b
    do_l = param_l
    return


def set_params_skydiver(param_m):
    global sk_m
    sk_m = param_m
    return


def set_params_manometer(param_l):
    global mn_l
    mn_l = param_l
    return


def set_params_second_order(a, b, c, d):
    global so_a
    global so_b
    global so_c
    global so_d
    so_a = a
    so_b = b
    so_c = c
    so_d = d
    return


def set_params_third_order(a, b, c, d, e):
    global o3_a
    global o3_b
    global o3_c
    global o3_d
    o3_a = a
    o3_b = b
    o3_c = c
    o3_d = d
    return


# --------------- GETTERS --------------  #
def get_params_population():
    global lm_a
    global lm_m
    constants = ("a = " + str(lm_a) + ";\\, M=" + str(lm_m))
    return constants


def get_params_pendulum():
    global mp_g
    global mp_l
    constants = "g = " + str(mp_g) + "\\;\\frac{N}{kg} " + ",\\;l = " + str(mp_l) + "\\;m\\;"
    return constants


def get_params_damped_oscillations():
    global do_l
    global do_m
    global do_g
    global do_b
    constants = ("l = " + str(do_l) + "\\ m " + ";\\ m = " + str(do_m) + "\\ kg;\\ g = " + str(do_g)
                 + "\\;\\frac{N}{kg} " + ";\\;b = " + str(do_b) + "\\;N\\;")
    return constants


def get_params_skydiver():
    global sk_g
    global sk_m
    global sk_c
    constants = ("g = " + str(sk_g) + "\\ ms^{-2} " + ";\\ m = " + str(sk_m) + "\\ kg;\\ c = " + str(sk_c))
    return constants


def get_params_manometer():
    global mn_g
    global mn_l
    constants = ("g = " + str(mn_g) + "\\ ms^{-2} " + ";\\ l = " + str(mn_l) + "\\ m")
    return constants


def get_params_second_order():
    global so_a
    global so_b
    global so_c
    global so_d
    constants = ("a = " + str(so_a) + ";\\, b=" + str(so_b) + ";\\, c=" + str(so_c) + ";\\, d=" + str(so_d))
    return constants


def get_params_third_order():
    global o3_a
    global o3_b
    global o3_c
    global o3_d
    constants = ("a = " + str(o3_a) + ";\\, b=" + str(o3_b) + ";\\, c=" + str(o3_c) + ";\\, d="
                 + str(o3_d) + ";\\, e =" + str(o3_e))
    return constants


# --------------- EQUATIONS --------------  #
def get_formulas_population(y0):
    global lm_a
    global lm_m
    a = sp.symbols('a')
    m = sp.symbols('M')
    x = sp.symbols('x')
    y = sp.Function('y')
    ode_symbol = sp.Eq(y(x).diff(x), - ((a * y(x)**2) / m) + (a * y(x)))
    analytic = sp.dsolve(ode_symbol)
    return sp.latex(ode_symbol), sp.latex(analytic)


def get_formulas_pendulum():
    l = sp.symbols('l')
    g = sp.symbols('g')
    x = sp.symbols('x')
    y = sp.Function('y')
    ode_symbol = sp.Eq(y(x).diff(x, x) + g / l * sp.sin(y(x)))
    ode = sp.Eq(y(x).diff(x, x) + mp_g/mp_l * y(x))
    analytic = sp.dsolve(ode)
    return sp.latex(ode_symbol), sp.latex(analytic)



def get_formulas_damped_oscillations(y0):
    global do_l
    global do_m
    global do_g
    global do_b
    l = sp.symbols('l')
    m = sp.symbols('m')
    g = sp.symbols('g')
    b = sp.symbols('b')
    x = sp.symbols('x')
    y = sp.Function('y')
    ode_symbol = sp.Eq(m * y(x).diff(x, x) + b * y(x).diff(x) + (g * m) / l * y(x))
    analytic = sp.dsolve(ode_symbol)
    return sp.latex(ode_symbol), sp.latex(analytic)


def get_formulas_skydiver(y0):
    g = sp.symbols('g')
    m = sp.symbols('m')
    c = sp.symbols('c')
    x = sp.symbols('x')
    y = sp.Function('y')
    ode_symbol = sp.Eq(y(x).diff(x, x) + g - (c/m) * y(x).diff(x))
    analytic = sp.dsolve(ode_symbol)
    return sp.latex(ode_symbol), sp.latex(analytic)


def get_formulas_manometer(y0):
    g = sp.symbols('g')
    l = sp.symbols('l')
    x = sp.symbols('x')
    y = sp.Function('y')
    ode_symbol = sp.Eq(y(x).diff(x, x) + 2 * g/l * y(x))
    analytic = sp.dsolve(ode_symbol)
    return sp.latex(ode_symbol), sp.latex(analytic)


def get_formulas_second_order(y0):
    global so_a
    global so_b
    global so_c
    global so_d
    a = sp.symbols('a')
    b = sp.symbols('b')
    c = sp.symbols('c')
    d = sp.symbols('d')
    x = sp.symbols('x')
    y = sp.Function('y')
    ode_symbol = sp.Eq(a * y(x).diff(x, x) + b * y(x).diff(x) + c * y(x), d)
    analytic = sp.dsolve(ode_symbol)
    return sp.latex(ode_symbol), sp.latex(analytic)


def get_formulas_third_order(y0):
    global o3_a
    global o3_b
    global o3_c
    global o3_d
    global o3_e
    a = sp.symbols('a')
    b = sp.symbols('b')
    c = sp.symbols('c')
    d = sp.symbols('d')
    e = sp.symbols('e')
    x = sp.symbols('x')
    y = sp.Function('y')
    ode_symbol = sp.Eq(a * y(x).diff(x, x, x) + b * y(x).diff(x, x) + c * y(x).diff(x, x) + d * y(x).diff(x), e)
    analytic = sp.dsolve(ode_symbol)
    return sp.latex(ode_symbol), sp.latex(analytic)
