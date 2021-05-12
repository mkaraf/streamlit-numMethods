import sympy as sp
from math import sin, cos, exp, pi
import numpy as np

#   purpose of this module is provide defined models of the systems of Ordinary Differential Equations

# global for Population Model
lm_a = 2.1
lm_m = 1000

# global vars for math pendulum
mp_g = 9.81
mp_l = 1.5

# global vars for dumped oscilations
do_roots = [0] * 2
k = 0.20
do_g = 9.81
do_m = 1   # mass
do_b = 14   # damping force
do_l = 0.20   # spring [m]


# general second order
so_a = 1
so_b = 1
so_c = 1
so_d = 1


def get_system_str():
    model = [""] * 4
    model[0] = "Mathematical pendulum"
    model[1] = "Damped oscillations"
    model[2] = "Population model"
    model[3] = "ay''(x) * by'(x) + cy(x) = d"
    return model


# SECOND ORDER ODE
# a y'' + b 'y + c y = d
def model_so(x, y):
    global so_a
    global so_b
    global so_c
    global so_d
    return y[1], (so_d - so_c * y[0] - so_b * y[1]) / so_a


def set_const_so(a, b, c, d):
    global so_a
    global so_b
    global so_c
    global so_d
    so_a = a
    so_b = b
    so_c = c
    so_d = d
    return


def get_const_so():
    global so_a
    global so_b
    global so_c
    global so_d
    constants = ("a = " + str(so_a) + ";\\, b=" + str(so_b)+ ";\\, c=" + str(so_c)+ ";\\, d=" + str(so_d))
    return constants

def get_formulas_so(y0):
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
    # a y'' = d - b y' - c y
    ode_symbol = sp.Eq(y(x).diff(x, x), d - b * y(x).diff(x) - c * y(x))
    ode = sp.Eq(a*y(x).diff(x, x), so_d - so_b * y(x).diff(x) - so_c * y(x))
    # analytic = sp.dsolve(ode, ics={y(0): y0[0], y(x).diff(x).subs(x, 0): y0[1]})
    analytic = sp.dsolve(ode_symbol)
    return sp.latex(ode_symbol), sp.latex(analytic)

# FUNCTIONS FOR POPULATION MODEL
# 'y = a * y - (a * y**2) / M
def model_population(x, y):
    global lm_a
    global lm_m
    return lm_a * y[0] - (lm_a * y[0]**2) / lm_m


def get_pop_model_formulas(y0):
    global lm_a
    global lm_m
    a = sp.symbols('a')
    m = sp.symbols('M')
    x = sp.symbols('x')
    y = sp.Function('y')
    ode_symbol = sp.Eq(y(x).diff(x), - ((a * y(x)**2) / m) + (a * y(x)))
    ode = sp.Eq(y(x).diff(x), - ((lm_a * y(x)**2) / lm_m) + (lm_a * y(x)))
    # analytic = sp.dsolve(ode, ics={y(0): y0[0]})
    analytic = sp.dsolve(ode_symbol)
    return sp.latex(ode_symbol), sp.latex(analytic)


def analytic_population_model(x, y):
    global lm_a
    global lm_m
    return y[0] * exp(lm_a * x) / (1 + (1 / lm_m) * y[0] * exp(lm_a * x))


def get_pop_model_const():
    global lm_a
    global lm_m
    constants = ("a = " + str(lm_a) + ";\\, M=" + str(lm_m))
    return constants


def set_pop_model_const(a, m):
    global lm_a
    global lm_m
    lm_a = a
    lm_m = m
    return


# FUNCTIONS FOR MATHEMATICAL PENDULUM
def model_math_pendulum(x, y):
    global mp_g
    global mp_l
    return y[1], -(mp_g / mp_l) * sin(y[0])


def get_math_form_analytic(y0):
    # global mp_g
    # global mp_l
    l = sp.symbols('l')
    g = sp.symbols('g')
    x = sp.symbols('x')
    y = sp.Function('y')
    ode_symbol = sp.Eq(y(x).diff(x, x) + g / l * sp.sin(y(x)))
    ode = sp.Eq(y(x).diff(x, x) + mp_g/mp_l * y(x))
    # analytic = sp.dsolve(ode, ics={y(0): y0[0], y(x).diff(x).subs(x, 0): y0[1]})
    analytic = sp.dsolve(ode)
    return sp.latex(ode_symbol), sp.latex(analytic)
    # return ('y = C1 \\cos(\\sqrt{\\frac{g}{l}}x) + C2 \\sin(\\sqrt{\\frac{g}{l}}x)')
    # "y'' = -\\frac{g}{l}\\sin(y)"


def analytic_math_pendulum(x, y):
    return y[0] * cos((mp_g / mp_l) ** (1 / 2) * x) + y[1] * sin((mp_g / mp_l) ** (1 / 2) * x)


def get_const_math_pendulum():
    global mp_g
    global mp_l
    constants = "g = " + str(mp_g) + "\\;\\frac{N}{kg} " + ",\\;l = " + str(mp_l) + "\\;m\\;"
    return constants


def set_var_math_pendulum(l):
    global mp_l
    mp_l = l
    return


# FUNCTIONS FOR DAMPED OSCILLATION
# my'' + do_b y' + gm/l'y = 0
def model_do(x, y):
    global do_l
    global do_m
    global do_g
    global do_b
    return y[1], (-y[1] * do_b - (do_m * do_g)/do_l * y[0]) / do_m


# my'' + do_b y' + gm/l'y = 0
def get_do_form_analytic(y0):
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
    ode_symbol = sp.Eq(m * y(x).diff(x, x) + b * y(x) .diff(x) + (g * m) / l * y(x))
    ode = sp.Eq(do_m * y(x).diff(x, x) + do_b * y(x) .diff(x) + (do_g * do_m) / do_l * y(x))
    # analytic = sp.dsolve(ode)
    analytic = sp.dsolve(ode, ics={y(0): y0[0], y(x).diff(x).subs(x, 0): y0[1]})
    return sp.latex(ode_symbol), sp.latex(analytic)


def get_damp_osc_analytic(m, b, l):
    global do_roots
    global do_m
    global do_b
    global do_l
    global formula_a
    picked = ""
    poly = np.poly1d([m, b, (m * do_g) / l])
    r = poly.r

    if not(any(np.iscomplex(r))):
        if r[0] != r[1]:
            f = get_analytical_real;
            picked = "real"
        else:
            f = get_analytical_equal
            picked = "double"
        # y[0] * exp(do_roots[0] * x) + y[1] * exp(do_roots[1] * x)
        formula_a = 'Not defined real roots'
    elif any(np.iscomplex(r)):
        picked = "complex"
        f = get_analytical_img
        formula_a = 'Not defined complex roots'
    # else:
    #     f = get_analytical_equal
    #     formula_a = 'One double root'

    do_roots = r
    do_m = m
    do_b = b
    do_l = l
    return f, picked


def get_analytical_real(x, y):
    global do_roots
    return y[0] * exp(do_roots[0] * x) + y[1] * exp(do_roots[1] * x)


def get_analytical_img(x, y):
    global do_roots
    return exp(do_roots[0].real * x) * (y[0] * cos(do_roots[0].imag * x) + y[1] * sin(do_roots[0].imag * x))


def get_analytical_equal(x, y):
    global do_roots
    return exp(-do_roots[0] * x) * (y[0] + y[1] * x)


def get_const_do():
    global do_l
    global do_m
    global do_g
    global do_b
    constants = ("l = " + str(do_l) + "\\ m " + ";\\ m = " + str(do_m) + "\\ kg;\\ g = " + str(do_g)
                 + "\\;\\frac{N}{kg} " + ";\\;b = " + str(do_b) + "\\;N\\;")
    return constants
