from math import sin, cos, exp
import numpy as np

#   purpose of this module is provide defined models of the systems of Ordinary Differential Equations

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

formula_a = ""


def get_model_str(id):
    model = ""
    if id == 0:
        model = "y'' = -\\frac{g}{l}\\sin(y)"
    if id == 1:
        model = "y'' = - by' -\\frac{mg}{l}y = 0"
    return model


def get_system_str():
    model = [""] * 2
    model[0] = "Mathematical pendulum"
    model[1] = "Damped oscillations"
    return model


# FUNCTIONS FOR MATHEMATICAL PENDULUM
def model_math_pendulum(x, y):
    global mp_g
    global mp_l
    return y[1], -(mp_g / mp_l) * sin(y[0])


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
# my'' + do_b y' + 49'y = 0
def model_do(x, y):
    global do_l
    global do_m
    global do_g
    global do_b
    return y[1], (-y[1] * do_b - (do_m * do_g)/do_l * y[0]) / do_m


def get_damp_osc_analytic(m, b, l):
    global do_roots
    global do_m
    global do_b
    global do_l
    global formula_a

    poly = np.poly1d([m, b, (m * do_g) / l])
    r = poly.r

    if not(any(np.iscomplex(r))):
        f = get_analytical_real
        s = ''
    elif any(np.iscomplex(r)):
        f = get_analytical_img
        s = 'img'
    else:
        f = get_analytical_equal
        s = 'equal'

    do_roots = r
    do_m = m
    do_b = b
    do_l = l
    return f, s


def get_analytical_real(x, y):
    global do_roots
    return y[0] * exp(do_roots[0] * x) + y[1] * exp(do_roots[1] * x)


def get_analytical_img(x, y):
    global do_roots
    return exp(do_roots[0].real * x) * (y[0] * cos(do_roots[0].imag * x) + y[1] * sin(do_roots[0].imag * x))


def get_analytical_equal(x, y):
    global do_roots
    return exp(-do_roots[0] * x) * (y[0] + y[1] * x)


def get_do_constants():
    global do_l
    global do_m
    global do_g
    global do_b
    constants = ("l = " + str(do_l) + "\\;\\m " + ",\\;m = " + str(do_m) + "\\;kg\\;g = " + str(do_g)
                 + "\\;\\frac{N}{kg} " + ",\\;b = " + str(do_b) + "\\;N\\;")
    return


# def get_


# def set_var_throw_motion(l):
#     global mp_l
#     mp_l = l
#     return

# def f0(x, y):
#     return y[0]
#
#
# def f1(x, y):
#     return 20 * x - 20 * y[0] + 21
#
#
# def f2(x, y):
#     return 1 + x**2 + 0 * y[0]
#
#
# def f3(x, y):
#     return y[0] - x**2 + 1
#
#
# # y'' + y = 4x + 10*sin(x)
# # y' = y'
# # y'' = 4x + 10sin(x) - y
# def f4(x, y):
#     return y[1], 4 * x + 10 * sin(x) - y[0]
#
#
# # y''' + 4y'' + 2y' - y = 0
# # y' = y'
# # y'' = y''
# # y''' = -4y'' -2y' +y
# def f5(x, y):
#     return y[2] + 0 * x, y[1], - 4 * y[2] - 2 * y[1] - y[0] + 0 * x



# ############################          CHANGED

# C = tmp = np.empty(4, float)
#
# f1 = lambda x, y: y[0]
# s1 = "y' - y = 0"
# f2 = lambda x, y: -y[0]
# s2 = "y' + y = 0"
# f3 = lambda x, y: -x * y[0]**2
# s3 = "y' + x * y^2 = 0"
# f4 = lambda x, y: x ** 2 - y[0]
# s4 = "y' + y = x^2"
# # f5 = lambda x, y: -20*y + 20*x + 21
# def f5(x, y):
#     return -20*y[0] + 20*x + 21
# s5 = "y' + 20y -21 = -20x"
# # f6 = lambda x, y: 1 + x**2
# def f6(x, y):
#     return (y[0] + x**2)
# s6 = "y' - 1 = x^2"
# f7 = lambda x, y: x*y[0]
# s7 = "y' xy = 0"
# f8 = lambda x, y: y[0] - x**2 + 1
# s8 = "y' - y = -x^2 +1"
# f9 = lambda x, y: y[0] + x
# s9 = "y' - y = x"
#
#
# # y'' + y = 4x + 10*sin(x)
# # y' = y'
# # y'' = 4x + 10sin(x) - y
# s10 = "y'' + y = 4x + 10*sin(x)"
# def f10(x, y):
#     return (y[1], 4 * x + 10 * sin(x) - y[0])
#
# # y''' + 4y'' + 2y' - y = 0
# # y' = y'
# # y'' = y''
# # y''' = -4y'' -2y' +y
# s11 = "y''' = -4y'' -2y' +y"
# def f11(x, y):
#     return (y[2], y[1], - 4 * y[2] - 2 * y[1] - y[0])

