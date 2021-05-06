from math import sin


def get_model_str():
    str_f = [""] * 6
    str_f[0] = "y' = y"
    str_f[1] = "y' = 20x -20y +21"
    str_f[2] = "y' = x^2 + 1"
    str_f[3] = "y' = y - x^2 + 1"
    str_f[4] = "y'' = -y + 4x + 10*sin(x)"
    str_f[5] = "y'''  = -4y'' + 2y' - y = 0"
    return str_f


def f0(x, y):
    return y[0]


def f1(x, y):
    return 20 * x - 20 * y[0] + 21


def f2(x, y):
    return 1 + x**2 + 0 * y[0]


def f3(x, y):
    return y[0] - x**2 + 1


# y'' + y = 4x + 10*sin(x)
# y' = y'
# y'' = 4x + 10sin(x) - y
def f4(x, y):
    return y[1], 4 * x + 10 * sin(x) - y[0]


# y''' + 4y'' + 2y' - y = 0
# y' = y'
# y'' = y''
# y''' = -4y'' -2y' +y
def f5(x, y):
    return y[2] + 0 * x, y[1], - 4 * y[2] - 2 * y[1] - y[0] + 0 * x


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
