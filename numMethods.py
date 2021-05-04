import numpy as np


#   TABLEAU'S FOR RK FAMILY METHODS    #

def get_tableau_rkx(x):
    if 2 == x:
        A = (0, 1 / 2)
        B = [(0, 0), (1 / 2, 0)]
        C = (0, 1)
    elif 3 == x:
        A = (0, 1 / 2, 1)
        B = [(0, 0, 0), (1 / 2, 0, 0), (-1, 2, 0)]
        C = (1 / 6, 2 / 3, 1 / 6)
    elif 4 == x:
        A = (0, 1 / 2, 1 / 2, 1)
        B = [(0, 0, 0, 0), (1 / 2, 0, 0, 0), (0, 1 / 2, 0, 0), (0, 0, 1, 0)]
        C = (1 / 6, 1 / 3, 1 / 3, 1 / 6)
    return A, B, C


def get_tableau_rkf45():
    # Coefficients used to compute the independent variable argument of f
    A = (0, 1 / 4, 3 / 8, 12 / 13, 1, 1 / 2)
    # Coefficients used to compute the dependent variable argument of f
    B = [(0, 0, 0, 0, 0, 0),
         (1 / 4, 0, 0, 0, 0, 0),
         (3 / 32, 9 / 32, 0, 0, 0, 0),
         (1932 / 2197, -7200 / 2197, 7296 / 2197, 0, 0, 0),
         (439 / 216, -8, 3680 / 513, -845 / 4104, 0, 0),
         (-8 / 27, 2, -3544 / 2565, 1859 / 4104, -11 / 40, 0)]
    # Coefficients used to compute 4th order RK estimate
    C = (25 / 216, 0, 1408 / 2565, 2197 / 4104, -1 / 5)
    # Coefficients used to compute local truncation error estimate.  These
    # come from subtracting a 4th order RK estimate from a 5th order RK
    # estimate.
    CT = (1 / 360, 0, -128 / 4275, -2197 / 75240, 1 / 50, 2 / 55)
    return A, B, C, CT


#   TABLEAU'S FOR ADAM'S FAMILY METHODS    #


def get_tableau_abx(x):
    if 2 == x:
        A = 1 / 2
        B = [3, -1]
    elif 3 == x:
        A = 1 / 12
        B = [23, -16, 5]
    elif 4 == x:
        A = 1 / 24
        B = [55, -59, 37, -9]
    return A, B


def get_tableau_abc4():
    A = 1 / 24
    B_PRED = [55, -59, 37, -9]
    B_COR = [9, 19, -5, 1]
    return A, B_PRED, B_COR


#   SINGLE-STEP METHODS     #


def euler(f, y0, x, h):
    n = len(x)
    y = np.empty(n, float)
    y[0] = y0
    for i in range(0, n - 1):
        y[i + 1] = y[i] + h * f(x[i], y[i])
    return y


def heun(f, y0, x, h):
    n = len(x)
    y = np.empty(n, float)
    y[0] = y0
    for i in range(0, n - 1):
        y_pred = y[i] + h * f(x[i], y[i])
        y[i + 1] = y[i] + h / 2 * (f(x[i], y[i]) + f(x[i], y_pred))
    return y


def rkx(f, y0, x, h, order):
    ode_system = len(y0)  # if more then one init condition -> system of ODE's
    n = len(x)  # number of required steps

    # init variables for calc
    y = [[0] * ode_system for i in range(n)]
    k = [[0] * ode_system for i in range(order)]
    var = np.empty(ode_system, float)
    y[0] = y0

    # get corresponding Butcher's tableau according required order
    A, B, C = get_tableau_rkx(order)

    # start calculation
    for i in range(0, n - 1):
        for o in range(0, order):
            var = np.multiply([*zip(*k)], B[o])
            mul = sum(np.array([*zip(*var)]))

            y_curr = np.add(mul, y[i])
            x_curr = x[i] + h * A[o]

            k[o][:] = np.multiply(f(x_curr, y_curr), h)

        y[i + 1][:] = np.add(y[i], sum(np.multiply([*zip(*k)], C).T))

    return y


#   MULTI-STEP METHODS     #


def abx(f, y0, x, h, order):
    n = len(x)

    y = np.empty(n - order, float)
    tmp = 0

    y = np.append(rkx(f, [y0], x[:order], h, 4), y)

    A, B = get_tableau_abx(order)

    for i in range(order - 1, n - 1):
        for j in range(0, order):
            tmp += f(x[i - j], y[i - j]) * B[j]
        y[i + 1] = y[i] + A * h * tmp
        tmp = 0
    return y


#   PREDICTOR-CORRECTOR METHODS  #


def abm4_pc(f, y0, x, h):
    n = len(x)
    order = 4

    y = np.empty(n - order, float)

    y = np.append(rkx(f, [y0], x[:order], h, order), y)

    tmp_pred = 0
    tmp_kor = 0

    A, B_PRED, B_COR = get_tableau_abc4()

    for i in range(order - 1, n - 1):
        for j in range(0, order):
            tmp_pred += f(x[i - j], y[i - j]) * B_PRED[j]
        y[i + 1] = y[i] + A * h * tmp_pred

        for k in range(0, order):
            tmp_kor += f(x[i - k + 1], y[i - k + 1]) * B_COR[k]
        y[i + 1] = y[i] + A * h * tmp_kor

        tmp_pred = 0
        tmp_kor = 0
    return y


#   ADAPTIVE-STEP METHODS     #


def rk_gen_k(A, B, f, y0, x, h):
    order = len(A)
    k = np.zeros(order, float)
    for i in range(0, order):
        k[i] = h * f(x + h * A[i], y0 + sum(np.multiply(k, B[i])))
    return k


def rkf45(f, y0, a, b, tol, h_max, h_min):
    A, B, C, CT = get_tableau_rkf45()

    # Initialize return status
    e = 0

    # Set x and y according to initial condition and assume that h starts
    # with a value that is as large as possible.
    x_cur = a
    y_cur = np.array(y0)
    h = h_max

    # Initialize arrays that will be returned
    x = np.array([x_cur])
    y = np.array([y_cur])

    while x_cur < b:
        # Adjust step size when we get to last interval
        if x_cur + h > b:
            h = b - x_cur

        # Compute values needed to compute truncation error estimate and
        # the 4th order RK estimate.
        k = rk_gen_k(A, B, f, y_cur, x_cur, h)

        # Compute the estimate of the local truncation error.
        # If it's small enough then we accept this step and save the 4th order estimate.
        r = abs(sum(np.multiply(k, CT))) / h
        if r <= tol:
            x_cur = x_cur + h
            y_cur = y_cur + sum(np.multiply(k[:5], C))
            x = np.append(x, x_cur)
            y = np.append(y, [y_cur], 0)

        h = h * min(max(0.84 * (tol / r) ** (1 / 4), 0.1), 4.0)

        if h > h_max:
            h = h_max
        elif h < h_min:
            # Could not converge to the required tolerance with chose minimum step size
            e = -1
            break

    return x, y, e


#   OLD GARBAGE     #


# def rk_gen_k(A, B, f, y0, x, h):
#     order = len(A)
#     k = np.zeros(order, float)
#     for i in range(0, order):
#         k[i] = h * f(x + h * A[i], y0 + sum(np.multiply(k, B[i])))
#     return k
#
#
# def rk_calc(A, B, C, f, y0, x, h):
#     steps = len(x)
#     order = len(A)
#     y = np.zeros(steps, float)
#     k = np.zeros(order, float)
#     y[0] = y0
#     for i in range(0, steps - 1):
#         k = rk_gen_k(A, B, f, y[i], x[i], h)
#         y[i + 1] = y[i] + sum(np.multiply(k, C))
#     return y
#
#
# def rk2(f, y0, x, h):
#     A = (0, 1 / 2)
#     B = [(0, 0), (1 / 2, 0)]
#     C = (0, 1)
#     return rk_calc(A, B, C, f, y0, x, h)
#
#
# def rk3(f, y0, x, h):
#     A = (0, 1 / 2, 1)
#     B = [(0, 0, 0), (1 / 2, 0, 0), (-1, 2, 0)]
#     C = (1 / 6, 2 / 3, 1 / 6)
#     return rk_calc(A, B, C, f, y0, x, h)
#
#
# def rk4(f, y0, x, h):
#     A = (0, 1 / 2, 1 / 2, 1)
#     B = [(0, 0, 0, 0), (1 / 2, 0, 0, 0), (0, 1 / 2, 0, 0), (0, 0, 1, 0)]
#     C = (1 / 6, 1 / 3, 1 / 3, 1 / 6)
#     return rk_calc(A, B, C, f, y0, x, h)


# def ab2(f, y0, x, h):
#     n = len(x)
#     y = np.empty(n - 2, float)
#
#     y = np.append(rk4(f, y0, x[:2], h), y)
#
#     for i in range(1, n - 1):
#         y[i + 1] = y[i] + 1 / 2 * h * (3 * f(x[i], y[i]) - f(x[i - 1], y[i - 1]))
#     return y
#
#
# def ab3(f, y0, x, h):
#     n = len(x)
#     y = np.empty(n - 3, float)
#
#     y = np.append(rk4(f, y0, x[:3], h), y)
#
#     for i in range(2, n - 1):
#         y[i + 1] = y[i] + 1 / 12 * h * (23 * f(x[i], y[i]) - 16 * f(x[i - 1], y[i - 1]) + 5 * f(x[i - 2], y[i - 2]))
#     return y
#
#
# def ab4(f, y0, x, h):
#     n = len(x)
#     y = np.empty(n - 4, float)
#
#     y = np.append(rk4(f, y0, x[:4], h), y)
#
#     for i in range(3, n - 1):
#         y[i + 1] = y[i] + 1 / 24 * h * (55 * f(x[i], y[i]) - 59 * f(x[i - 1], y[i - 1]) + 37 * f(x[i - 2], y[i - 2])
#                                         - 9 * f(x[i - 3], y[i - 3]))
#     return y


# def abm4_pc(f, y0, x, h):
#     n = len(x)
#     y = np.empty(n - 4, float)
#
#     y = np.append(rk4(f, y0, x[:4], h), y)
#
#     for i in range(3, n - 1):
#         y_p = y[i] + 1 / 24 * h * (55 * f(x[i], y[i]) - 59 * f(x[i - 1], y[i - 1]) + 37 * f(x[i - 2], y[i - 2])
#                                    - 9 * f(x[i - 3], y[i - 3]))
#
#         y[i + 1] = y[i] + 1 / 24 * h * (9 * f(x[i+1], y_p) + 19 * f(x[i], y[i]) - 5 * f(x[i - 1], y[i - 1])
#                                         + f(x[i - 2], y[i - 2]))
#
#     return y
