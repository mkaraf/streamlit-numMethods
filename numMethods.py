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
    ode_system = len(y0)  # if more then one init condition -> system of ODE's
    n = len(x)  # number of required steps

    # init variables for calc
    y = [[0] * ode_system for i in range(n)]
    y[0] = y0

    for i in range(0, n - 1):
        y[i + 1][:] = np.add(y[i], np.multiply(f(x[i], y[i]), h))
    return y


def heun(f, y0, x, h):
    ode_system = len(y0)  # if more then one init condition -> system of ODE's
    n = len(x)  # number of required steps

    # init variables for calc
    y = [[0] * ode_system for i in range(n)]
    y[0] = y0

    for i in range(0, n - 1):
        y_pred = np.add(y[i], np.multiply(f(x[i], y[i]), h))
        y[i + 1][:] = np.add(y[i], np.multiply(np.add(f(x[i], y[i]), f(x[i+1], y_pred)), h/2))
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

            if 1 == ode_system:
                y_curr = [y_curr]

            k[o][:] = np.multiply(f(x_curr, y_curr), h)

        y[i + 1][:] = np.add(y[i], sum(np.multiply([*zip(*k)], C).T))
    return y


#   MULTI-STEP METHODS     #


def abx(f, y0, x, h, order):
    ode_system = len(y0)  # if more then one init condition -> system of ODE's
    n = len(x)  # number of required steps

    # init variables for calc
    tmp = np.empty(ode_system, float)

    y_m = [[0] * ode_system for i in range(n - order)]
    y_s = rkx(f, y0, x[:order], h, 4)
    y = np.vstack((y_s, y_m))

    A, B = get_tableau_abx(order)

    for i in range(order - 1, n - 1):
        for j in range(0, order):
            tmp = np.add(tmp, np.multiply(f(x[i - j], y[i - j]), B[j]))

        y[i + 1] = np.add(y[i], np.multiply(np.multiply(A, h), tmp))
        tmp = 0
    return y


#   PREDICTOR-CORRECTOR METHODS  #


def abm4_pc(f, y0, x, h):
    ode_system = len(y0)  # if more then one init condition -> system of ODE's
    n = len(x)  # number of required steps
    order = 4

    # init variables for calc
    tmp = np.empty(ode_system, float)

    y_m = [[0] * ode_system for i in range(n - order)]
    y_s = rkx(f, y0, x[:order], h, 4)
    y = np.vstack((y_s, y_m))

    tmp_pred = 0
    tmp_kor = 0

    A, B_PRED, B_COR = get_tableau_abc4()

    for i in range(order - 1, n - 1):
        for j in range(0, order):
            tmp_pred = np.add(tmp_pred, np.multiply(f(x[i - j], y[i - j]), B_PRED[j]))
        y[i + 1][:] = np.add(y[i], np.multiply(np.multiply(A, h), tmp_pred))

        for k in range(0, order):
            tmp_kor = np.add(tmp_kor, np.multiply(f(x[i - k + 1], y[i - k + 1]), B_COR[k]))
        y[i + 1] = np.add(y[i], np.multiply(np.multiply(A, h), tmp_kor))

        tmp_pred = 0
        tmp_kor = 0
    return y


#   ADAPTIVE-STEP METHODS     #


def rkf45(f, y0, a, b, tol, h_max, h_min):
    A, B, C, CT = get_tableau_rkf45()
    order = 6

    # Initialize return status
    e = 0

    ode_system = len(y0)  # if more then one init condition -> system of ODE's

    # init variables for calc
    var = np.empty(ode_system, float)

    # Set x and y according to initial condition and assume that h starts
    # with a value that is as large as possible.
    x_cur = a
    y_cur = np.array(y0)
    h = h_max

    # Initialize arrays that will be returned
    x = np.array([x_cur])
    y = np.array([y_cur])
    k = [[0] * ode_system for i in range(order)]

    while x_cur < b:
        # Adjust step size when we get to last interval
        if x_cur + h > b:
            h = b - x_cur

        for o in range(0, order):
            var = np.multiply([*zip(*k)], B[o])
            mul = sum(np.array([*zip(*var)]))

            if 1 == ode_system:
                k[o][:] = np.multiply(f(x_cur + h * A[o], np.add(mul, [y_cur])), h)
            else:
                k[o][:] = np.multiply(f(x_cur + h * A[o], np.add(mul, y_cur)), h)

        # Compute the estimate of the local truncation error.
        # If it's small enough then we accept this step and save the 4th order estimate.
        r = max(abs(sum(np.multiply([*zip(*k)], CT).T)) / h)
        if 0 == r:
            e = -1
            break

        if r <= tol:
            x_cur = x_cur + h
            y_cur = np.add(y_cur, sum(np.multiply([*zip(*k[:5])], C).T))

            x = np.append(x, x_cur)
            y = np.vstack((y, y_cur))

        h = h * min(max(0.84 * (tol / r) ** (1 / 4), 0.1), 4.0)

        if h > h_max:
            h = h_max
        elif h < h_min:
            # Could not converge to the required tolerance with chose minimum step size
            e = -1
            break

    return x, y, e
