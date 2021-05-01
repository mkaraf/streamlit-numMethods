import numpy as np


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
        y[i + 1] = y[i] + h / 2 * f(x[i], (y[i] + f(x[i], y_pred)))
    return y


def rk_gen_k(A, B, f, y0, x, h):
    order = len(A)
    k = np.zeros(order, float)
    for i in range(0, order):
        k[i] = h * f(x + h * A[i], y0 + sum(np.multiply(k, B[i])))
    return k


def rk_calc(A, B, C, f, y0, x, h):
    steps = len(x)
    order = len(A)
    y = np.zeros(steps, float)
    k = np.zeros(order, float)
    y[0] = y0
    for i in range(0, steps - 1):
        k = rk_gen_k(A, B, f, y[i], x[i], h)
        y[i + 1] = y[i] + sum(np.multiply(k, C))
    return y


def rk2(f, y0, x, h):
    A = (0, 1 / 2)
    B = [(0, 0), (1 / 2, 0)]
    C = (0, 1)
    return rk_calc(A, B, C, f, y0, x, h)


def rk3(f, y0, x, h):
    A = (0, 1 / 2, 1)
    B = [(0, 0, 0), (1 / 2, 0, 0), (-1, 2, 0)]
    C = (1 / 6, 2 / 3, 1 / 6)
    return rk_calc(A, B, C, f, y0, x, h)


def rk4(f, y0, x, h):
    A = (0, 1 / 2, 1 / 2, 1)
    B = [(0, 0, 0, 0), (1 / 2, 0, 0, 0), (0, 1 / 2, 0, 0), (0, 0, 1, 0)]
    C = (1 / 6, 1 / 3, 1 / 3, 1 / 6)
    return rk_calc(A, B, C, f, y0, x, h)


def ab2(f, y0, x, h):
    n = len(x)
    y = np.empty(n - 2, float)

    y = np.append(rk4(f, y0, x[:2], h), y)

    for i in range(1, n - 1):
        y[i + 1] = y[i] + 1 / 2 * h * (3 * f(x[i], y[i]) - f(x[i - 1], y[i - 1]))
    return y


def ab3(f, y0, x, h):
    n = len(x)
    y = np.empty(n - 3, float)

    y = np.append(rk4(f, y0, x[:3], h), y)

    for i in range(2, n - 1):
        y[i + 1] = y[i] + 1 / 12 * h * (23 * f(x[i], y[i]) - 16 * f(x[i - 1], y[i - 1]) + 5 * f(x[i - 2], y[i - 2]))
    return y


def ab4(f, y0, x, h):
    n = len(x)
    y = np.empty(n - 4, float)

    y = np.append(rk4(f, y0, x[:4], h), y)

    for i in range(3, n - 1):
        y[i + 1] = y[i] + 1 / 24 * h * (55 * f(x[i], y[i]) - 59 * f(x[i - 1], y[i - 1]) + 37 * f(x[i - 2], y[i - 2])
                                        - 9 * f(x[i - 3], y[i - 3]))
    return y


def rkf45(f, y0, a, b, tol, h_max, h_min):
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

    # Set status
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

        if r == 0:
            # Something went wrong. Division by zero.
            e = -1
            break
        h = h * min(max(0.84 * (tol / r) ** (1 / 4), 0.1), 4.0)

        if h > h_max:
            h = h_max
        elif h < h_min:
            # Could not converge to the required tolerance with choosed minimum step size
            e = -2
            break

    return x, y, e


# def rkf(f, a, b, x0, tol, hmax, hmin):
#     """Runge-Kutta-Fehlberg method to solve x' = f(x,t) with x(t[0]) = x0.
#     USAGE:
#         t, x = rkf(f, a, b, x0, tol, hmax, hmin)
#     INPUT:
#         f     - function equal to dx/dt = f(x,t)
#         a     - left-hand endpoint of interval (initial condition is here)
#         b     - right-hand endpoint of interval
#         x0    - initial x value: x0 = x(a)
#         tol   - maximum value of local truncation error estimate
#         hmax  - maximum step size
#         hmin  - minimum step size
#     OUTPUT:
#         t     - NumPy array of independent variable values
#         x     - NumPy array of corresponding solution function values
#     NOTES:
#         This function implements 4th-5th order Runge-Kutta-Fehlberg Method
#         to solve the initial value problem
#            dx
#            -- = f(x,t),     x(a) = x0
#            dt
#         on the interval [a,b].
#         Based on pseudocode presented in "Numerical Analysis", 6th Edition,
#         by Burden and Faires, Brooks-Cole, 1997.
#     """
#
#     # Coefficients used to compute the independent variable argument of f
#
#     a2 = 2.500000000000000e-01  # 1/4
#     a3 = 3.750000000000000e-01  # 3/8
#     a4 = 9.230769230769231e-01  # 12/13
#     a5 = 1.000000000000000e+00  # 1
#     a6 = 5.000000000000000e-01  # 1/2
#
#     # Coefficients used to compute the dependent variable argument of f
#
#     b21 = 2.500000000000000e-01  # 1/4
#     b31 = 9.375000000000000e-02  # 3/32
#     b32 = 2.812500000000000e-01  # 9/32
#     b41 = 8.793809740555303e-01  # 1932/2197
#     b42 = -3.277196176604461e+00  # -7200/2197
#     b43 = 3.320892125625853e+00  # 7296/2197
#     b51 = 2.032407407407407e+00  # 439/216
#     b52 = -8.000000000000000e+00  # -8
#     b53 = 7.173489278752436e+00  # 3680/513
#     b54 = -2.058966861598441e-01  # -845/4104
#     b61 = -2.962962962962963e-01  # -8/27
#     b62 = 2.000000000000000e+00  # 2
#     b63 = -1.381676413255361e+00  # -3544/2565
#     b64 = 4.529727095516569e-01  # 1859/4104
#     b65 = -2.750000000000000e-01  # -11/40
#
#     # Coefficients used to compute local truncation error estimate.  These
#     # come from subtracting a 4th order RK estimate from a 5th order RK
#     # estimate.
#
#     r1 = 2.777777777777778e-03  # 1/360
#     r3 = -2.994152046783626e-02  # -128/4275
#     r4 = -2.919989367357789e-02  # -2197/75240
#     r5 = 2.000000000000000e-02  # 1/50
#     r6 = 3.636363636363636e-02  # 2/55
#
#     # Coefficients used to compute 4th order RK estimate
#
#     c1 = 1.157407407407407e-01  # 25/216
#     c3 = 5.489278752436647e-01  # 1408/2565
#     c4 = 5.353313840155945e-01  # 2197/4104
#     c5 = -2.000000000000000e-01  # -1/5
#
#     # Set t and x according to initial condition and assume that h starts
#     # with a value that is as large as possible.
#
#     t = a
#     x = np.array(x0)
#     h = hmax
#
#     # Initialize arrays that will be returned
#
#     T = np.array([t])
#     X = np.array([x])
#
#     while t < b:
#
#         # Adjust step size when we get to last interval
#
#         if t + h > b:
#             h = b - t;
#
#         # Compute values needed to compute truncation error estimate and
#         # the 4th order RK estimate.
#
#         k1 = h * f(t, x)
#         k2 = h * f(t + a2 * h, x + b21 * k1)
#         k3 = h * f(t + a3 * h, x + b31 * k1 + b32 * k2)
#         k4 = h * f(t + a4 * h, x + b41 * k1 + b42 * k2 + b43 * k3)
#         k5 = h * f(t + a5 * h, x + b51 * k1 + b52 * k2 + b53 * k3 + b54 * k4)
#         k6 = h * f(t + a6 * h, x + b61 * k1 + b62 * k2 + b63 * k3 + b64 * k4 + b65 * k5)
#
#         # Compute the estimate of the local truncation error.  If it's small
#         # enough then we accept this step and save the 4th order estimate.
#
#         r = abs(r1 * k1 + r3 * k3 + r4 * k4 + r5 * k5 + r6 * k6) / h
#         if len(np.shape(r)) > 0:
#             r = max(r)
#         if r <= tol:
#             t = t + h
#             x = x + c1 * k1 + c3 * k3 + c4 * k4 + c5 * k5
#             T = np.append(T, t)
#             X = np.append(X, [x], 0)
#
#         # Now compute next step size, and make sure that it is not too big or
#         # too small.
#         if r == 0:
#             break
#         h = h * min(max(0.84 * (tol / r) ** 0.25, 0.1), 4.0)
#
#         if h > hmax:
#             h = hmax
#         elif h < hmin:
#             raise RuntimeError(
#                 "Error: Could not converge to the required tolerance %e with minimum stepsize  %e." % (tol, hmin))
#             break
#
#     # endwhile
#
#     return (T, X)

def abm4_pc(f, y0, x, h):
    n = len(x)
    y = np.empty(n - 4, float)

    y = np.append(rk4(f, y0, x[:4], h), y)

    for i in range(3, n - 1):
        y_p = y[i] + 1 / 24 * h * (55 * f(x[i], y[i]) - 59 * f(x[i - 1], y[i - 1]) + 37 * f(x[i - 2], y[i - 2])
                                   - 9 * f(x[i - 3], y[i - 3]))

        y[i + 1] = y[i] + 1 / 24 * h * (9 * y_p + 19 * f(x[i], y[i]) - 5 * f(x[i - 1], y[i - 1])
                                        + f(x[i - 2], y[i - 2]))

    return y
