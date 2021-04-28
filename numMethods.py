import numpy as np
#import streamlit as st

def Euler(f,y0,x,h):
    n = len(x)
    y = np.empty(n, float)
    y[0] = y0
    for i in range(0, n-1):
        y[i+1] = y[i] + h * f(x[i], y[i])
    return y

def Heun(f,y0,x,h):
    n = len(x)
    y = np.empty(n, float)
    y[0] = y0
    for i in range(0, n-1):
        yPred = y[i] + h * f(x[i], y[i])
        y[i+1] = y[i] + h/2 * f(x[i], (y[i] + f(x[i], yPred)))
    return y

def Rk4_Replaced(f,y0,x,h):
    n = len(x)
    y = np.empty(n, float)
    y[0] = y0
    for i in range(0, n-1):
        K1 = h * f(x[i], y[i])
        K2 = h * f(x[i] + h / 2, y[i] + K1 / 2)
        K3 = h * f(x[i] + h / 2, y[i] + K2 / 2)
        K4 = h * f(x[i] + h, y[i] + K3)
        y[i+1] = y[i] + (K1 + 2 * K2 + 2 * K3 + K4) / 6
    return y

def rkGen(A,B,C,f,y0,x,h):
    order = len(A)
    K = np.zeros(order, float)
    for i in range(0, order):
        K[i] = h * f(x[i] + A[i], y0 + sum(np.multiply(K,B[i])))
    y = y0 + sum(np.multiply(K,C))
    return y

def rkCalc(A,B,C,f,y0,x,h):
    steps = len(x)
    y = np.zeros(steps, float)
    y[0] = y0
    for i in range(0, steps-1):
        y[i+1] = rkGen(A,B,C,f,y[i],x,h)
    return y

def Rk2(f,y0,x,h):
#Explicit midpoint method
    A = (0, 1/2)
    B = [(0, 0), (1/2, 0)]
    C = (0, 1)
    return rkCalc(A,B,C,f,y0,x,h)

def Rk3(f,y0,x,h):
    A = (0, 1/2, 1)
    B = [(0, 0, 0), (1/2, 0, 0), (-1, 2, 0)]
    C = (1/6, 2/3, 1/6)
    return rkCalc(A, B, C, f, y0, x, h)

def Rk4(f,y0,x,h):
    A = (0,1/2,1/2,1)
    B = [(0, 0, 0, 0), (1/2, 0, 0, 0), (0, 1/2, 0, 0), (0, 0, 1, 0)]
    C = (1 / 6, 1 / 3, 1 / 3, 1 / 6)
    return rkCalc(A,B,C,f,y0,x,h)