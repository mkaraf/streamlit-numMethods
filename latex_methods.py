#   purpose of this module is only provide getter functions of strings with general
#   description of specific numerical method in latex format

def get_euler_latex():
    lat_euler = ['y_{(i+1)} = y_{(i)}  + h f(x_{(i)}, y_{(i)})']
    return lat_euler


def get_heun_latex():
    lat_heun = [
        'y_{pred} = y_{(i)}  + h f(x_{(i)}, y_{(i)})',
        'y_{(i+1)} = y_{(i)} + \\frac {1}{2} h (f(x_{(i)}, y_{(i)}) + f(x_{(i+1)}, y_{pred}))'
    ]
    return lat_heun


def get_rk2_latex():
    lat_rk2 = [
        'C = \\begin{pmatrix} 0 \\\\\\ 1  \\end{pmatrix}'
        '\\quad'
        '\\ A = \\begin{pmatrix} 0 & 0 \\\\ \\ 1/2 & 0 \\end{pmatrix}'
        '\\quad'
        '\\ B = \\begin{pmatrix} 0 & 1/2  \\end{pmatrix}',
        'k_1 = f(x_{(i)}, y_{(i)})',
        'k_2 = f(x_{(i)} + h C_2, y_{(i)} + h(A_{21}k_1))',
        'y_{(i+1)} = y_{(i)}  + h \\displaystyle\\sum_{j=1}^2 B_jk_j'
    ]
    return lat_rk2


def get_rk3_latex():
    lat_rk3 = [
        'C = \\begin{pmatrix} 0 \\\\\\ 1/2 \\\\\\ 1 \\end{pmatrix}'
        '\\quad'
        '\\ A = \\begin{pmatrix} 0 & 0 & 0 \\\\\\ 1/2 & 0 & 0 \\\\\\ -1 & 2 & 0 \\end{pmatrix}'
        '\\quad'
        '\\ B = \\begin{pmatrix} 1/6 & 2/3 & 1/6  \\end{pmatrix}',
        'k_1 = f(x_{(i)}, y_{(i)})',
        'k_2 = f(x_{(i)} + h C_2, y_{(i)} + h(A_{21}k_1))',
        'k_3 = f(x_{(i)} + h C_3, y_{(i)} + h(A_{31}k_1 + A_{32}k_2))',
        'y_{(i+1)} = y_{(i)}  + h \\displaystyle\\sum_{j=1}^3 B_jk_j'
    ]
    return lat_rk3


def get_rk4_latex():
    lat_rk4 = [
        'C = \\begin{pmatrix} 1/6 \\\\\\ 1/3 \\\\\\ 1/3 \\\\\\ 1/6 \\end{pmatrix}'
        '\\quad'
        '\\ A = \\begin{pmatrix} 0 & 0 & 0 & 0 \\\\\\ 1/2 & 0 & 0 & 0 \\\\\\ 0 & 1/2 & 0 & 0 '
        '\\\\\\ 0 & 0 & 1 & 0 \\end{pmatrix}'
        '\\quad'
        '\\ B = \\begin{pmatrix} 0 & 1/2 & 1/2 & 1 \\end{pmatrix}',
        'k_1 = f(x_{(i)}, y_{(i)})',
        'k_2 = f(x_{(i)} + h C_2, y_{(i)} + h(A_{21}k_1))',
        'k_3 = f(x_{(i)} + h C_3, y_{(i)} + h(A_{31}k_1) + h(A_{32}k_2))',
        'k_4 = f(x_{(i)} + h C_4, y_{(i)} + h(A_{41}k_1 + A_{42}k_2 + A_{43}k_3))',
        'y_{(i+1)} = y_{(i)}  + h \\displaystyle\\sum_{j=1}^4 B_jk_j'
    ]
    return lat_rk4


def get_ab2_latex():
    lat_ab2 = [
        'y_{(i+1)} = y_{(i)} + \\frac{1}{2} h (3f(x_{(i)},y_{(i)})-f(x_{(i-1)},y_{(i-1)}))'
    ]
    return lat_ab2


def get_ab3_latex():
    lat_ab2 = [
        'y_{(i+1)} = y_{(i)} + \\frac{1}{12} h (23f(x_{(i)},y_{(i)})-16f(x_{(i-1)},y_{(i-1)})+5f(x_{(i-2)},y_{(i-2)}))'
    ]
    return lat_ab2


def get_ab4_latex():
    lat_ab2 = [
        'y_{(i+1)} = y_{(i)} + \\frac{1}{24} h (55f(x_{(i)},y_{(i)}) '
        '- 59f(x_{(i-1)},y_{(i-1)})+37f(x_{(i-2)},y_{(i-2)}) - 9f(x_{(i-3)},y_{(i-3)}))'
    ]
    return lat_ab2


def get_abm4_latex():
    lat_abm4 = [
        'y_{pred} = y_{(i)} + \\frac{1}{24} h (55f(x_{(i)},y_{(i)}) - 59f(x_{(i-1)},y_{(i-1)})'
        '+37f(x_{(i-2)},y_{(i-2)}) - 9f(x_{(i-3)},y_{(i-3)}))',
        'y_{(i+1)} = y_{(i)} + \\frac{1}{24} h (9f(x_{(i+1)},y_{pred})'
        '+19f(x_{(i)},y_{(i)})-5f(x_{(i-1)},y_{(i-1)}) + f(x_{(i-2)},y_{(i-2)}))'
    ]
    return lat_abm4


def get_rkf45_latex():
    lat_rkf45 = [
        'C = \\begin{pmatrix} 0 \\\\ 1/4 \\\\ 3/8 \\\\ 12/13 \\\\ 1 \\\\ 1/2 \\end{pmatrix}'
        '\\quad'
        '\\ A = \\begin{pmatrix}'
        '\\ 0 & 0 & 0 & 0 & 0 & 0 \\\\'
        '\\ 1/4 & 0 & 0 & 0 & 0 & 0 \\\\'
        '\\ 3/32 & 9/32 & 0 & 0 & 0 & 0 \\\\'
        '\\ 1932/2197 & -7200/2197 & 7296/2197 & 0 & 0 & 0 \\\\'
        '\\ 439/216 & -8 & 3680/513 & -845/4104 & 0 & 0 \\\\'
        '\\ -8/27 & 2 & -3544/2565 & 1859/4104 & -11/40 & 0'
        '\\end{pmatrix}',
        'B^* = \\begin{pmatrix} 25/216 & 0 & 1408/2565 & 2197/4104 & -1/5 & 0 \\end{pmatrix}',
        'B = \\begin{pmatrix} 16/135 & 0 & 6656/12825 & 28561/56430 & -9/50 & 2/55\\end{pmatrix}',
        'k_1 = f(x_{(i)}, y_{(i)})',
        'k_2 = f(x_{(i)} + h C_2, y_{(i)} + h(A_{21}k_1))',
        'k_3 = f(x_{(i)} + h C_3, y_{(i)} + h(A_{31}k_1 + A_{32}k_2))',
        'k_4 = f(x_{(i)} + h C_4, y_{(i)} + h(A_{41}k_1 + A_{42}k_2 + A_{43}k_3))',
        'k_5 = f(x_{(i)} + h C_5, y_{(i)} + h(A_{51}k_1 + A_{52}k_2 + A_{53}k_3 + A_{54}k_4))',
        'k_6 = f(x_{(i)} + h C_6, y_{(i)} + h(A_{61}k_1 + A_{62}k_2 + A_{63}k_3 + A_{64}k_4 + A_{65}k_5))',
        'R = \\frac{1}{h} \\lvert 1/360k_1 - 128/4275k_3 - 2167/752460k_4 + 1/50k_5 + 2/55k_6 \\rvert',
        'if\\;R\\le tolerance,\\;then\\;approximation\\;is\\;accepted:',
        'y_{(i+1)}^{(4)} = y_{(i)}^{(4)}  + h \\displaystyle\\sum_{j=1}^5 B_j^*k_j',
        'x_{i+1} = x_{i} + h',
        'Adjustation\\;of\\;the\\;step',
        'h_{adj} = 0,84 \\lparen \\frac{\\varepsilon*h}{R}\\rparen ^{0,25}',
        'if\\;h_{adj}>h_{max}\\;then:',
        '\\;h=h_{max},'
        'else\\; if\\; h_{adj}>h_{min}\\; then:',
        '\\;h=h_{adj}',
        'else\\; stop\\;the\\;algorithm\\;because',
        'converge\\;to\\;the\\;required\\;tolerance\\;with\\;chosen\\;minimum\\;step\\;size\\;is\\;not\\;possible'
    ]
    return lat_rkf45
