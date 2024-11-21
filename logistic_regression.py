import math

def grad_a(a, b, c, d, e, f, g, h, x, y):
    z = a * x**7 + b * x**6 + c * x**5 + d * x**4 + e * x**3 + f * x**2 + g * x + h
    sigmoid_z = sigmoid(z)
    return -(y - sigmoid_z) * sigmoid_z * (1 - sigmoid_z) * x**7

def grad_b(a, b, c, d, e, f, g, h, x, y):
    z = a * x**7 + b * x**6 + c * x**5 + d * x**4 + e * x**3 + f * x**2 + g * x + h
    sigmoid_z = sigmoid(z)
    return -(y - sigmoid_z) * sigmoid_z * (1 - sigmoid_z) * x**6

def grad_c(a, b, c, d, e, f, g, h, x, y):
    z = a * x**7 + b * x**6 + c * x**5 + d * x**4 + e * x**3 + f * x**2 + g * x + h
    sigmoid_z = sigmoid(z)
    return -(y - sigmoid_z) * sigmoid_z * (1 - sigmoid_z) * x**5

def grad_d(a, b, c, d, e, f, g, h, x, y):
    z = a * x**7 + b * x**6 + c * x**5 + d * x**4 + e * x**3 + f * x**2 + g * x + h
    sigmoid_z = sigmoid(z)
    return -(y - sigmoid_z) * sigmoid_z * (1 - sigmoid_z) * x**4

def grad_e(a, b, c, d, e, f, g, h, x, y):
    z = a * x**7 + b * x**6 + c * x**5 + d * x**4 + e * x**3 + f * x**2 + g * x + h
    sigmoid_z = sigmoid(z)
    return -(y - sigmoid_z) * sigmoid_z * (1 - sigmoid_z) * x**3

def grad_f(a, b, c, d, e, f, g, h, x, y):
    z = a * x**7 + b * x**6 + c * x**5 + d * x**4 + e * x**3 + f * x**2 + g * x + h
    sigmoid_z = sigmoid(z)
    return -(y - sigmoid_z) * sigmoid_z * (1 - sigmoid_z) * x**2

def grad_g(a, b, c, d, e, f, g, h, x, y):
    z = a * x**7 + b * x**6 + c * x**5 + d * x**4 + e * x**3 + f * x**2 + g * x + h
    sigmoid_z = sigmoid(z)
    return -(y - sigmoid_z) * sigmoid_z * (1 - sigmoid_z) * x

def grad_h(a, b, c, d, e, f, g, h, x, y):
    z = a * x**7 + b * x**6 + c * x**5 + d * x**4 + e * x**3 + f * x**2 + g * x + h
    sigmoid_z = sigmoid(z)
    return -(y - sigmoid_z) * sigmoid_z * (1 - sigmoid_z)

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def gradDS(datasetx, datasety, a, b, c, d, e, f, g, h):
    grad_a_sum = grad_b_sum = grad_c_sum = grad_d_sum = 0
    grad_e_sum = grad_f_sum = grad_g_sum = grad_h_sum = 0

    for i in range(len(datasetx)):
        grad_a_sum += grad_a(a, b, c, d, e, f, g, h, datasetx[i], datasety[i])
        grad_b_sum += grad_b(a, b, c, d, e, f, g, h, datasetx[i], datasety[i])
        grad_c_sum += grad_c(a, b, c, d, e, f, g, h, datasetx[i], datasety[i])
        grad_d_sum += grad_d(a, b, c, d, e, f, g, h, datasetx[i], datasety[i])
        grad_e_sum += grad_e(a, b, c, d, e, f, g, h, datasetx[i], datasety[i])
        grad_f_sum += grad_f(a, b, c, d, e, f, g, h, datasetx[i], datasety[i])
        grad_g_sum += grad_g(a, b, c, d, e, f, g, h, datasetx[i], datasety[i])
        grad_h_sum += grad_h(a, b, c, d, e, f, g, h, datasetx[i], datasety[i])

    return grad_a_sum, grad_b_sum, grad_c_sum, grad_d_sum, grad_e_sum, grad_f_sum, grad_g_sum, grad_h_sum

def dist2_polinomial(a_0, b_0, c_0, d_0, e_0, f_0, g_0, h_0, a_n, b_n, c_n, d_n, e_n, f_n, g_n, h_n):
    return ((a_n - a_0)**2 + (b_n - b_0)**2 + (c_n - c_0)**2 + (d_n - d_0)**2 + 
            (e_n - e_0)**2 + (f_n - f_0)**2 + (g_n - g_0)**2 + (h_n - h_0)**2)**0.5

def gradienteDescendentePolinomial(a_0, b_0, c_0, d_0, e_0, f_0, g_0, h_0, tol, lr):
    datasetx = [0, 1, 2, 3, 4, 5, 6, 7]
    datasety = [0, 0, 0, 0, 1, 1, 1, 1]

    a_n, b_n, c_n, d_n, e_n, f_n, g_n, h_n = a_0, b_0, c_0, d_0, e_0, f_0, g_0, h_0
    i = 0

    while True:
        grad_a, grad_b, grad_c, grad_d, grad_e, grad_f, grad_g, grad_h = gradDS(
            datasetx, datasety, a_n, b_n, c_n, d_n, e_n, f_n, g_n, h_n
        )

        a_n1 = a_n - lr * grad_a
        b_n1 = b_n - lr * grad_b
        c_n1 = c_n - lr * grad_c
        d_n1 = d_n - lr * grad_d
        e_n1 = e_n - lr * grad_e
        f_n1 = f_n - lr * grad_f
        g_n1 = g_n - lr * grad_g
        h_n1 = h_n - lr * grad_h

        err = dist2_polinomial(a_n, b_n, c_n, d_n, e_n, f_n, g_n, h_n, a_n1, b_n1, c_n1, d_n1, e_n1, f_n1, g_n1, h_n1)

        if err <= tol:
            break

        a_n, b_n, c_n, d_n, e_n, f_n, g_n, h_n = a_n1, b_n1, c_n1, d_n1, e_n1, f_n1, g_n1, h_n1
        i += 1

        print(i, err)

    return i, a_n, b_n, c_n, d_n, e_n, f_n, g_n, h_n

print("Carregando...")
g = gradienteDescendentePolinomial(0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1e-7, 1e-5)

print(f"f(x) = 1 / (1 + e^-({g[1]}x^7 + {g[2]}x^6 + {g[3]}x^5 + {g[4]}x^4 + {g[5]}x^3 + {g[6]}x^2 + {g[7]}x + {g[8]}))")



"""
f(x) = 1 / (1 + e^-(0.4827767916190292x^7 + -0.2847783145953203x^6 + -0.3894553410222366x^5 + -0.4019223741928856x^4 + -0.40245997298068337x^3 + -0.4019150724730882x^2 + -0.40152638025649556x + -2.1583536059370343))
"""
