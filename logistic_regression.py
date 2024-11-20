'''
polinômio de 7° grau:
f(x) = ax^7 + bx^6 + cx^5 + dx^4 + ex^3 + fx^2 + gx + h
f'(x) = 7ax^6 + 6bx^5 + 5cx^4 + 4dx^3 + 3ex^2 + 2fx + g


# regressão logística
(arg min) = (y1 - 1/(1 + e^-(ax^7 + bx^6 + cx^5 + dx^4 + ex^3 + fx^2 + gx + h)))^2 + ... + (y7 - 1/(1 + e^-(ax^7 + bx^6 + cx^5 + dx^4 + ex^3 + fx^2 + gx + h)))^2

d(a) = 2*(y1 - 1/(1 + e^-(ax^7 + ... + h))) * ((x^7 * e^-(ax^7 + ... + h))/(1 + e^-(ax^7 + ... + h))^2) + ... + 2*(y7 - 1/(1 + e^-(ax^7 + ... + h))) * ((x^0 * e^-(ax^7 + ... + h))/(1 + e^-(ax^7 + ... + h))^2)
d(b) até d(h)
'''





# Gradiente Descendente
# def gds(dataset_x, dataset_y, coef):
    # n = len(coef)
    # gradientes = [0] * n

    # for i in range(len(dataset_x)):

    #     x = dataset_x[i]
    #     y = dataset_y[i]

    #     z = 0

    #     for j in range(n):
    #         z += coef[j]*(x**(n-j-1))

    #     e = math.exp(-z)
    #     sigmoid = 1 / (1 + e)


    #     derivadas = [0] * n

    #     for j in range(n):
    #         derivadas[j] = (e * (x**(n-j-1))) / ((1 + e)**2)
    
    
    # coef = [1, 1, 1, 1, 1, 1, 1, 1]


import math


dataset_x = [1,2,3,4,5,6,7,8]
dataset_y = [1,1,1,1,3,3,3,3]

a = 1
b = 1
c = 1
d = 1
e = 1
f = 1
g = 1
h = 1

# -----------------------------------------------------------------------------------------------------------------------------------
def grad(dataset_x, dataset_y, a, b, c, d, e, f, g, h):

    x = dataset_x
    y = dataset_y

    print(f"grad: x = {x}")
    print(f"grad: y = {y}\n")

    grad_a = 0
    grad_b = 0
    grad_c = 0
    grad_d = 0
    grad_e = 0
    grad_f = 0
    grad_g = 0
    grad_h = 0

    for i in range(len(dataset_x)):

        z = a*(x[i]**7) + b*(x[i]**6) + c*(x[i]**5) + d*(x[i]**4) + e*(x[i]**3) + f*(x[i]**2) + g*(x[i]**1) + h*(x[i]**0)
        euler = math.exp(-z)
        sigmoid = 1 / (1 + euler)

        print(f"grad FOR i: z = {z}")
        print(f"grad FOR i: euler = {euler}")
        print(f"grad FOR i: sigmoid = {sigmoid}\n")

        df_a = (euler * (x[i]**7)) / ((1 + euler)**2)
        grad_a += 2 * ((y[i] - sigmoid) * -(df_a))

        df_b = (euler * (x[i]**6)) / ((1 + euler)**2)
        grad_b += 2 * ((y[i] - sigmoid) * -(df_b))

        df_c = (euler * (x[i]**5)) / ((1 + euler)**2)
        grad_c += 2 * ((y[i] - sigmoid) * -(df_c))

        df_d = (euler * (x[i]**4)) / ((1 + euler)**2)
        grad_d += 2 * ((y[i] - sigmoid) * -(df_d))

        df_e = (euler * (x[i]**3)) / ((1 + euler)**2)
        grad_e += 2 * ((y[i] - sigmoid) * -(df_e))

        df_f = (euler * (x[i]**2)) / ((1 + euler)**2)
        grad_f += 2 * ((y[i] - sigmoid) * -(df_f))

        df_g = (euler * (x[i]**1)) / ((1 + euler)**2)
        grad_g += 2 * ((y[i] - sigmoid) * -(df_g))

        df_h = (euler * (x[i]**0)) / ((1 + euler)**2)
        grad_h += 2 * ((y[i] - sigmoid) * -(df_h))

        print(f"grad FOR i:")
        print(f"grad_a = {grad_a}")
        print(f"grad_b = {grad_b}")
        print(f"grad_c = {grad_c}")
        print(f"grad_d = {grad_d}")
        print(f"grad_e = {grad_e}")
        print(f"grad_f = {grad_f}")
        print(f"grad_g = {grad_g}")
        print(f"grad_h = {grad_h}")


    return (grad_a, grad_b, grad_c, grad_d, grad_e, grad_f, grad_g, grad_h)
# -----------------------------------------------------------------------------------------------------------------------------------
def dist(an1, bn1, cn1, dn1, en1, fn1, gn1, hn1, an, bn, cn, dn, en, fn, gn, hn):
    return ((an1 - an)**2 + (bn1 - bn)**2 + (cn1 - cn)**2 + (dn1 - dn)**2 + (en1 - en)**2 + (fn1 - fn)**2 + (gn1 - gn)**2 + (hn1 - hn)**2) ** 0.5
# -----------------------------------------------------------------------------------------------------------------------------------
def grad_des(dataset_x, dataset_y, a_o, b_o, c_o, d_o, e_o, f_o, g_o, h_o, tol, lr):
    an = a_o
    bn = b_o
    cn = c_o
    dn = d_o
    en = e_o
    fn = f_o
    gn = g_o
    hn = h_o

    i = 1

    print("--------------------------------------")

    while True:
        an1 = an - lr * grad(dataset_x, dataset_y, an, bn, cn, dn, en, fn, gn, hn)[0]
        bn1 = bn - lr * grad(dataset_x, dataset_y, an, bn, cn, dn, en, fn, gn, hn)[1]
        cn1 = cn - lr * grad(dataset_x, dataset_y, an, bn, cn, dn, en, fn, gn, hn)[2]
        dn1 = dn - lr * grad(dataset_x, dataset_y, an, bn, cn, dn, en, fn, gn, hn)[3]
        en1 = en - lr * grad(dataset_x, dataset_y, an, bn, cn, dn, en, fn, gn, hn)[4]
        fn1 = fn - lr * grad(dataset_x, dataset_y, an, bn, cn, dn, en, fn, gn, hn)[5]
        gn1 = gn - lr * grad(dataset_x, dataset_y, an, bn, cn, dn, en, fn, gn, hn)[6]
        hn1 = hn - lr * grad(dataset_x, dataset_y, an, bn, cn, dn, en, fn, gn, hn)[7]

        print(f"\ngrad_des: while True")
        print(f"an1 = {an1}")
        print(f"bn1 = {bn1}")
        print(f"cn1 = {cn1}")
        print(f"dn1 = {dn1}")
        print(f"en1 = {en1}")
        print(f"fn1 = {fn1}")
        print(f"gn1 = {gn1}")
        print(f"hn1 = {hn1}")

        i += 1

        if dist(an1, bn1, cn1, dn1, en1, fn1, gn1, hn1, an, bn, cn, dn, en, fn, gn, hn) > tol:
            an = an1
            bn = bn1
            cn = cn1
            dn = dn1
            en = en1
            fn = fn1
            gn = gn1
            hn = hn1

        else:
            break
    
    print("--------------------------------------")

    return (an, bn, cn, dn, en, fn, gn, hn)
# -----------------------------------------------------------------------------------------------------------------------------------

lr = 1e-6
tol = 1e-6


print("Carregando...")
g = grad_des(dataset_x, dataset_y, a, b, c, d, e, f, g, h, tol, lr)
print(f"f(x) = {round(g[0],2)}x^7 + {round(g[1],2)}x^6 + {round(g[2],2)}x^5 + {round(g[3],2)}x^4 + {round(g[4],2)}x^3 + {round(g[5],2)}x^2 + {round(g[6],2)}x + {round(g[7],2)}")
