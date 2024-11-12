'''
polinômio de 7° grau:
f(x) = ax^7 + bx^6 + cx^5 + dx^4 + ex^3 + fx^2 + gx + h
f'(x) = 7ax^6 + 6bx^5 + 5cx^4 + 4dx^3 + 3ex^2 + 2fx + g

(arg min)  = (y1 - (a1x^7 + b1x^6 + c1x^5 + d1x^4 + e1x^3 + f1x^2 g1x + h))^2 + ... + (y7 - (a7x^7 + b7x^6 + c7x^5 + d7x^4 + e7x^3 + f7x^2 g7x + h))^2

df(a) = 2*(y1 - (a1x^7 + b1x^6 + c1x^5 + d1x^4 + e1x^3 + f1x^2 g1x + h)) * (-x^7) + ... + até o final
df(b) até df(h)

# gradientes[k] += 2*(y - (coef[0]*(x**7) + coef[1]*(x**6) + coef[2]*(x**5) + coef[3]*(x**4) + coef[4]*(x**3) + coef[5]*(x**2) + coef[6]*(x) + coef[7])) * (-x**(7-k))
'''


dataset_x = [-0.27, 0.1, 0.42, 0.59, 1.1, 1.08, 1.16, 1.33]
dataset_y = [6.33, 8.76, 14.0, 16.3, 36, 40.53, 58.9, 85.3]

coef = [1, 1, 1, 1, 1, 1, 1, 1]


# Gradiente Descendente
def gds(dataset_x, dataset_y, coef):
    n = len(coef)
    gradientes = [0] * n

    for i in range(len(dataset_x)):

        x = dataset_x[i]
        y = dataset_y[i]

        for k in range(n):
            soma = 0

            for j in range(n):
                soma += coef[j]*(x**(n-j-1))
            
            gradientes[k] += 2 * (y - soma) * -(x**(7-k))
    
    return gradientes


# Função para calcular a distância
def distancia(coef1, coef2):
    return sum((coef1[i] - coef2[i]) ** 2 for i in range(len(coef1))) ** (1/2)


# Gradiente Descendente aplicado na fórmula: (Xn+1 = Xn - Lr * gds)
def gds_formula(dataset_x, dataset_y, coef, lr, tol):
    n = len(coef)
    Xn = [9999] * n  # Chute Iicial
    Xn1 = coef
    
    c = 1
    dist = distancia(Xn, Xn1)

    while dist > tol:

        grad = gds(dataset_x, dataset_y, coef)
        Xn = Xn1 + []  # Substitui os valores de Xn para Xn+1

        for i in range(n):
            Xn1[i] = Xn[i] - lr * grad[i] / n

        dist = distancia(Xn, Xn1)
        c += 1

        # teste para ver se está diminuindo ou crescendo
        # if c % 1000 == 0:
        #     print(c, dist, Xn1)
    
    return (c, Xn1)



lr = 1e-6
tol = 5e-5

print("Carregando...")
g = gds_formula(dataset_x, dataset_y, coef, lr, tol)
print(f"f(x) = {round(g[1][0],2)}x^7 + {round(g[1][1],2)}x^6 + {round(g[1][2],2)}x^5 + {round(g[1][3],2)}x^4 + {round(g[1][4],2)}x^3 + {round(g[1][5],2)}x^2 + {round(g[1][6],2)}x + {round(g[1][7],2)}")
