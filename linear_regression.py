
dataset_x = [-0.27, 0.1, 0.42, 0.59, 1.1, 1.08, 1.16, 1.33]
dataset_y = [6.33, 8.76, 14.0, 16.3, 36, 40.53, 58.9, 85.3]

coef = [1, 1, 1, 1, 1, 1, 1, 1]


# Gradiente
def gd(dataset_x, dataset_y, coef):
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
    return sum((coef1[i] - coef2[i]) ** 2 for i in range(len(coef1))) ** 0.5


# Gradiente Descendente: (Xn+1 = Xn - Lr * gd)
def gds(dataset_x, dataset_y, coef, lr, tol):
    n = len(coef)
    Xn = [9999] * n  # Chute Iicial
    Xn1 = coef
    
    c = 1
    dist = distancia(Xn, Xn1)

    while dist > tol:

        grad = gd(dataset_x, dataset_y, coef)
        Xn = Xn1 + []  # Substitui os valores de Xn para Xn+1

        for i in range(n):
            Xn1[i] = Xn[i] - lr * grad[i] / n

        dist = distancia(Xn, Xn1)
        c += 1
    
    return (c, Xn1)



lr = 1e-6
tol = 5e-5

print("Carregando...")
g = gds(dataset_x, dataset_y, coef, lr, tol)

print(f"\nf(x) = {round(g[1][0],2)}x^7 + {round(g[1][1],2)}x^6 + {round(g[1][2],2)}x^5 + {round(g[1][3],2)}x^4 + {round(g[1][4],2)}x^3 + {round(g[1][5],2)}x^2 + {round(g[1][6],2)}x + {round(g[1][7],2)}")
