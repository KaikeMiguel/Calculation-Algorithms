'''
polinômio de 7° grau:
f(x) = ax^7 + bx^6 + cx^5 + dx^4 + ex^3 + fx^2 + gx + h
f'(x) = 7ax^6 + 6bx^5 + 5cx^4 + 4dx^3 + 3ex^2 + 2fx + g

(arg min)  = (y1 - (a1x^7 + b1x^6 + c1x^5 + d1x^4 + e1x^3 + f1x^2 g1x + h))^2 + ... + (y7 - (a7x^7 + b7x^6 + c7x^5 + d7x^4 + e7x^3 + f7x^2 g7x + h))^2

df(a) = 2*(y1 - (a1x^7 + b1x^6 + c1x^5 + d1x^4 + e1x^3 + f1x^2 g1x + h)) * (-x^7) + ... + até o final
df(b) até df(h)
'''


db_x = [0,1,2,3,4,5,6,7]
db_y = [0, 1, 4, 9, 16, 25, 36, 49]
coef = [1, 1, 1, 1, 1, 1, 1, 1]



def gradiente_descendente(db_x, db_y, coef):
    # declarando o gradiente descendcente de todos os coefientes
    gradientes = [0] * len(coef)

    for i in range(len(db_x)):

        x = db_x[i]
        y = db_y[i]

        for k in range(len(coef)):
            # deriva de "a" até "h"
            gradientes[k] += 2*(y - (coef[0]*(x**7) + coef[1]*(x**6) + coef[2]*(x**5) + coef[3]*(x**4) + coef[4]*(x**3) + coef[5]*(x**2) + coef[6]*(x) + coef[7])) * (-x**(7-k))
    
    return gradientes



print(gradiente_descendente(db_x, db_y, coef))
