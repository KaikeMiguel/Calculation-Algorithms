import math
import csv


path = "./dataset/heart_failure_clinical_records_dataset.csv"

# Abre o dataset e pega as colunas "age" (dataset_x) e "death" (dataset_y)
with open(path, mode='r', newline='', encoding='utf-8') as file:
    leitor = csv.reader(file)

    primeira_linha = next(leitor) 

    n = len(primeira_linha) # Pega a quantidade de colunas

    next(leitor) # Pula o cabeçalho da dataset

    dataset_x = [] # age
    dataset_y = [] # death

    for linha in leitor:
        dataset_x.append(linha[0])
        dataset_y.append(linha[n-1])


coeficientes = [0,0]

def sigmoid(z):
    e = math.exp(-z)
    return 1 / (1 + e)

def derivadas(co,x,y):
    n = len(co)
    z = 0
    for i in range(n):
        z += co[i]*(x**(n-i-1)) # for p/ (ax + b)
    
    e = sigmoid(z)

    return 


def grad(dataset_x, dataset_y, co):
    n = len(co)
    gradientes = [0] * n

    for i in range(len(dataset_x)):
        x = dataset_x[i]
        y = dataset_y[i]
