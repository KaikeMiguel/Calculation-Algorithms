import csv
import math
import matplotlib.pyplot as plt

# Caminho para o arquivo CSV
file_path = "./src/heart_failure.csv"

# Carregar o dataset manualmente
dataset_x = []  # Variável independente: age
dataset_y = []  # Variável dependente: DEATH_EVENT

with open(file_path, mode='r', newline='', encoding='utf-8') as file:
    reader = csv.reader(file)
    headers = next(reader)  # Pular cabeçalho
    
    for row in reader:
        try:
            age = float(row[0])
            death_event = int(row[-1])
            dataset_x.append(age)
            dataset_y.append(death_event)
        except ValueError:
            # Pular linhas com dados inválidos
            continue

# Função Sigmoide
def sigmoid(z):
    return 1 / (1 + math.exp(-z))

# Função para calcular o gradiente de a e b
def grad_a(a, b, x, y):
    z = a * x + b
    e = sigmoid(z)
    return (e - y) * x

def grad_b(a, b, x, y):
    z = a * x + b
    e = sigmoid(z)
    return e - y

# Função para calcular a distância euclidiana entre parâmetros
def dist2_polinomial(a0, b0, a1, b1):
    return math.sqrt((a1 - a0)**2 + (b1 - b0)**2)

# Função para calcular a Acurácia
def acuracia(dataset_x, dataset_y, a, b):
    acertos = 0
    for i in range(len(dataset_x)):
        x = dataset_x[i]
        y = dataset_y[i]
        z = a * x + b
        y_pred = 1 if sigmoid(z) >= 0.5 else 0
        if y_pred == y:
            acertos += 1
    return acertos / len(dataset_x)

# Função para calcular o F1-Score
def f1_score(dataset_x, dataset_y, a, b):
    vP = 0  # Verdadeiro positivo
    fP = 0  # Falso positivo
    fN = 0  # Falso negativo

    for i in range(len(dataset_x)):
        x = dataset_x[i]
        y = dataset_y[i]
        z = a * x + b
        y_pred = 1 if sigmoid(z) >= 0.5 else 0

        if y_pred == 1 and y == 1:
            vP += 1
        elif y_pred == 1 and y == 0:
            fP += 1
        elif y_pred == 0 and y == 1:
            fN += 1

    precisao = vP / (vP + fP) if (vP + fP) > 0 else 0
    recall = vP / (vP + fN) if (vP + fN) > 0 else 0

    if precisao + recall > 0:
        return 2 * (precisao * recall) / (precisao + recall)
    return 0

# Gradiente Descendente para Regressão Logística
def logistic_regression_gd(x, y, a_init, b_init, lr, tol, max_iter):
    a, b = a_init, b_init
    n = len(x)
    cost_history = []
    
    for iteration in range(1, max_iter + 1):
        grad_a_total = 0.0
        grad_b_total = 0.0
        cost = 0.0
        
        for i in range(n):
            z = a * x[i] + b
            pred = sigmoid(z)
            error = y[i] - pred
            grad_a_total += error * x[i]
            grad_b_total += error
            # Calcular log-loss
            if pred == 0:
                pred = 1e-15  # Evitar log(0)
            elif pred == 1:
                pred = 1 - 1e-15
            cost += -y[i] * math.log(pred) - (1 - y[i]) * math.log(1 - pred)
        
        # Média da função de custo
        cost /= n
        cost_history.append(cost)
        
        # Atualizar parâmetros
        a_new = a + lr * grad_a_total / n
        b_new = b + lr * grad_b_total / n
        
        # Verificar convergência
        if dist2_polinomial(a, b, a_new, b_new) < tol:
            print(f"Convergência alcançada na iteração {iteration}")
            return a_new, b_new, iteration, cost_history
        
        a, b = a_new, b_new
        
        # Cálculo das métricas (Acurácia e F1-Score) a cada 1000 iterações
        if iteration % 1000 == 0:
            acu = acuracia(x, y, a, b)
            f1 = f1_score(x, y, a, b)
            print(f"Iteração {iteration}: Custo = {cost:.6f}, Acurácia = {acu:.2%}, F1-Score = {f1:.2f}")
    
    print("Número máximo de iterações alcançado.")
    return a, b, max_iter, cost_history

# Normalizar os dados (Z-score)
def normalize(data):
    mean = sum(data) / len(data)
    variance = sum((xi - mean) ** 2 for xi in data) / len(data)
    std_dev = math.sqrt(variance)
    return [(xi - mean) / std_dev for xi in data]

# Normalizar a variável 'age'
normalized_x = normalize(dataset_x)

# Parâmetros iniciais
a_initial = 0.0
b_initial = 0.0
learning_rate = 0.01  # Taxa de aprendizado
tolerance = 1e-9      # Tolerância para convergência
max_iterations = 100000  # Número máximo de iterações

print("Iniciando o treinamento da Regressão Logística...")
a_final, b_final, iterations, cost_history = logistic_regression_gd(
    normalized_x, dataset_y, a_initial, b_initial, learning_rate, tolerance, max_iterations
)

# Cálculo final de custo, acurácia e F1-Score
final_cost = cost_history[-1]
final_accuracy = acuracia(normalized_x, dataset_y, a_final, b_final)
final_f1_score = f1_score(normalized_x, dataset_y, a_final, b_final)

# Exibir os resultados finais
print(f"\nResultados finais após {iterations} iterações:")
print(f"Custo final: {final_cost:.6f}")
print(f"Acurácia final: {final_accuracy:.2%}")
print(f"F1-Score final: {final_f1_score:.2f}")

# Gerar pontos para plotar a curva sigmoide ajustada com classificação binária
x_plot = [normalized_x[i] for i in sorted(range(len(normalized_x)), key=lambda i: normalized_x[i])]
y_plot = [1 if sigmoid(a_final * xi + b_final) >= 0.5 else 0 for xi in x_plot]

# Plotar os dados e a curva ajustada com classificação binária
plt.figure(figsize=(14, 6))

# Subplot 1: Dados e Curva Sigmoide com classificação binária
plt.subplot(1, 2, 1)
plt.scatter(normalized_x, dataset_y, alpha=0.6, label='Dados reais', color='blue')
plt.plot(x_plot, y_plot, color='red', label='Curva Sigmoide Ajustada (classificação binária)')
plt.title("Regressão Logística - Idade Normalizada vs. Morte")
plt.xlabel("Idade Normalizada")
plt.ylabel("Classificação de Morte (0 ou 1)")
plt.legend()
plt.grid(True)

# Subplot 2: Histórico da Função de Custo
plt.subplot(1, 2, 2)
plt.plot(cost_history, color='green')
plt.title("Histórico da Função de Custo (Log-loss)")
plt.xlabel("Iterações")
plt.ylabel("Custo")
plt.grid(True)

plt.tight_layout()
plt.show()
