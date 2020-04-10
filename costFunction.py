import numpy as np
import time

t1 = time.time()
x = np.array([6, 9, 16, 21, 24, 29, 34, 41, 45, 50, 55])
y = np.array([18, 27, 50, 63, 72, 85, 100, 120, 135, 150, 165])

def cost(y_prediction, y):
    return (1 / (2 * y.__len__())) * np.sum((y_prediction - y) ** 2)

result = [100]
minimumCostResult = 0
w_best_param = 0
b_best_param = 0
y_prediction = 0
w_param = [i / 1000 for i in range(2900, 3000)]
b_param = [i / 1000 for i in range(0, 1000)]
for w in w_param:
    for b in b_param:
        y_pred = w * x.T + b
        costResult = cost(y_pred, y)
        if costResult < min(result):
            minimumCostResult = costResult
            w_best_param = w
            b_best_param = b
            y_prediction = y_pred
        result.append(costResult)

print(f"minimum cost result: {minimumCostResult}")
print(f"best w param: {w_best_param}")
print(f"best b param: {b_best_param}")
print(f"Y best prediction values:\n {y_prediction}")
print(f"Y real values:\n {y}")

t2 = time.time()
print(f"Execution time: {t2-t1}")
