import random
import titanic
import iris
import digits
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.datasets import load_digits
from tqdm import tqdm
import numpy as np
from RandomForest import RandomForest

"""
# Считывание данных (титаник)
X = titanic.read()
X, target = titanic.preparing(X)

# Разделение выборки
N = len(X)
train_len = int(N * 0.8)
validation_len = int(N * 0.1)
test_len = int(N * 0.1)
Train, Validation, Test, T_Train, T_Validation, T_Test = titanic.divide(X, target, train_len, validation_len, test_len)
"""

"""
# Считывание данных (ирисы)
X, target = load_iris(return_X_y=True)
x1, target1 = X[0], target[0]
x2, target2 = X[1], target[1]
X, target = X[2:], target[2:]

# Разделение выборки (ирисы)
N = len(X)
train_len = int(N * 0.8)
validation_len = int(N * 0.1)
test_len = int(N * 0.1)
Train, Validation, Test, T_Train, T_Validation, T_Test = iris.splitting(X, target, train_len, validation_len, test_len)
"""

# Считывание данных (цифры)
Digits = load_digits()
N = len(Digits["data"])

# Длины выборок
train = int(N * 0.7)  # 80%
validation = int(N * 0.2)  # 10%
test = int(N * 0.1)  # 10%

# Стандартизация, разделение
data = digits.standardisation(Digits["data"])
Train, Validation, Test, T_Train, T_Validation, T_Test = digits.splitting(data, Digits["target"], train, validation, test)


accuracy = []
forests = []
for i in tqdm(range(10)):
    classifier = RandomForest(n_trees=random.randint(3, 30), max_depth=random.randint(5, 20), min_quantity=random.randint(10, 50))
    classifier.fit(Train, T_Train)
    prediction = classifier.predict(Validation)
    acc = np.sum(prediction == T_Validation) / len(T_Validation)
    accuracy.append(acc)
    forests.append(classifier)

index = accuracy.index(max(accuracy))
classifier = forests[index]
print(f"Best forest:\nMax depth: {classifier.max_depth}\nQuantity of trees: {classifier.n_trees}\nMin quantity: {classifier.min_quantity}")

prediction = classifier.predict(Test)
print(np.sum(prediction == T_Test) / len(T_Test))


"""
test_score = []
n_trees = []

for i in tqdm(range(3, 30)):
    classifier = RandomForest(n_trees=i)
    classifier.fit(Train, T_Train)
    n_trees.append(i)
    prediction = classifier.predict(Test)
    acc = np.sum(prediction == T_Test) / len(T_Test)
    test_score.append(acc)

plt.plot(n_trees, test_score)
plt.show()
"""

"""
# Вектора уверенности (титаник)
x = np.array([3, 0, 22, 1, 0, 7.25], dtype=float)  # Class 0
pr = classifier.predict(x, option=True)
print(pr)
T0 = 0
T1 = 0
for i in pr:
    T0 += i[0]
    T1 += i[1]
print(T0 / classifier.n_trees, T1 / classifier.n_trees)

print()

x = np.array([1, 1, 38, 1, 0, 71.2833], dtype=float)  # Class 1
pr = classifier.predict(x, option=True)
print(pr)
T0 = 0
T1 = 0
for i in pr:
    T0 += i[0]
    T1 += i[1]
print(T0 / classifier.n_trees, T1 / classifier.n_trees)
"""
