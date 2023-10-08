import csv
import random
import numpy as np


# Считывание данных из файла
def read():
    with open("titanic.csv", "r") as file:
        reader = csv.reader(file)
        matrix = []
        for row in reader:
            string = row[:2] + row[3:]
            matrix.append(string)
        matrix = matrix[1:]
        return matrix


# Подготовка данных
def preparing(data):
    matrix = []
    t = []
    random.shuffle(data)
    for i in data:
        t.append(i[0])
        matrix.append(i[1:])
    for j in matrix:
        if j[1] == "male":
            j[1] = 0
        else:
            j[1] = 1
    matrix = np.array(matrix, dtype=float)
    t = np.array(t, dtype=int)
    return matrix, t


# Разделение выборки
def divide(data, t, train_len, validation_len, test_len):
    train = np.array(data[:train_len])
    validation = np.array(data[train_len:train_len + validation_len])
    test = np.array(data[len(data) - test_len:])

    t_train = np.array(t[:train_len])
    t_validation = np.array(t[train_len:train_len + validation_len])
    t_test = np.array(t[len(data) - test_len:])

    return train, validation, test, t_train, t_validation, t_test
