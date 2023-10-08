import numpy as np
import random


# Разделяем датасет на выборки и перемешиваем
def splitting(data, target, train_len, validation_len, test_len):
    train = np.zeros((train_len, len(data[0])))
    validation = np.zeros((validation_len, len(data[0])))
    test = np.zeros((test_len, len(data[0])))
    train_target = np.zeros(train_len, dtype=int)
    validation_target = np.zeros(validation_len, dtype=int)
    test_target = np.zeros(test_len, dtype=int)
    N = len(data)
    indexes = []

    for i in range(train_len):
        while True:
            index = random.randint(0, N - 1)
            if len(indexes) == 0:
                indexes.append(index)
                train[i] = data[index]
                train_target[i] = target[index]
            else:
                for ind in indexes:
                    if ind == index:
                        continue
                indexes.append(index)
                train[i] = data[index]
                train_target[i] = target[index]
            break

    for j in range(validation_len):
        while True:
            index = random.randint(0, N - 1)
            for ind in indexes:
                if ind == index:
                    continue
            indexes.append(index)
            validation[j] = data[index]
            validation_target[j] = target[index]
            break

    for k in range(test_len):
        while True:
            index = random.randint(0, N - 1)
            for ind in indexes:
                if ind == index:
                    continue
            indexes.append(index)
            test[k] = data[index]
            test_target[k] = target[index]
            break

    return train, validation, test, train_target, validation_target, test_target