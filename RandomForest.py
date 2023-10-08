import numpy as np


# Информация об узлах дерева (характеристика, значение характеристики, левая часть дерева, правая часть дерева и
# значение в конечном узле
class Node:
    def __init__(self, feature=None, threshold=None, leftNode=None, rightNode=None, value=None, t=None):
        self.feature = feature
        self.threshold = threshold
        self.leftNode = leftNode
        self.rightNode = rightNode
        self.value = value
        self.t = t

    def is_terminal(self):
        return self.value is not None


class Tree:
    def __init__(self, max_depth=10, min_quantity=18):
        self.max_depth = max_depth
        self.min_quantity = min_quantity
        self.tree = None  # Построенное дерево

    # Обучение алгоритма
    def fit(self, X, target):
        self.tree = self.grow_tree(X, target)

    # Предсказание
    def prediction(self, X, option=False):
        if option:
            return self.confidence(X, self.tree)
        return np.array([self.one_prediction(x, self.tree) for x in X])

    # Расчёт энтропии
    def entropy(self, target):
        unique, count = np.unique(target, return_counts=True)
        p = count / len(target)
        entropy = -np.sum([pi * np.log2(pi) for pi in p if pi > 0])
        gini = np.sum([pi * (1 - pi) for pi in p])
        return gini

    # Возращает наиболее частую метку
    def argmaxT(self, target):
        unique, count = np.unique(target, return_counts=True)
        return unique[np.argmax(count)]

    # Целева функция (прирост информации)
    def information_gain(self, X_column, target, threshold):
        if len(np.unique(target)) == 1:
            return 0

        # Информативность в родителе
        n = len(target)
        parent = self.entropy(target)

        left_indexes = np.argwhere(X_column <= threshold).T[0]
        right_indexes = np.argwhere(X_column > threshold).T[0]

        # Информативность в потомках
        entopy_left, n_left = self.entropy(target[left_indexes]), len(left_indexes)
        entopy_right, n_right = self.entropy(target[right_indexes]), len(right_indexes)

        child = (n_left / n) * entopy_left + (n_right / n) * entopy_right

        return parent - child

    # Лучшее разбиение
    def best_dividing(self, X, target):
        best_feature, best_threshold = None, None
        best_gain = -1

        index = np.random.choice(X.shape[1])

        for i in [index]:
            thresholds = np.unique(X[:, i])
            for threshold in thresholds:
                gain = self.information_gain(X[:, i], target, threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = i
                    best_threshold = threshold
        """
        # Проходим по всем характеристикам и по всем порогам
        features = np.array([X[:, i] for i in range(X.shape[1])])
        np.random.shuffle(features)
        features = features[: round(0.6 * len(features))]
        for i in range(len(features)):
            thresholds = np.unique(features[i, :])
            np.random.shuffle(thresholds)
            thresholds = thresholds[: round(0.6 * len(thresholds))]
            for threshold in thresholds:
                gain = self.information_gain(features[i, :], target, threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = i
                    best_threshold = threshold
        """
        return best_feature, best_threshold

    # Рекурсивный алгоритм построения дерева
    def grow_tree(self, X, target, depth=0):
        # Кол-во элементов
        quantity = len(target)
        # Кол-во меток
        labels = len(np.unique(target))

        # Критерий остановки алгоритма
        if (depth >= self.max_depth or quantity <= self.min_quantity) and labels != 1:
            unique, count = np.unique(target, return_counts=True)
            return Node(value=self.argmaxT(target), t=(count[0] / len(target), count[1] / len(target)))
        elif labels == 1:
            value = self.argmaxT(target)
            if value == 0:
                return Node(value=value, t=(1, 0))
            else:
                return Node(value=value, t=(0, 1))

        l_len, r_len, count = 0, 0, 0
        while (l_len == 0 or r_len == 0) and count < 10:
            best_feature, best_threshold = self.best_dividing(X, target)
            left_indexes = np.argwhere(X[:, best_feature] <= best_threshold).T[0]
            right_indexes = np.argwhere(X[:, best_feature] > best_threshold).T[0]
            l_len = len(left_indexes)
            r_len = len(right_indexes)
            count += 1

        if count == 10:
            unique, count = np.unique(target, return_counts=True)
            return Node(value=self.argmaxT(target), t=(count[0] / len(target), count[1] / len(target)))

        leftNode = self.grow_tree(X[left_indexes, :], target[left_indexes], depth + 1)
        rightNode = self.grow_tree(X[right_indexes, :], target[right_indexes], depth + 1)

        return Node(best_feature, best_threshold, leftNode, rightNode)

    # Рекурсивный алгоритм предсказания
    def one_prediction(self, x, tree):
        # Критерий остановки
        if tree.is_terminal():
            return tree.value

        if x[tree.feature] < tree.threshold:
            return self.one_prediction(x, tree.leftNode)
        return self.one_prediction(x, tree.rightNode)

    # Рекурсивный алгоритм предсказания
    def confidence(self, x, tree):
        # Критерий остановки
        if tree.is_terminal():
            return tree.t

        if x[tree.feature] < tree.threshold:
            return self.confidence(x, tree.leftNode)
        return self.confidence(x, tree.rightNode)


class RandomForest:
    def __init__(self, n_trees, max_depth=10, min_quantity=18):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_quantity = min_quantity
        self.trees = []
        self.quantity = None
        self.samples = []
        self.out_of_bag_samples = []

    def fit(self, X, target):
        self.quantity = X.shape[0]
        for i in range(self.n_trees):
            samples = np.random.choice(self.quantity, self.quantity, replace=True)
            out_of_bag_samples = np.array([j for j in range(self.quantity) if j not in samples])

            tree = Tree(max_depth=self.max_depth, min_quantity=self.min_quantity)
            tree.fit(X[samples, :], target[samples])

            self.trees.append(tree)
            self.samples.append(samples)
            self.out_of_bag_samples.append(out_of_bag_samples)

    def predict(self, X, option=False):
        predictions = []
        if option:
            for i in range(self.n_trees):
                predictions.append(self.trees[i].prediction(X, option=True))
        else:
            for i in range(self.n_trees):
                predictions.append(self.trees[i].prediction(X))

            predictions = [self.trees[0].argmaxT(np.array(predictions)[:, i]) for i in range(X.shape[0])]

        return predictions
