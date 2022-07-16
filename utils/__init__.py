def accuracy(y_pred, y):
    return sum(y_pred == y) / len(y)