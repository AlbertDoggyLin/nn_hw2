import numpy as np
from model.MLP import MLP
from model.RBFN import RBFN
from sklearn.model_selection import train_test_split

def runDataOnMLP(track: list[str], dataset: list[str]):
    """track must be list seperated by comma, dataset must be seperated by space"""
    track = [list(map(lambda x:float(x), i.split(','))) for i in track]
    dataset: np.ndarray = np.array(
        [list(map(lambda x:float(x), i.split())) for i in dataset])
    X = []
    for data in dataset[:, 0:-1]:
        tmp = np.array(data)
        X.append(np.concatenate((tmp[0:-3], tmp[-3:]-6)))
    y = dataset[:, -1:]
    X = np.array(X)
    model = MLP(X.shape[1], layers=4)
    trainX, testX, trainy, testy = train_test_split(
        X, y, test_size=0.3, random_state=42)
    model.fit(trainX, trainy, epoches=100000)
    for i in list(zip(X, y)):
        print(i[0], model.predict(i[0]), i[1])
    print(f'loss for test dataset:{model.loss(testX, testy)}')
    return model

def runDataOnRBFN(track: list[str], dataset: list[str]):
    """track must be list seperated by comma, dataset must be seperated by space"""
    track = [list(map(lambda x:float(x), i.split(','))) for i in track]
    dataset: np.ndarray = np.array(
        [list(map(lambda x:float(x), i.split())) for i in dataset])
    X = []
    for data in dataset[:, 0:-1]:
        tmp = np.array(data)
        X.append(np.concatenate((tmp[0:-3], tmp[-3:]-6)))
    y = dataset[:, -1:]
    X = np.array(X)
    model = RBFN(neuCount=30)
    trainX, testX, trainy, testy = train_test_split(
        X, y, test_size=0.3, random_state=42)
    model.fit(trainX, trainy, epoches=10000)
    for i in list(zip(X, y))[0:100]:
        print(i[0], model.predict(np.array([i[0]]), first=True), i[1])
    print(f'loss for test dataset:{model.loss(testX, testy)}')
    return model