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
    print(f'loss for test dataset:{model.loss(testX, testy)}')
    return model