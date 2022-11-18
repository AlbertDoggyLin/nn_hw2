from typing import Union, Callable, Any
import numpy as np
from model.RBFN.RBFNLayer import Layer
from tqdm import tqdm
class RBFN:
    def __init__(self, neuCount: int = 2):
        self._neuCount:int=neuCount
        self._baseLayer:Union[Layer, None]=None
        self._actFunc:Union[None, Callable[[np.array[np.array[float]]], np.array[np.array[float]]]]=None
        self._upperBound:Union[None, np.ndarray[Any, float]]=None
        self._lowerBound:Union[None, np.ndarray[Any, float]]=None
    
    def _getMid(self, input: np.ndarray):
        mids = np.zeros((self._neuCount, input.shape[1]))
        midsAfter: np.ndarray = input[np.random.choice(input.shape[0], self._neuCount, replace=False)]
        while True:
            checkFlag = True
            for i in range(self._neuCount):
                if np.linalg.norm(mids[i]-midsAfter[i], 1):
                    checkFlag = False
                    break
            if checkFlag:
                break
            mids = midsAfter
            midsAfter = np.zeros(mids.shape)
            pointsCounter = np.zeros(mids.shape[0])
            for points in input:
                lenWithMids = [np.linalg.norm(points-mid) for mid in mids]
                maxIndex = np.argmin(lenWithMids)
                midsAfter[maxIndex] += points
                pointsCounter[maxIndex] += 1
            for i in range(self._neuCount):
                if pointsCounter[i]!=0:midsAfter[i] /= pointsCounter[i]
        self.mids=mids
        return mids
    
    def setBases(self, X:np.ndarray[Any, np.ndarray[Any, float]], Y):
        #mids=self._getMid(np.hstack((X, Y)))[:,0:-1]
        mids=self._getMid(X)
        def layerActFunction(input, mid):
            baseDis=np.linalg.norm(input-mid)
            return 1/np.sqrt(baseDis*baseDis+0.1)
        self._baseLayer=Layer(mids, layerActFunction)

    def predict(self, input:np.ndarray[Any, np.ndarray[Any, float]], normalized:bool=False, first=False):
        if not normalized: input = self._adjustData(input)
        if self._baseLayer is None: raise Exception('Please fit first, no base layer found')
        if normalized: return self._baseLayer.cal(input, first=first)
        else: return self._baseLayer.cal(input, first=True)*(self._upperBound[-1]-self._lowerBound[-1])+self._lowerBound[-1]

    def kickback(self, Y:np.ndarray[Any, np.ndarray[Any, float]], Yhat:np.ndarray[Any, np.ndarray[Any, float]], lr:float=0.1):
        self._baseLayer.adj(Y-Yhat)

    def loss(self, X: np.ndarray[Any, np.ndarray], Y: np.ndarray[Any, np.ndarray], normalized:bool=False):
        if not normalized: X, Y = self._adjustData(X, Y)
        Ydiff = self.predict(X, normalized=True, first=True) - Y
        return sum(Ydiff**2/X.shape[0])[0]
    
    def _adjustData(self, X:Union[np.ndarray, np.ndarray[Any, np.ndarray[Any, float]]], Y:Union[np.ndarray, np.ndarray[Any, np.ndarray[Any, float]], None] = None):
        if Y is None:
            return (X-self._lowerBound[0:-1])/(self._upperBound[0:-1]-self._lowerBound[0:-1])
        data = np.hstack((X, Y))
        adjustedData:np.ndarray = (data-self._lowerBound)/(self._upperBound-self._lowerBound)
        return adjustedData[:, 0:-1], adjustedData[:, -1:]

    def fit(self, X: np.ndarray, Y: np.ndarray, epoches: int = 1000, lr: float = 0.01):
        X, Y = X, Y
        def setBound(X:Union[np.ndarray, np.ndarray[Any, np.ndarray[Any, float]]], Y:Union[np.ndarray, np.ndarray[Any, np.ndarray[Any, float]]]):
            data = np.hstack((X, Y))
            self._upperBound=np.max(data, axis=0)
            self._lowerBound=np.min(data, axis=0)
        setBound(X, Y)
        X, Y = self._adjustData(X, Y)
        self.setBases(X, Y)
        self.predict(X, normalized=True, first=True)
        currentLoss=self.loss(X, Y, normalized=True)
        pbar:tqdm=tqdm(range(epoches),total=epoches, desc="training progress", postfix={'current loss':currentLoss}, ncols=90)
        for i in pbar:
            Yhat = self.predict(X, normalized=True)
            self.kickback(Y, Yhat, lr)
            if not i%500:
                currentLoss=self.loss(X, Y, normalized=True)
                pbar.set_postfix({'current loss':f'{currentLoss:.3f}'})
    
    def pt(self):
        print()
        print(self._baseLayer._mids)
        print(self._baseLayer._w.transpose())
        print(self._baseLayer._theta)
        print('\n')
