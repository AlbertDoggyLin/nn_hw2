from typing import Union, Any
import numpy as np
from model.MLP.MLPLayer import Layer
from tqdm import tqdm

class MLP:
    def __init__(self, inputDim: int, outputDim: int = 1, layers: int = 2):
        self._inputLayer: Union[Layer, None] = Layer(
            inputDim, 
            inputDim,
            lambda x: 1/(1+np.exp(-x)), 
            lambda x: (1/(1+np.exp(-x))) * (1 - 1/(1+np.exp(-x))))
        self._outputLayer: Union[Layer, None] = Layer(
            inputDim, 
            outputDim,
            lambda x: 1/(1+np.exp(-x)), 
            lambda x: (1/(1+np.exp(-x))) * (1 - 1/(1+np.exp(-x))))
        self._hiddenLayers: list[Union[Layer, None]] = [Layer(
            inputDim, 
            inputDim,
            lambda x: 1/(1+np.exp(-x)), 
            lambda x: (1/(1+np.exp(-x))) * (1 - 1/(1+np.exp(-x)))) for _ in range(layers-2)]
        self._upperBound:Union[None, np.ndarray[Any, float]]=None
        self._lowerBound:Union[None, np.ndarray[Any, float]]=None

    def predict(self, input:np.ndarray[Any, np.ndarray[Any, float]], normalized:bool=False):
        if not normalized: nextInput = self._adjustData(input)
        nextInput = self._inputLayer.cal(input)
        for layer in self._hiddenLayers:
            nextInput = layer.cal(nextInput)
        if normalized: return self._outputLayer.cal(nextInput)
        else: return self._outputLayer.cal(nextInput)*(self._upperBound[-1]-self._lowerBound[-1])+self._lowerBound[-1]

    def kickback(self, Y:np.ndarray[Any, np.ndarray], Yhat:np.ndarray[Any, np.ndarray], lr:float=0.1):
        self._outputLayer.adj(Y-Yhat, lr)
        nextNeuWiseSum = self._outputLayer.getNeuralWiseSum()
        for layer in self._hiddenLayers[-1::]:
            layer.adj(nextNeuWiseSum, lr)
            nextNeuWiseSum = layer.getNeuralWiseSum()
        self._inputLayer.adj(nextNeuWiseSum, lr)

    def loss(self, X: np.ndarray[Any, np.ndarray], Y: np.ndarray[Any, np.ndarray], normalized:bool=False):
        if not normalized: X, Y = self._adjustData(X, Y)
        Ydiff = self.predict(X, normalized=True) - Y
        return sum(Ydiff**2/X.shape[0])[0]
    
    def _adjustData(self, X:Union[np.ndarray, np.ndarray[Any, np.ndarray[Any, float]]], Y:Union[np.ndarray, np.ndarray[Any, np.ndarray[Any, float]], None] = None):
        if Y is None:
            return (X-self._lowerBound[0:-1])/(self._upperBound[0:-1]-self._lowerBound[0:-1])
        data = np.hstack((X, Y))
        adjustedData:np.ndarray = (data-self._lowerBound)/(self._upperBound-self._lowerBound)
        return adjustedData[:, 0:-1], adjustedData[:, -1:]

    def fit(self, X: np.ndarray, Y: np.ndarray, epoches: int = 1000, lr: float = 0.8):
        X, Y = X, Y
        def setBound(X:Union[np.ndarray, np.ndarray[Any, np.ndarray[Any, float]]], Y:Union[np.ndarray, np.ndarray[Any, np.ndarray[Any, float]]]):
            data = np.hstack((X, Y))
            self._upperBound=np.max(data, axis=0)
            self._lowerBound=np.min(data, axis=0)
        setBound(X, Y)
        X, Y = self._adjustData(X, Y)
        n=X.shape[0]
        currentLoss=self.loss(X, Y, normalized=True)
        pbar:tqdm=tqdm(range(epoches),total=epoches, desc="training progress", postfix={'current loss':currentLoss}, ncols=90)
        for i in pbar:
            Yhat = self.predict(X, normalized=True)
            self.kickback(Y, Yhat, lr)
            if not i%500:
                currentLoss=self.loss(X, Y, normalized=True)
                pbar.set_postfix({'current loss':f'{currentLoss:.3f}'})
    
    def pt(self):
        print(self._inputLayer._w.transpose())
        print()
        for layer in self._hiddenLayers:
            print(layer._w.transpose())
            print()
        print(self._outputLayer._w.transpose())
        print('\n')
