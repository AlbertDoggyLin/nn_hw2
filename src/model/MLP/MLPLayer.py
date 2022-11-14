from typing import Callable, Union, Any
import numpy as np
class Layer:
    def __init__(self, inputDim: int, neuNum:int, 
      actFunc: Callable[[np.ndarray[Any, float]], np.ndarray[Any, float]], actDeri: Callable[[np.ndarray[Any, float]], np.ndarray[Any, float]]):
        self._w: np.ndarray[Any, np.ndarray[Any, float]] = np.zeros((inputDim, neuNum))
        for i in range(inputDim):
            for j in range(neuNum):
                self._w[i][j]=1 if i==j else (np.random.rand()-0.5)*0.3
        self._actFunc: Callable[[np.ndarray[Any, float]], np.ndarray[Any, float]] = actFunc
        self._actDeri: Callable[[np.ndarray[Any, float]], np.ndarray[Any, float]] = actDeri
        self._lastInput:Union[np.ndarray[Any, np.ndarray[Any, float]], None] = None
        self._lastV:Union[np.ndarray[Any, np.ndarray[Any, float]], None] = None
        self._lastOut:Union[np.ndarray[Any, np.ndarray[Any, float]], None] = None
        self._delta:Union[np.ndarray[Any, np.ndarray[Any, float]], None] = None
        self._momentum: np.ndarray[Any, np.ndarray[Any, float]] = np.zeros((inputDim, neuNum))

    def cal(self, input: Union[np.ndarray[Any, np.ndarray], np.ndarray]):
        self._lastInput = input
        self._lastV=input@self._w
        self._lastOut = self._actFunc(self._lastV)
        return self._lastOut

    def _computeDelta(self, neuralWiseSum:np.ndarray):
        self._delta = self._actDeri(self._lastV)*neuralWiseSum

    def adj(self, neuralWiseSum:np.ndarray[Any, np.ndarray[Any, float]], lr:float):
        self._computeDelta(neuralWiseSum)
        if self._delta.shape[0]==0:
            print(self._delta.shape)
        self._momentum=lr*(0.7*self._momentum+self._lastInput.transpose()@self._delta)/self._delta.shape[0]
        self._w+=self._momentum

    def getNeuralWiseSum(self):
        return self._delta@(self._w.transpose())