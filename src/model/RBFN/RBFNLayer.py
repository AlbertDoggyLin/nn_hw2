from typing import Callable, Any
import numpy as np
class Layer:
    def __init__(self, mids:np.ndarray[Any, np.ndarray[Any, float]],
     actFunc:Callable[[np.ndarray[Any, np.ndarray[Any, float]], np.ndarray[Any, np.ndarray[Any, float]]], np.ndarray[Any, float]]):
        self._w:np.ndarray[Any, np.ndarray[Any, float]]=np.random.random((mids.shape[0], 1))*2-1
        self._actFunc:Callable[[np.ndarray[Any, float], np.ndarray[Any, float]], float]=actFunc
        self._mids:np.ndarray[Any, np.ndarray[Any, float]]=mids
        self._theta:float=np.random.random()*2-1
        self._lastOut:np.ndarray[Any, np.ndarray[Any, float]]
        self._lastDeltaW:np.ndarray[Any, np.ndarray[Any, float]]=np.zeros(self._w.shape)
        self._lastDeltaTheta:float=0


    def cal(self, input:np.ndarray[Any, np.ndarray[Any, float]], first=False):
        if first:
            neuOuts=[]
            for x in input:
                neuOuts.append([])
                for mid in self._mids:
                    neuOuts[-1].append(self._actFunc(x, mid))
            self._lastOut=np.array(neuOuts)
        return self._lastOut@self._w-self._theta

    def adj(self, gradients:np.ndarray[Any, np.ndarray[Any, float]], lr:float=0.01):
        self._lastDeltaW=0.7*self._lastDeltaW+lr*(np.sum(self._lastOut*gradients, axis=0).transpose()/gradients.shape[0]).reshape(self._w.shape)
        self._lastDeltaTheta=0.7*self._lastDeltaTheta-lr*np.sum(gradients)/gradients.shape[0]
        self._w+=self._lastDeltaW
        self._theta+=self._lastDeltaTheta

    
