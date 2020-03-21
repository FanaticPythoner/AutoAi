from abc import ABCMeta, abstractmethod

class ICustomWrapper:
    '''
        Interface for custom AIModel Wrapper
    '''
    __metaclass__ = ABCMeta

    @abstractmethod
    def fit(self, X, y):
        '''
            Train the model on a given X and Y dataset

            RETURNS -> Void
        '''
        raise NotImplementedError

    @abstractmethod
    def predict(self, X):
        '''
            Predict X values

            RETURNS -> np.Array(Float) : Predictions
        '''
        raise NotImplementedError