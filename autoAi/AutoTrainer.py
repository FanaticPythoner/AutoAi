import sklearn.ensemble
import sklearn.linear_model
import sklearn.naive_bayes
import sklearn.svm

# DynamicKerasWrapper
from keras.layers import Dense
from keras.models import Sequential
from math import ceil
import numpy as np
from sklearn import metrics
from keras.optimizers import Adam
from autoAi.Interfaces import ICustomWrapper

class AutoTrainer():
    '''
        Class that contains all supported auto-training models

        For documentation, refer to https://github.com/FanaticPythoner/AutoAi
    '''

    def getModelsTypes(self):
        '''
            Return all supported autotrain models instances

            RETURNS -> List : models
        '''
        return [
            DynamicKerasWrapper(),
            sklearn.ensemble.RandomForestClassifier(),
            sklearn.ensemble.RandomForestRegressor(),
            sklearn.ensemble.GradientBoostingClassifier(),
            sklearn.ensemble.AdaBoostClassifier(),
            sklearn.ensemble.AdaBoostRegressor(),
            sklearn.ensemble.ExtraTreesClassifier(),
            sklearn.ensemble.GradientBoostingRegressor(),
            sklearn.ensemble.BaggingClassifier(),
            sklearn.ensemble.BaggingRegressor(),
            sklearn.ensemble.ExtraTreesRegressor(),
            sklearn.ensemble.IsolationForest(),
            sklearn.ensemble.StackingClassifier(estimators=[('lr', sklearn.linear_model.LogisticRegression(multi_class='multinomial', dual=False, max_iter=1200000)),
                                                            ('rf', sklearn.ensemble.RandomForestClassifier(n_estimators=50)),
                                                            ('gnb', sklearn.naive_bayes.GaussianNB())],
                                                final_estimator=sklearn.ensemble.RandomForestClassifier(n_estimators=50)),

            sklearn.ensemble.StackingRegressor(estimators=[('lr', sklearn.linear_model.RidgeCV()),
                                                           ('rf', sklearn.svm.LinearSVR())],
                                                final_estimator=sklearn.ensemble.RandomForestRegressor(n_estimators=50)),

            sklearn.ensemble.VotingClassifier(estimators=[('lr', sklearn.linear_model.LogisticRegression(multi_class='multinomial', dual=False, max_iter=1200000)),
                                                          ('rf', sklearn.ensemble.RandomForestClassifier(n_estimators=50)),
                                                          ('gnb', sklearn.naive_bayes.GaussianNB())]),

            sklearn.ensemble.VotingRegressor(estimators=[('lr', sklearn.linear_model.LinearRegression()),
                                                         ('rf', sklearn.ensemble.RandomForestRegressor(n_estimators=50))]),


            sklearn.linear_model.LogisticRegression(),
            sklearn.linear_model.LinearRegression(),
            sklearn.linear_model.Ridge(),
            sklearn.linear_model.RidgeCV(),
            sklearn.linear_model.ARDRegression(),
            sklearn.linear_model.BayesianRidge(),
            sklearn.linear_model.HuberRegressor(),
            sklearn.linear_model.OrthogonalMatchingPursuit(),
            sklearn.linear_model.PassiveAggressiveClassifier(),
            sklearn.linear_model.PassiveAggressiveRegressor(),
            sklearn.linear_model.Perceptron(),
            sklearn.linear_model.RidgeClassifier(),
            sklearn.linear_model.SGDClassifier(),
            sklearn.linear_model.SGDRegressor(),
            sklearn.linear_model.TheilSenRegressor()
        ]


class DynamicKerasWrapper(ICustomWrapper):
    '''
        Class that dynamically create a keras model during fitting
    '''
    def __init__(self, scalingFactor=3, learningRate=0.05):
        self.scalingFactor = scalingFactor
        self.isSet = False
        self.learningRate = learningRate


    def preFit(self, X, y):
        '''
            Create a dynamic Keras model

            RETURNS -> Void
        '''
        if not self.isSet:
            input_dim = self._getInputDim(X)
            output_dim = self._getOutputDim(y)
            neuronsCount = ceil(X.shape[0] / (self.scalingFactor * (input_dim + output_dim)))
            loss, metric, output_dim = self._getLoss(X, y)

            self._createModel(input_dim=input_dim, output_dim=output_dim,
                              nodes=neuronsCount, layersCount=2, loss=loss,
                              metric=metric)
            self.isSet = True


    def fit(self, X, y):
        '''
            Train the model on a given X and Y dataset

            RETURNS -> Void
        '''
        self.model.fit(X, y, verbose=0)


    def predict(self, X):
        '''
            Predict X values

            RETURNS -> np.Array(Float) : Predictions
        '''
        return self.model.predict(X)


    def _getOutputDim(self, y):
        '''
            Get the number of input dim from dataset

            RETURNS -> Int
        '''
        if len(y.shape) == 1:
            return 1
        else:
            return y.shape[1]


    def _getInputDim(self, x):
        '''
            Get the number of input dim from dataset

            RETURNS -> Int
        '''
        if len(x.shape) == 1:
            return 1
        else:
            return x.shape[1]


    def _createModel(self, input_dim, output_dim, nodes, layersCount, loss, metric):
        '''
            Create the keras model with the specified parameters

            RETURNS -> Void
        '''
        self.model = Sequential()
        self.model.add(Dense(nodes, input_dim=input_dim, activation='relu'))

        for _ in range(layersCount - 1):
            self.model.add(Dense(nodes, activation='relu'))
        self.model.add(Dense(output_dim, activation='sigmoid'))

        self.model.compile(loss=loss, 
                      optimizer=Adam(learning_rate=self.learningRate), 
                      metrics=[metric])


    def _getLoss(self, X, y):
        '''
            Get the optimal loss function for a given
            Y dataset

            RETURNS -> (Str : Loss, Str : Metric, Int : Output_Dim)
        '''

        def isClassifier(X, y):
            try:
                _ = metrics.accuracy_score(y, y)
                return True
            except ValueError as e:
                n = y.shape[0]
                p = X.shape[1]
                _ = 1-(1-metrics.r2_score(y, y))*(n-1)/(n-p-1)
                return False

        def uniqueIndices(y):
            seen = set()
            res = []
            for i, n in enumerate(y):
                if n not in seen:
                    res.append(i)
                    seen.add(n)
            return len(res)

        def isAllInteger(y):
            return np.equal(np.mod(y, 1), 0)

        isClassifier = isClassifier(X, y)
        output_dim = self._getOutputDim(y)
        maxVal = np.max(y)
        minVal = np.min(y)

        if output_dim == 1 and maxVal <= 1 and minVal >= 0:
            return ('binary_crossentropy', 'accuracy', output_dim)

        if output_dim == 1 and (maxVal >= 1 or minVal <= 0):
            return ('mean_squared_error', 'mean_squared_error', output_dim)

        isAllInt = isAllInteger(y).all()

        if isAllInt and minVal == 0 and maxVal == 1:
            return ('categorical_crossentropy', 'mean_squared_error', output_dim)

        if isAllInt:
            return ('sparse_categorical_crossentropy', 'sparse_categorical_accuracy', uniqueIndices(y))

        if output_dim >= 1 and not isAllInt:
            return ('mean_squared_error', 'mean_squared_error', output_dim)

        raise Exception("Unsupported output type")