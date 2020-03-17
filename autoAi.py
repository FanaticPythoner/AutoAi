#Copyright https://github.com/FanaticPythoner 2020

from sklearn.model_selection import train_test_split
from sklearn import metrics
import pickle as cPickle
import os
from glob import glob
from random import shuffle
import unicodedata
import string
import re
from colorama import init, Fore, Back, Style
import sys
import numpy as np

class AutoTrainer():
    '''
        Class that contains all supported auto-training models

        For documentation, refer to https://github.com/FanaticPythoner/pyAiTrainer
    '''
    def getModelsTypes(self):
        '''
            Return all supported autotrain models

            RETURNS -> List : models
        '''
        import sklearn.ensemble
        import sklearn.linear_model
        return [
            (sklearn.ensemble.RandomForestClassifier, {}),
            (sklearn.ensemble.RandomForestRegressor, {}),
            (sklearn.ensemble.GradientBoostingClassifier, {}),
            (sklearn.ensemble.AdaBoostClassifier, {}),
            (sklearn.ensemble.AdaBoostRegressor, {}),
            (sklearn.ensemble.ExtraTreesClassifier, {}),
            (sklearn.ensemble.GradientBoostingRegressor, {}),
            (sklearn.ensemble.BaggingClassifier, {}),
            (sklearn.ensemble.BaggingRegressor, {}),
            (sklearn.ensemble.ExtraTreesRegressor, {}),
            (sklearn.ensemble.IsolationForest, {}),
            (sklearn.ensemble.StackingClassifier, {"estimators" : 150}),
            (sklearn.ensemble.StackingRegressor, {"estimators" : 150}),
            (sklearn.ensemble.VotingClassifier, {"estimators" : 150}),
            (sklearn.ensemble.VotingRegressor, {"estimators" : 150}),

            (sklearn.linear_model.LogisticRegression, {}),
            (sklearn.linear_model.LinearRegression, {}),
            (sklearn.linear_model.Ridge, {}),
            (sklearn.linear_model.RidgeCV, {}),
            (sklearn.linear_model.ARDRegression, {}),
            (sklearn.linear_model.BayesianRidge, {}),
            (sklearn.linear_model.ElasticNet, {}),
            (sklearn.linear_model.ElasticNetCV, {}),
            (sklearn.linear_model.Hinge, {}),
            (sklearn.linear_model.HuberRegressor, {}),
            (sklearn.linear_model.Lars, {}),
            (sklearn.linear_model.LarsCV, {}),
            (sklearn.linear_model.Lasso, {}),
            (sklearn.linear_model.LassoCV, {}),
            (sklearn.linear_model.LassoLars, {}),
            (sklearn.linear_model.LassoLarsCV, {}),
            (sklearn.linear_model.LassoLarsIC, {}),
            (sklearn.linear_model.LogisticRegressionCV, {}),
            (sklearn.linear_model.ModifiedHuber, {}),
            (sklearn.linear_model.MultiTaskElasticNet, {}),
            (sklearn.linear_model.MultiTaskElasticNetCV, {}),
            (sklearn.linear_model.MultiTaskLasso, {}),
            (sklearn.linear_model.MultiTaskLassoCV, {}),
            (sklearn.linear_model.OrthogonalMatchingPursuit, {}),
            (sklearn.linear_model.OrthogonalMatchingPursuitCV, {}),
            (sklearn.linear_model.PassiveAggressiveClassifier, {}),
            (sklearn.linear_model.PassiveAggressiveRegressor, {}),
            (sklearn.linear_model.Perceptron, {}),
            (sklearn.linear_model.RANSACRegressor, {}),
            (sklearn.linear_model.RidgeClassifier, {}),
            (sklearn.linear_model.RidgeClassifierCV, {}),
            (sklearn.linear_model.SGDClassifier, {}),
            (sklearn.linear_model.SGDRegressor, {}),
            (sklearn.linear_model.TheilSenRegressor, {})
        ]

class AIModel():
    '''
        Class that allows machine learning / neural network model handling with
        autotraining support.

        For documentation, refer to https://github.com/FanaticPythoner/pyAiTrainer
    '''
    def __init__(self, modelName, baseDumpPath="Output_Models", autoTrainer=False, autoTrainerInstance=None):
        super().__init__()
        self._validFilenameChars = "-_.() %s%s" % (string.ascii_letters, string.digits)
        self.x = None
        self.y = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.model = None
        self.modelType = None
        self.isClassifier = True
        self.modelName = self._slugify(modelName)
        self.baseDumpPath = os.path.join(baseDumpPath, self.modelName)
        self._validateOrCreateFilePath(self.baseDumpPath)
        self.autoTrainer = autoTrainer
        self.autoTrainerInstance = None

        if autoTrainer:
            self.enableAutoTrainer(autoTrainerInstance)


    def enableAutoTrainer(self, autoTrainerInstance=None):
        '''
            Enable auto training

            RETURNS -> Void
        '''
        self.autoTrainer = True

        funcModelTypes = getattr(autoTrainerInstance, "getModelsTypes", None)

        if autoTrainerInstance is not None and callable(funcModelTypes):
            self.autoTrainerInstance = autoTrainerInstance
        elif autoTrainerInstance is None:
            self.autoTrainerInstance = AutoTrainer()
        elif autoTrainerInstance is not None and not callable(funcModelTypes):
            raise Exception("Invalid auto trainer instance type.")


    def _validateOrCreateFilePath(self, path):
        '''
            Validate a given directory path and create it if it doesnt exist
        '''
        if not os.path.exists(path):
            try:
                os.makedirs(path)
            except Exception as e:
                raise Exception("The specified dump path does not exist or is invalid.")


    def updateModel(self, model):
        '''
            Set the current trainer as a SVR model
            
            RETURNS -> Void
        '''
        self.model = model
        self.modelType = self.model.__class__.__name__


    def updateDataSet(self, x, y, test_size=0.2):
        '''
            Update the current dataset and train / test split

            RETURNS -> Void
        '''
        if self.model is None and not self.autoTrainer and not self.autoTrainerInstance:
            raise Exception("Cannot update dataset: The current model is not defined.")

        self.x = x
        self.y = y
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x.values, self.y.values.ravel(), test_size = test_size)


    def getAccuracy(self):
        '''
            Get the current model accuracy score

            RETURNS -> Float : Accuracy
        '''
        y_pred = self.model.predict(self.x_test)
        try:
            accuracyScore = metrics.accuracy_score(self.y_test, y_pred)
            self.isClassifier = True
        except ValueError as e:
            n = y_pred.shape[0]
            p = self.x_test.shape[1]
            accuracyScore = 1-(1-metrics.r2_score(self.y_test, y_pred))*(n-1)/(n-p-1)
            self.isClassifier = False
        return accuracyScore


    def _printAccuracy(self, iterNum):
        '''
            Print the current accuracy / loss

            RETURNS -> Void
        '''
        init(convert=True)
        toPrintString = "\n[" + Fore.BLUE + 'ITER' + Fore.RESET + "] " + str(iterNum) + " iterations. Type \"" + self.modelType + "\"."
        accuracyScore = self.getAccuracy()
        if self.isClassifier:
            toPrintString += " Accuracy: " + Style.BRIGHT + Fore.BLUE + str(accuracyScore) + Fore.RESET + Style.NORMAL
        else:
            toPrintString += " Adjusted R2: " + Style.BRIGHT + Fore.BLUE + str(accuracyScore) + Fore.RESET + Style.NORMAL
        print(toPrintString)


    def getCompatibleAutotrainerModels(self):
        '''
            Get all compatible models for the current loaded data

            RETURNS -> List : Compatible Transformed Models Instances
        '''
        models = self.autoTrainerInstance.getModelsTypes()
        modelsLen = len(models)
        compModelsInstances = []

        oldModel = self.model

        for model, args in models:
            instance = model(**args)
            try:
                sys.stdout = open(os.devnull, "w")
                sys.stderr = open(os.devnull, "w")
                instance.fit(self.x_train[0:1], self.y_train[0:1])
                sys.stdout = sys.__stdout__
                sys.stderr = sys.__stderr__
                compModelsInstances.append(instance)
            except Exception as e:
                pass

        incompModelsCount = modelsLen - len(compModelsInstances)
        return (compModelsInstances, incompModelsCount, modelsLen)


    def train(self, max_iter=100, batchSize=100, dumpEachIter=50, verboseLevel=1):
        '''
            Train the current model with dataset

            RETURNS -> Void
        '''
        compModelsInstances = []

        if self.autoTrainer:
            compModelsInstances, incompCount, totalCount =  self.getCompatibleAutotrainerModels()
            compModelsInstances = [x for x in compModelsInstances if type(x) != type(self.model)]

            if self.model is not None:
                compModelsInstances.append(self.model)
                
            if verboseLevel in (1, 2):
                init(convert=True)
                print("\n[" + Fore.MAGENTA + 'AUTOTRAINER' + Fore.RESET + "] Automatic trainer is activated.")
                if incompCount > 0:
                    print("\n[" + Fore.YELLOW + 'WARNING' + Fore.RESET + "] Found " + str(incompCount) + " of " + str(totalCount) + " incompatible models for the current dataset.")
        else:
            compModelsInstances = [self.model]

        for instance in compModelsInstances:
            self.updateModel(instance)

            if verboseLevel == 2:
                init(convert=True)
                if self.autoTrainer:
                    print("\n[" + Fore.MAGENTA + 'AUTOTRAINER' + Fore.RESET + "] Loaded model type \"" + Fore.MAGENTA + instance.__class__.__name__ + Fore.RESET + "\".")
                else:
                    print("\n[" + Fore.MAGENTA + 'TRAINER' + Fore.RESET + "] Loaded model type \"" + Fore.MAGENTA + instance.__class__.__name__ + Fore.RESET + "\".")

            if self.model is None:
                raise Exception("Cannot train model: The current model is not defined.")

            if (self.x is None or 
                self.y is None or self.x_train is None or
                self.y_train is None or self.x_test is None or
                self.y_test is None or self.modelType is None):
                raise Exception("Cannot train model: The current dataset is not defined.")

            self.model.fit(self.x_train[0:batchSize], self.y_train[0:batchSize])


            batchSizeModifier = 0
            batchSizeEndModifier = 0
            iIter = 1
            xTrainLen = len(self.x_train)

            if batchSize > xTrainLen:
                batchSize = xTrainLen

            while iIter < max_iter:
                i = batchSize
                batchSizeEndModifier = 0
                batchSizeModifier = 0
                reachedEnd = False

                randomize = np.arange(len(self.x_train))
                np.random.shuffle(randomize)
                self.x_train = self.x_train[randomize]
                self.y_train = self.y_train[randomize]

                while i < xTrainLen:
                    x_trainTemp = self.x_train[i - batchSize + batchSizeModifier: i + batchSizeEndModifier]
                    y_trainTemp = self.y_train[i - batchSize + batchSizeModifier: i + batchSizeEndModifier]
                    self.model.fit(x_trainTemp, y_trainTemp)
                    
                    if iIter % dumpEachIter == 0:
                        if verboseLevel in (1, 2):
                            self._printAccuracy(iIter)
                        self.dumpModel(verboseLevel=verboseLevel)

                        if iIter == max_iter:
                            break
    
                    if iIter > max_iter:
                        break
    
                    if reachedEnd:
                        break
                    
                    iIter += 1
                    oldI = i
                    i += batchSize
                    if i > xTrainLen:
                        reachedEnd = True
                        i = oldI
                        batchSizeModifier = batchSize
                        batchSizeEndModifier = (xTrainLen - i)
                    else:
                        batchSizeModifier = 0
                        batchSizeEndModifier = 0

        if iIter % dumpEachIter != 0:
            if verboseLevel in (1, 2):
                self._printAccuracy(iIter)

            self.dumpModel(verboseLevel=verboseLevel)


    def predict(self, x_vals):
        '''
            Get the prediction X from the current model

            RETURNS -> Float : Predictions
        '''
        return self.model.predict(x_vals)


    def _slugify(self, fileName):
        '''
            Get the valid file name form of a AIModel name

            RETURNS -> Str : AIModel Name
        '''
        cleanedFilename = unicodedata.normalize('NFKD', fileName).encode('ASCII', 'ignore')     
        return re.sub("\s\s+" , " ", ''.join(chr(c) for c in cleanedFilename if chr(c) in self._validFilenameChars))


    def dumpModel(self, basePath=None, verboseLevel=1):
        '''
            Save the current model to path

            RETURNS -> Void
        '''

        if basePath is None:
            basePath = self.baseDumpPath

        self._validateOrCreateFilePath(basePath)

        writePath = os.path.join(basePath, self.modelType)
        accuracy = self.getAccuracy()

        if self.isClassifier:
            writePath += "_Classifier_"
        else:
            writePath += "_Regressor_"

        writePath += str(accuracy) + ".pymodel"

        os.makedirs(os.path.dirname(writePath), exist_ok=True)

        pickle_out = open(writePath, 'wb')
        cPickle.dump(self.model, pickle_out)
        pickle_out.close()

        if verboseLevel == 2:
            init(convert=True)
            writePath = "\n[" + Fore.GREEN + 'DUMP' + Fore.RESET + "] Dumping model done. " + os.path.abspath(writePath)
            print(writePath)


    def loadBestModel(self, basePath=None, raiseIfDontExist=True):
        '''
            Load the best model for the given model type

            RETURNS -> Void
        '''
        if basePath is None:
            basePath = os.path.join(self.baseDumpPath)

        self._validateOrCreateFilePath(basePath)

        accuracyList = []

        for dirCategory in glob(basePath):
            if dirCategory.split(os.path.sep)[-1] == self.modelName:
                for file in glob(os.path.join(dirCategory, "*.*")):
                    fileSplitted = file.split(".pymodel")[0].split(os.path.sep)[-1].split('_')
                    modelType = fileSplitted[-2]

                    if modelType == "Classifier":
                        self.isClassifier = True
                    elif modelType == "Regressor":
                        self.isClassifier = False
                    accuracy = fileSplitted[-1]
                    modelType = fileSplitted[0]
                    accuracyList.append((file, accuracy, self.isClassifier, modelType))

        if len(accuracyList) == 0:
            if raiseIfDontExist:
                raise Exception("No model available.")
            else:
                print("\nNo model available.")
                return


        init(convert=True)
        printStr = "\n[" + Fore.GREEN + 'LOAD' + Fore.RESET + "] Loaded best model for \"" + self.modelName + "\"."
        accuracyList = sorted(accuracyList, key=lambda x: float(x[1]), reverse=True)
        self.updateModel(cPickle.load(open(accuracyList[0][0], 'rb')))
        printStr += " Type \"" + accuracyList[0][3] + "\""

        if accuracyList[0][2] == True:
            printStr += " Category \"Classifier\". Accuracy: " + Style.BRIGHT + Fore.GREEN + str(accuracyList[0][1]) + Fore.RESET + Style.NORMAL
        else:
            printStr += " Category \"Regressor\". Adjusted R2: " + Style.BRIGHT + Fore.GREEN + str(accuracyList[0][1]) + Fore.RESET + Style.NORMAL

        print(printStr)
