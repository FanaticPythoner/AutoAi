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
import warnings
from sklearn.exceptions import ConvergenceWarning
import shutil

import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler

from copy import deepcopy

from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import LassoCV

import hashlib

class AutoTrainer():
    '''
        Class that contains all supported auto-training models

        For documentation, refer to https://github.com/FanaticPythoner/AutoAi
    '''
    def getModelsTypes(self):
        '''
            Return all supported autotrain models

            RETURNS -> List : models
        '''
        import sklearn.ensemble
        import sklearn.linear_model
        import sklearn.naive_bayes
        import sklearn.svm
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
            (sklearn.ensemble.StackingClassifier, {"estimators" : [
                                                                      ('lr', sklearn.linear_model.LogisticRegression(multi_class='multinomial', dual=False, max_iter=1200000)),
                                                                      ('rf', sklearn.ensemble.RandomForestClassifier(n_estimators=50)),
                                                                      ('gnb', sklearn.naive_bayes.GaussianNB())
                                                                  ],
                                                  "final_estimator" : sklearn.ensemble.RandomForestClassifier(n_estimators=50)}),
            (sklearn.ensemble.StackingRegressor, {"estimators" : [
                                                                     ('lr', sklearn.linear_model.RidgeCV()),
                                                                     ('rf', sklearn.svm.LinearSVR())
                                                                 ],
                                                  "final_estimator" : sklearn.ensemble.RandomForestRegressor(n_estimators=50)}),
            (sklearn.ensemble.VotingClassifier, {"estimators" : [
                                                                    ('lr', sklearn.linear_model.LogisticRegression(multi_class='multinomial', dual=False, max_iter=1200000)),
                                                                    ('rf', sklearn.ensemble.RandomForestClassifier(n_estimators=50)),
                                                                    ('gnb', sklearn.naive_bayes.GaussianNB())
                                                                ]}),
            (sklearn.ensemble.VotingRegressor, {"estimators" : [
                                                                   ('lr', sklearn.linear_model.LinearRegression()),
                                                                   ('rf', sklearn.ensemble.RandomForestRegressor(n_estimators=50))
                                                               ]}),

            (sklearn.linear_model.LogisticRegression, {}),
            (sklearn.linear_model.LinearRegression, {}),
            (sklearn.linear_model.Ridge, {}),
            (sklearn.linear_model.RidgeCV, {}),
            (sklearn.linear_model.ARDRegression, {}),
            (sklearn.linear_model.BayesianRidge, {}),
            (sklearn.linear_model.HuberRegressor, {}),
            (sklearn.linear_model.OrthogonalMatchingPursuit, {}),
            (sklearn.linear_model.PassiveAggressiveClassifier, {}),
            (sklearn.linear_model.PassiveAggressiveRegressor, {}),
            (sklearn.linear_model.Perceptron, {}),
            (sklearn.linear_model.RidgeClassifier, {}),
            (sklearn.linear_model.SGDClassifier, {}),
            (sklearn.linear_model.SGDRegressor, {}),
            (sklearn.linear_model.TheilSenRegressor, {})
        ]

class AIModel():
    '''
        Class that allows machine learning / neural network model handling with
        autotraining support.

        For documentation, refer to https://github.com/FanaticPythoner/AutoAi

        Parameters:

            1. modelName           : Name of the AiModel project
            2. baseDumpPath        : Base path to dump the models during training
            3. autoTrainer         : Use the automatic trainer or not
            4. autoTrainerInstance : Instance of custom auto trainer class if specified, otherwise use the default
                                     AutoTrainer.
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


    def _validateKeepBestModelOnly(self, keepBestModelOnly):
        '''
            Validate a keepBestModelOnly

            RETURNS -> Void
        '''
        if type(keepBestModelOnly) is not bool:
            raise Exception("Invalid keepBestModelOnly: must be a bool")


    def enableAutoTrainer(self, autoTrainerInstance=None):
        '''
            Enable auto training

            Parameters:

                1. autoTrainerInstance : Instance of custom auto trainer class if specified, otherwise use the default
                                         AutoTrainer.

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
            Set the current trainer model
            
            Parameters:

                1. model : Instance of model that the current AiModel will use

            RETURNS -> Void
        '''
        self.model = model
        self.modelType = self.model.__class__.__name__


    def updateDataSet(self, x, y, test_size=0.2):
        '''
            Update the current dataset and train / test split

            Parameters:

                1. x         : Dataset independent variables
                2. y         : Dataset dependent variables
                3. test_size : Size of the dataset to use for testing (0.2 = 20%, etc.)

            RETURNS -> Void
        '''
        if self.model is None and not self.autoTrainer and not self.autoTrainerInstance:
            raise Exception("Cannot update dataset: The current model is not defined.")

        self.x = x
        self.y = y
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x.values, self.y.values.ravel(), test_size = test_size)


    def _clearWorseModels(self):
        '''
            Clear all models except the best one.

            RETURNS -> Void
        '''
        highest = 0
        fileNameExclude = ""
        for filename in os.listdir(self.baseDumpPath):
            file_path = os.path.join(self.baseDumpPath, filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                accuracy = float('.'.join(filename.split(".")[:-1]).split("_")[-1])
                if accuracy > highest:
                    highest = accuracy
                    fileNameExclude = filename
                    
        for filename in os.listdir(self.baseDumpPath):
            file_path = os.path.join(self.baseDumpPath, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    if filename != fileNameExclude:
                        os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))

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
        def uniqueIndices(list):
            seen = set()
            res = []
            for i, n in enumerate(list):
                if n not in seen:
                    res.append(i)
                    seen.add(n)
            return res

        models = self.autoTrainerInstance.getModelsTypes()
        modelsLen = len(models)
        compModelsInstances = []

        uniqueIndicesLst = uniqueIndices(self.y_train)
        xTrainMinimal = self.x_train[uniqueIndicesLst]
        yTrainMinimal = self.y_train[uniqueIndicesLst]

        for model, args in models:
            tmpXTrain = xTrainMinimal
            tmpYTrain = yTrainMinimal
            if model.__name__ in ('StackingClassifier', 'StackingRegressor', 'VotingClassifier', 'VotingRegressor'):
                tmpXTrain = self.x_train
                tmpYTrain = self.y_train
                
            instance = model(**args)
            try:
                sys.stdout = open(os.devnull, "w")
                sys.stderr = open(os.devnull, "w")
                instance.fit(tmpXTrain, tmpYTrain)
                sys.stdout = sys.__stdout__
                sys.stderr = sys.__stderr__
                compModelsInstances.append(instance)
            except Exception as e:
                pass

        incompModelsCount = modelsLen - len(compModelsInstances)
        return (compModelsInstances, incompModelsCount, modelsLen)


    def train(self, max_iter=100, batchSize=100, dumpEachIter=50, verboseLevel=1, keepBestModelOnly=False):
        '''
            Train the current model with dataset

            Parameters:

                1. max_iter          : Maximum training iteration to reach
                2. batchSize         : Size of every batch to fit during training
                3. dumpEachIter      : Maximum iteration to reach before creating a model file (.pymodel)
                4. verboseLevel      : Verbose level (0, 1 or 2)
                5. keepBestModelOnly : If True, keep only the most accurate model, delete the others after training

            RETURNS -> Void
        '''
        self._validateKeepBestModelOnly(keepBestModelOnly)

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

        iIter = 0
        lastDumpiIter = -1
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

            if self.model.__class__.__name__ in ('StackingClassifier', 'StackingRegressor', 'VotingClassifier', 'VotingRegressor'):
                randomize = np.arange(len(self.x_train))
                np.random.shuffle(randomize)
                x_trainTemp = self.x_train[randomize]
                y_trainTemp = self.y_train[randomize]
            else:
                x_trainTemp = self.x_train[0:batchSize]
                y_trainTemp = self.y_train[0:batchSize]
                
            seenOneConvergenceWarning = False

            if not seenOneConvergenceWarning:
                warnings.filterwarnings("error", category=ConvergenceWarning)
            else:
                warnings.filterwarnings("ignore", category=ConvergenceWarning)

            try:
                self.model.fit(x_trainTemp, y_trainTemp)
            except ConvergenceWarning as e:
                if not seenOneConvergenceWarning:
                    print(Fore.RED, e, Fore.RESET)
                    seenOneConvergenceWarning = True
                else:
                    pass

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
                    if self.model.__class__.__name__ in ('StackingClassifier', 'StackingRegressor', 'VotingClassifier', 'VotingRegressor'):
                        randomize = np.arange(len(self.x_train))
                        np.random.shuffle(randomize)
                        x_trainTemp = self.x_train[randomize]
                        y_trainTemp = self.y_train[randomize]
                    else:
                        x_trainTemp = self.x_train[i - batchSize + batchSizeModifier: i + batchSizeEndModifier]
                        y_trainTemp = self.y_train[i - batchSize + batchSizeModifier: i + batchSizeEndModifier]

                    if not seenOneConvergenceWarning:
                        warnings.filterwarnings("error", category=ConvergenceWarning)
                    else:
                        warnings.filterwarnings("ignore", category=ConvergenceWarning)

                    try:
                        self.model.fit(x_trainTemp, y_trainTemp)
                    except ConvergenceWarning as e:
                        if not seenOneConvergenceWarning:
                            print(Fore.RED, e, Fore.RESET)
                            seenOneConvergenceWarning = True
                        else:
                            pass
                    
                    if iIter % dumpEachIter == 0 and lastDumpiIter != iIter:
                        lastDumpiIter = iIter
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

        if iIter % dumpEachIter != 0 and lastDumpiIter != iIter:
            lastDumpiIter = iIter
            if verboseLevel in (1, 2):
                self._printAccuracy(iIter)

            self.dumpModel(verboseLevel=verboseLevel)

        if keepBestModelOnly:
            self._clearWorseModels()


    def predict(self, x_vals):
        '''
            Get the prediction X from the current model

            Parameters:

                1. x_vals : Independent variables used to predict a new dependent variable

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

            Parameters:

                1. basePath     : Path to dump the current loaded model
                2. verboseLevel : Verbose level (0, 1 or 2)

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


    def loadBestModel(self, basePath=None, raiseIfDontExist=True, verboseLevel=1):
        '''
            Load the best model for the given model type

            Parameters:

                1. basePath         : Path to load
                2. raiseIfDontExist : If no model are available, raise an exception or just ignore
                3. raiseIfDontExist : Verbose level (0, 1 or 2)

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

        if verboseLevel in (1,2):
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

class AutoPreprocessor():
    '''
        Class that allows automatic data preprocessing.

            Parameters:

                1. datasetPath      : Path of the dataset
                2. datasetType      : Type of the dataset ('csv'...)
                3. yDataIndices     : List of indices of the dependent variables in the dataset
                4. ignoreColIndices : List of indices of the columns to ignore in the dataset
                5. ignoreRowIndices : List of indices of the rows to ignore in the dataset

        For documentation, refer to https://github.com/FanaticPythoner/AutoAi
    '''
    
    def __init__(self, datasetPath, datasetType, yDataIndices, ignoreColIndices=[], ignoreRowIndices=[]):
        super().__init__()
        self.supportedNanHandlingMethods = ('delete', 'mean', 'median', 'mode', 'fixed', 'predict')
        self.supportedFeatureSelectionMethod = ('backward', 'recursive', 'embedded')
        self.supportedScaleDataType = ('robust', 'minmax', 'standard')
        self.supportedDatasetType = ('csv')
        self.datasetPath = None
        self.datasetType = None
        self.nanDataHandling = None
        self.nanFixedValue = None
        self.addIsNanFeature = True
        self.dataset = None
        self.allIndices = []
        self.yIndices = []
        self.xIndices = []
        self.ordinalIndices = []
        self.allRowsIndices = []
        self.categoricalIndices = []
        self.scaleDataType = None
        self.scalers = None

        self.featureSelectionMethod = None
        self.predictAutoTrainer = None
        self.ohEncoder = OneHotEncoder()
        self.labelEncoder = LabelEncoder()
        self.updateDataset(datasetPath, datasetType, yDataIndices, ignoreColIndices, ignoreRowIndices)

        self.categoricalIndicesNames = []
        self.ordinalIndicesNames = []
        self.xIndicesNames = []
        self.yIndicesNames = []

        self.predictMaxIter = None
        self.predictBatchSize = None
        self.predictDumpEachIter = None
        self.predictTestSize = None
        self.predictVerboseLevel = None
        self.predictKeepBestOnly = None


    def _validateFeatureSelectionMethod(self, featureSelectionMethod=None):
        '''
            Validate the current feature selection method

            RETURNS -> Void
        '''
        if featureSelectionMethod is None:
            featureSelectionMethod = self.featureSelectionMethod

        if featureSelectionMethod is not None and featureSelectionMethod not in self.supportedFeatureSelectionMethod:
            raise Exception("Unsupported featureSelectionMethod")


    def updateFeatureSelectionMethod(self, featureSelectionMethod):
        '''
            Update the AutoProcessor feature selection method

            Supported methods for featureSelectionMethod:

            1. 'backward'  : Backward Elimination feature selection
            2. 'recursive' : Recursive Elimination feature selection
            3. 'embedded'  : Embedded Elimination feature selection

            RETURNS -> Void
        '''
        self._validateDataset()
        self._validateFeatureSelectionMethod(featureSelectionMethod)
        self.featureSelectionMethod = featureSelectionMethod


    def _embeddedElimination(self):
        '''
            Perform automatic emedded elimination on independent variables

            RETURNS -> Void
        '''

        nof_listOriginal = np.arange(0,self.dataset.iloc[:, self.xIndices].values.shape[1])
        nof_list = []
        embeddedModel = LassoCV()
        embeddedModel.fit(self.dataset.iloc[:, self.xIndices].values, self.dataset.iloc[:, self.yIndices].values.ravel())
        coef = pd.Series(embeddedModel.coef_, index = self.dataset.iloc[:, self.xIndices].columns).values

        for i, val in enumerate(coef):
            if val != 0:
                nof_list.append(i)

        nof_list = np.array(nof_list)

        self.xIndicesNames = [self.dataset.iloc[:, self.xIndices].columns[x] for x in nof_list]
        self.xIndices = [self.dataset.columns.get_loc(val) for val in self.xIndicesNames]
        self.categoricalIndicesNames = [val for val in self.categoricalIndicesNames if val in self.xIndicesNames or val in self.yIndicesNames]
        self.categoricalIndices = [self.dataset.columns.get_loc(val) for val in self.categoricalIndicesNames]
        self.ordinalIndicesNames = [val for val in self.ordinalIndicesNames if val in self.xIndicesNames or val in self.yIndicesNames]
        self.ordinalIndices = [self.dataset.columns.get_loc(val) for val in self.ordinalIndicesNames]

        x = self.dataset.iloc[:, self.xIndices]

        if (nof_listOriginal == nof_list).all():
            return

        deletedList = np.unique(nof_listOriginal - nof_list)

        for j in deletedList:
            deletedXVal = x.columns[self.xIndices[j]]
            self.dataset.drop(deletedXVal, axis=1, inplace=True)
            self.xIndices = [self.dataset.columns.get_loc(val) for val in self.xIndicesNames if val != deletedXVal]
            self.categoricalIndices = [self.dataset.columns.get_loc(val) for val in self.categoricalIndicesNames if val != deletedXVal]
            self.ordinalIndices = [self.dataset.columns.get_loc(val) for val in self.ordinalIndicesNames if val != deletedXVal]

        self.yIndices = [self.dataset.columns.get_loc(val) for val in self.yIndicesNames]


    def _recursiveElimination(self):
        '''
            Perform automatic recursive elimination (RFE) on independent variables

            RETURNS -> Void
        '''

        nof_listOriginal = np.arange(1,self.dataset.iloc[:, self.xIndices].values.shape[1])
        nof_list = deepcopy(nof_listOriginal)
        nof = 0           
        high_score = 0
        for n in range(len(nof_list)):
            X_train, X_test, y_train, y_test = train_test_split(self.dataset.iloc[:, self.xIndices].values,self.dataset.iloc[:, self.yIndices].values, test_size = 0.2, random_state = 42)
            model = LinearRegression()
            rfe = RFE(model,nof_list[n])
            X_train_rfe = rfe.fit_transform(X_train,y_train.ravel())
            X_test_rfe = rfe.transform(X_test)
            model.fit(X_train_rfe,y_train)
            score = model.score(X_test_rfe,y_test)
            if(score>high_score):
                high_score = score
                nof = nof_list[n]


        self.xIndicesNames = [self.dataset.iloc[:, self.xIndices].columns[x] for x in range(nof)]
        self.xIndices = [self.dataset.columns.get_loc(val) for val in self.xIndicesNames]
        self.categoricalIndicesNames = [val for val in self.categoricalIndicesNames if val in self.xIndicesNames or val in self.yIndicesNames]
        self.categoricalIndices = [self.dataset.columns.get_loc(val) for val in self.categoricalIndicesNames]
        self.ordinalIndicesNames = [val for val in self.ordinalIndicesNames if val in self.xIndicesNames or val in self.yIndicesNames]
        self.ordinalIndices = [self.dataset.columns.get_loc(val) for val in self.ordinalIndicesNames]

        x = self.dataset.iloc[:, self.xIndices]

        if (nof_listOriginal == nof_list).all():
            return    

        deletedList = np.unique(nof_listOriginal - nof_list)

        for j in deletedList:
            deletedXVal = x.columns[self.xIndices[j]]
            self.dataset.drop(deletedXVal, axis=1, inplace=True)
            self.xIndices = [self.dataset.columns.get_loc(val) for val in self.xIndicesNames if val != deletedXVal]
            self.categoricalIndices = [self.dataset.columns.get_loc(val) for val in self.categoricalIndicesNames if val != deletedXVal]
            self.ordinalIndices = [self.dataset.columns.get_loc(val) for val in self.ordinalIndicesNames if val != deletedXVal]

        self.yIndices = [self.dataset.columns.get_loc(val) for val in self.yIndicesNames]


    def _backwardElimination(self, SL=0.05):
        '''
            Perform automatic backward elimination on independent variables
            with a given treshhold (SL)

            RETURNS -> Void
        '''

        datasetCopy = deepcopy(self.dataset)
        x = datasetCopy.iloc[:, self.xIndices]
        y = datasetCopy.iloc[:, self.yIndices]

        newXIndicesNames = self.xIndicesNames
        newXIndices = self.xIndices
        newYIndices = self.yIndices
        newCatIndicesNames = self.categoricalIndicesNames
        newOrdIndicesNames = self.ordinalIndicesNames

        numVars = len(x.values[0])
        tempNpArr = np.zeros((x.values.shape[0],x.values.shape[1])).astype(int)
        for i in range(0, numVars):
            regressor_OLS = sm.OLS(y.values, datasetCopy.iloc[:, newXIndices].values).fit()
            maxVar = max(regressor_OLS.pvalues).astype(float)
            adjR_before = regressor_OLS.rsquared_adj.astype(float)
            if maxVar > SL:
                for j in range(0, numVars - i):
                    if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                        tempNpArr[:,j] = x.values[:, j]

                        deletedXVal = x.columns[newXIndices[j]]
                        newXIndicesNames = [val for val in newXIndicesNames if val != deletedXVal]
                        newCatIndicesNames = [val for val in newCatIndicesNames if val != deletedXVal]
                        newOrdIndicesNames = [val for val in newOrdIndicesNames if val != deletedXVal]

                        datasetCopy.drop(deletedXVal, axis=1, inplace=True)
                        newXIndices = [datasetCopy.columns.get_loc(val) for val in self.xIndicesNames if val != deletedXVal]
                        newYIndices = [datasetCopy.columns.get_loc(val) for val in self.yIndicesNames if val != deletedXVal]
                        x = datasetCopy.iloc[:, newXIndices]
                        y = datasetCopy.iloc[:, newYIndices]

                        tmp_regressor = sm.OLS(y.values, x.values).fit()
                        adjR_after = tmp_regressor.rsquared_adj.astype(float)
                        if (adjR_before >= adjR_after):
                            return
                        else:
                            continue


        self.dataset = deepcopy(datasetCopy)
        self.xIndices = newXIndices
        self.yIndices = newYIndices
        self.xIndicesNames = newXIndicesNames
        self.categoricalIndicesNames = newCatIndicesNames
        self.categoricalIndices =  [datasetCopy.columns.get_loc(val) for val in self.categoricalIndicesNames]
        self.ordinalIndicesNames = newOrdIndicesNames
        self.ordinalIndices = [datasetCopy.columns.get_loc(val) for val in self.ordinalIndicesNames]
        self.allIndices = [x for x in range(datasetCopy.values.shape[1]) if x in self.xIndices or x in self.yIndices]
 

    def _validateDataset(self):
        '''
            Validate the current dataset

            RETURNS -> Void
        '''
        if self.dataset is None or type(self.dataset) is not pd.DataFrame:
            raise Exception("Invalid dataset")

        if self.datasetType not in self.supportedDatasetType:
            raise Exception("Unsupported dataset type")

        if type(self.datasetPath) is not str:
            raise Exception("Invalid datasetPath type: must be \"str\"")

        if type(self.allIndices) is not list:
            raise Exception("Invalid allIndices type : must type \"list\"")

        if type(self.yIndices) is not list:
            raise Exception("Invalid yIndices type : must type \"list\"")

        if type(self.xIndices) is not list:
            raise Exception("Invalid xIndices type : must type \"list\"")

        if len([x for x in self.allIndices if x in [y for y in range(self.dataset.iloc[:, :].values.shape[1])]]) != len(self.allIndices):
            raise Exception("Invalid allIndices : all allIndices must be present in the dataset")

        if len([x for x in self.xIndices if x in self.allIndices]) != len(self.xIndices):
            raise Exception("Invalid xIndices : all xIndices must be present in allIndices")

        if len([x for x in self.yIndices if x in self.allIndices]) != len(self.yIndices):
            raise Exception("Invalid yIndices : all yIndices must be present in allIndices")


    def updateDataset(self, datasetPath, datasetType, yDataIndices, ignoreColIndices=[], ignoreRowIndices=[]):
        '''
            Update the AutoPreprocessor dataset

            Parameters:

                1. datasetPath      : Path of the dataset
                2. datasetType      : Type of the dataset ('csv'...)
                3. yDataIndices     : List of indices of the dependent variables in the dataset
                4. ignoreColIndices : List of indices of the columns to ignore in the dataset
                5. ignoreRowIndices : List of indices of the rows to ignore in the dataset

            RETURNS -> Void
        '''
        if type(ignoreColIndices) is not list:
            raise Exception("Invalid ignoreColIndices type : must type \"list\"")

        if type(ignoreRowIndices) is not list:
            raise Exception("Invalid ignoreRowIndices type : must type \"list\"")        

        if type(yDataIndices) is not list:
            raise Exception("Invalid yDataIndices type : must type \"list\"")

        if datasetType not in ("csv"):
            raise Exception("Unsupported dataset type")

        if not os.path.exists(datasetPath):
            raise Exception("The specified dataset path does not exist")

        if datasetType not in self.supportedDatasetType:
            raise Exception('Unsupported dataset type')

        if datasetType == 'csv':
            try:
                self.dataset = pd.read_csv(datasetPath)
            except Exception as e:
                self.dataset = None
                raise Exception("Invalid dataset ty file.")

        self.allIndices = [x for x in range(self.dataset.iloc[:, :].values.shape[1]) if x not in ignoreColIndices]

        for i in yDataIndices:
            if i not in self.allIndices:
                raise Exception("Invalid y data index : " + str(i))

        currRows = self.dataset.iloc[:, :].values.shape[0]
        for i in ignoreRowIndices:
            if i > currRows or i < 0:
                raise Exception("Invalid ignore row index : " + str(i))

        self.datasetType = datasetType
        self.datasetPath = datasetPath
        self.allRowsIndices = [x for x in range(self.dataset.iloc[:, :].values.shape[0]) if x not in ignoreRowIndices]
        self.yIndices = yDataIndices
        self.xIndices = [x for x in self.allIndices if x not in self.yIndices]


    def _validateCategoricalIndices(self, categoricalIndices=None):
        '''
            Validate the current dataset categorical indices

            RETURNS -> Void
        '''
        if categoricalIndices is None:
            categoricalIndices = self.categoricalIndices

        if type(categoricalIndices) is not list:
            raise Exception("Invalid categoricalIndices type : must type \"list\"")       
        
        for i in categoricalIndices:
            if i not in self.allIndices:
                raise Exception("Invalid categorical data index : " + str(i))


    def updateCategoricalIndices(self, categoricalIndices):
        '''
            Update the dataset indices to be treated as categorical

            Parameters:

                1. categoricalIndices : List of indices of the categorical columns in the dataset

            RETURNS -> Void
        '''
        self._validateDataset()
        self._validateCategoricalIndices(categoricalIndices)
        self.categoricalIndices = categoricalIndices


    def _validateOrdinalIndices(self, ordinalIndices=None):
        '''
            Validate the current dataset ordinal indices

            RETURNS -> Void
        '''
        if ordinalIndices is None:
            ordinalIndices = self.ordinalIndices

        if type(ordinalIndices) is not list:
            raise Exception("Invalid ordinalIndices type : must type \"list\"")       
        
        for i in ordinalIndices:
            if i not in self.allIndices:
                raise Exception("Invalid ordinal data index : " + str(i))


    def updateOrdinalIndices(self, ordinalIndices):
        '''
            Update the dataset indices to be treated as ordinal

            Parameters:

                1. ordinalIndices : List of indices of the ordinal columns in the dataset

            RETURNS -> Void
        '''
        self._validateDataset()
        self._validateOrdinalIndices(ordinalIndices)
        self.ordinalIndices = ordinalIndices


    def _validateNanHandlingMethod(self, nanDataHandling=None, nanFixedValue=None, addIsNanFeature=None, predictAutoTrainer=None, isCustomPredictAutoAiModel=False, 
                                   predictMaxIter=500, predictBatchSize=100, predictDumpEachIter=250, predictTestSize=0.2,
                                   predictVerboseLevel=0, predictKeepBestOnly=False):
        '''
            Validate the current NanHandlingMethod

            RETURNS -> Boolean : isAutoTrainer
        '''
        if nanDataHandling is None:
            nanDataHandling = self.nanDataHandling

        if nanFixedValue is None:
            nanFixedValue = self.nanFixedValue

        if addIsNanFeature is None:
            addIsNanFeature = self.addIsNanFeature

        if predictAutoTrainer is None and not isCustomPredictAutoAiModel:
            predictAutoTrainer = self.predictAutoTrainer


        if nanDataHandling is not None:
            if  nanDataHandling not in self.supportedNanHandlingMethods:
                raise Exception("Unsupported nanDataHandling method")

            if (type(nanFixedValue) not in (int, float) and nanDataHandling == 'fixed'):
               raise Exception("Invalid fixed value combination: if nanDataHandling is set to \"fixed\", the parameter type of \"nanFixedValue\" must be either float or int")

            if (nanFixedValue is not None and nanDataHandling != 'fixed'):
               raise Exception("Invalid value combination: if nanFixedValue is defined, the nanDataHandling parameter must be set to \"fixed\"")

            if (predictAutoTrainer is not None and nanDataHandling != 'predict'):
                raise Exception("Invalid value combination: if predictAutoTrainer is defined, the nanDataHandling parameter must be set to \"predict\"")

            isAutoTrainer = (predictAutoTrainer is not None and callable(getattr(predictAutoTrainer, "getModelsTypes", None)))

            if not isAutoTrainer and nanDataHandling == 'predict':
                raise Exception("Invalid predict value combination: if nanDataHandling is set to \"predict\", the parameter type of \"nanFixedValue\" must be autoAi.AutoTrainer")

            if isAutoTrainer and type(predictMaxIter) is not int:
                raise Exception('Invalid predictMaxIter type: predictMaxIter must be a int')

            if isAutoTrainer and (type(predictBatchSize) is not int or predictBatchSize > predictMaxIter):
                raise Exception('Invalid predictBatchSize type: predictBatchSize must be a int smaller than predictMaxIter')

            if isAutoTrainer and (type(predictDumpEachIter) is not int or predictDumpEachIter > predictMaxIter):
                raise Exception('Invalid predictDumpEachIter type: predictDumpEachIter must be a int smaller than predictMaxIter')

            if isAutoTrainer and (type(predictTestSize) is not float or predictTestSize >= 1 or predictTestSize <= 0):
                raise Exception('Invalid predictTestSize type: predictTestSize must be a int smaller than predictMaxIter between 0 and 1 exclusive')

            if isAutoTrainer and (type(predictVerboseLevel) is not int or predictVerboseLevel not in (0, 1, 2)):
                raise Exception('Invalid predictVerboseLevel type: predictVerboseLevel must be a int equal to 0, 1 or 2')

            if isAutoTrainer and type(predictKeepBestOnly) is not bool:
                raise Exception("Invalid predictKeepBestOnly type: predictKeepBestOnly must be a boolean")

            if type(addIsNanFeature) is not bool:
                raise Exception("Invalid addIsNanFeature type: addIsNanFeature must be a boolean")

            return isAutoTrainer


    def updateNaNHandlingMethod(self, nanDataHandling, nanFixedValue=None, addIsNanFeature=False, predictAutoTrainer=AutoTrainer(), 
                                predictMaxIter=500, predictBatchSize=100, predictDumpEachIter=250, predictTestSize=0.2,
                                predictVerboseLevel=0, predictKeepBestOnly=True):
        '''
            Update the NaN values handling method

            Parameters:

                1. nanDataHandling     : Method to use for NaN data handling. Supported methods for nanDataHandling: 
                                         1. 'delete'  : Delete any row that have a NaN
                                         2. 'mean'    : Replace NaN values with the mean of the columns containing 
                                                        the NaN values (Suitable for continuous data without outliers)
                                         3. 'median'  : Replace NaN values with the median of the columns containing 
                                                        the NaN values (Suitable for continuous data with outliers)
                                         4. 'mode'    : Replace NaN values with the mode of the columns containing the 
                                                        NaN values
                                         5. 'fixed'   : Replace NaN values with a fixed defined value (must be either 
                                                        of type Int or Float)
                                         6. 'predict' : Replace NaN values with a autoAi.AutoTrainer object instance
   
                2. nanFixedValue       : If nanDataHandling is 'fixed', it's the fixed value to replace the NaN values by
                3. addIsNanFeature     : If True, add a new column 'IsNaN_{COLUMN_NAME}' for each column with NaN values
                4. predictAutoTrainer  : If nanDataHandling is 'predict', it's the AutoTrainer instance to train the models
                5. predictMaxIter      : Maximum training iteration to reach
                6. predictBatchSize    : Size of every batch to fit during training
                7. predictDumpEachIter : Maximum iteration to reach before creating a model file (.pymodel)
                8. predictTestSize     : Size of the dataset to use for testing (0.2 = 20%, etc.)
                4. predictVerboseLevel : Verbose level (0, 1 or 2)
                5. predictKeepBestOnly : If True, keep only the most accurate model, delete the others after training

            RETURNS -> Void
        '''
        self._validateDataset()
        isAutoTrainer = self._validateNanHandlingMethod(nanDataHandling, nanFixedValue, addIsNanFeature, predictAutoTrainer, True, 
                                                        predictMaxIter, predictBatchSize, predictDumpEachIter, predictTestSize,
                                                        predictVerboseLevel, predictKeepBestOnly)
        self.nanDataHandling = nanDataHandling
        self.nanFixedValue = nanFixedValue
        self.addIsNanFeature = addIsNanFeature

        if isAutoTrainer:
            self.predictAutoTrainer = predictAutoTrainer
            self.predictMaxIter = predictMaxIter
            self.predictBatchSize = predictBatchSize
            self.predictDumpEachIter = predictDumpEachIter
            self.predictTestSize = predictTestSize
            self.predictVerboseLevel = predictVerboseLevel
            self.predictKeepBestOnly = predictKeepBestOnly
        else:
            self.predictAutoTrainer = None
            self.predictMaxIter = None
            self.predictBatchSize = None
            self.predictDumpEachIter = None
            self.predictTestSize = None
            self.predictVerboseLevel = None    
            self.predictKeepBestOnly = None        


    def _validateScaleData(self, scaleDataType=None):
        '''
            Validate the current scale data

            RETURNS -> Void
        '''
        if scaleDataType is None:
            scaleDataType = self.scaleDataType

        if scaleDataType is not None and (not all(isinstance(x, str) for x in scaleDataType) or not set(scaleDataType).issubset(self.supportedScaleDataType)):
            raise Exception("Invalid scaleDataType: must be a list of str. Each elements must be present in supportedScaleDataType")


    def updateScaleData(self, scaleDataType):
        '''
            Update the current scale data type

            Supported scale data types:

            1. 'minmax'   : Subtracts the minimum value then divides by range
            2. 'robust'   : Subtracts the median then divide by interquartile range
            3. 'standard' : Subtracts the mean then scale to unit variance

            RETURNS -> Void
        '''
        self._validateScaleData(scaleDataType)
        self.scaleDataType = scaleDataType
        self.scalers = []

        if self.scaleDataType is not None:
            for element in self.scaleDataType:
                if element == 'minmax':
                    self.scalers.append(MinMaxScaler())
                elif element == 'robust':
                    self.scalers.append(RobustScaler())
                elif element == 'standard':
                    self.scalers.append(StandardScaler())


    def getFullDataset(self, asArray=False):
        '''
            Return the current processed dataset

            Parameters:

                1. asArray : Return a Numpy array (if True) or a pandas dataframe

            RETURNS -> pd.DataFrame
        '''
        if asArray:
            return self.dataset.iloc[:, :].values
        else:
            return self.dataset.iloc[:, :]


    def getXDataset(self, asArray=False):
        '''
            Return the current processed dataset X values

            Parameters:

                1. asArray : Return a Numpy array (if True) or a pandas dataframe

            RETURNS -> pd.DataFrame
        '''
        if asArray:
            return self.dataset.iloc[:, self.xIndices].values
        else:
            return self.dataset.iloc[:, self.xIndices]


    def getYDataset(self, asArray=False):
        '''
            Return the current processed dataset Y values

            Parameters:

                1. asArray : Return a Numpy array (if True) or a pandas dataframe

            RETURNS -> pd.DataFrame
        '''
        if asArray:
            return self.dataset.iloc[:, self.yIndices].values
        else:
            return self.dataset.iloc[:, self.yIndices]


    def _getIsNaNsColValues(self, currDataset, iCol):
        '''
            Get a new IsNaN column for the specified column

            RETURNS -> pd.DataFrame : newColumn
        '''
        newCol = np.zeros(currDataset.values.shape)
        for i, val in enumerate(currDataset.values):
            if np.isnan(val):
                newCol[i] = 1
        newDf = pd.DataFrame(data=newCol, columns=[self.dataset.columns[iCol] + '_IsNaN'])
        newDf.reset_index(drop=True, inplace=True)
        return newDf


    def _updateColNamesAfterInsert(self, oldiColName, newDf):
        '''
            Update the current columns names after inserting a column

            RETURNS -> Void
        '''
        if oldiColName in self.xIndicesNames:
            oldiColIndex = self.xIndicesNames.index(oldiColName)
            self.xIndicesNames.insert(oldiColIndex, newDf.columns[0])

        if oldiColName in self.yIndicesNames:
            oldiColIndex = self.yIndicesNames.index(oldiColName)
            self.yIndicesNames.insert(oldiColIndex, newDf.columns[0])

        if oldiColName in self.categoricalIndicesNames:
            oldiColIndex = self.categoricalIndicesNames.index(oldiColName)
            self.categoricalIndicesNames.insert(oldiColIndex, newDf.columns[0])

        if oldiColName in self.ordinalIndicesNames:
            oldiColIndex = self.ordinalIndicesNames.index(oldiColName)
            self.ordinalIndicesNames.insert(oldiColIndex, newDf.columns[0])


    def _executeNaN(self, iCol):
        '''
            Execute the current NaN handling method on a column on the current
            dataset

            RETURNS -> Void
        '''
        currDataset = deepcopy(self.dataset.iloc[:, [iCol]])

        hasNaNs = currDataset.isnull().values.any()

        if not hasNaNs:
            return

        newDf = None
        if self.addIsNanFeature:
            newDf = self._getIsNaNsColValues(currDataset, iCol)

        if self.nanDataHandling == 'predict':
            modelDatasetTemp = deepcopy(self.dataset)
            modelDatasetTemp.dropna(inplace = True)
            colName = modelDatasetTemp.columns[iCol]

            sha = hashlib.sha256()
            sha.update('_'.join([x for x in modelDatasetTemp.columns if x != colName]).encode())
            modelProjectName = colName + "_" + sha.hexdigest()[:8]

            model = AIModel(modelProjectName, baseDumpPath="NanPrecitModels", 
                            autoTrainer=True, autoTrainerInstance=self.predictAutoTrainer)
            modelXTemp = modelDatasetTemp.iloc[:, [x for x in self.allIndices if x != iCol]]
            modelYTemp = modelDatasetTemp.iloc[:, [iCol]]
            model.updateDataSet(modelXTemp, modelYTemp, test_size=0.2)
            model.train(max_iter=self.predictMaxIter, batchSize=self.predictBatchSize,
                        dumpEachIter=self.predictDumpEachIter, verboseLevel=self.predictVerboseLevel,
                        keepBestModelOnly=self.predictKeepBestOnly)
            model.loadBestModel(verboseLevel=self.predictVerboseLevel)

            for iRow, val in enumerate(currDataset.iloc[:, :].values):
                if np.isnan(val):
                    prediction = model.predict(self.dataset.iloc[[iRow], [x for x in self.allIndices if x != iCol]].values)
                    currDataset.loc[[iRow]] = prediction

            self.dataset.iloc[:, [iCol]] = currDataset


        elif self.nanDataHandling == 'delete':
            currDataset = self.dataset.iloc[:, :]
            currDataset.dropna(inplace = True)
            self.dataset.iloc[:, :] = currDataset
        
        elif self.nanDataHandling == 'mean':
            currDataset = currDataset.replace(np.NaN, currDataset.mean())
            self.dataset.iloc[:, [iCol]] = currDataset
            if newDf is not None:
                self.dataset.insert(loc=iCol + 1, value=newDf, column=newDf.columns[0])
                oldiColName = newDf.columns[0].split('_')[:-1]
                self._updateColNamesAfterInsert(oldiColName, newDf)
                self._updateAllIndices()

        elif self.nanDataHandling == 'median':
            currDataset = currDataset.replace(np.NaN, currDataset.median())
            self.dataset.iloc[:, [iCol]] = currDataset
            if newDf is not None:
                self.dataset.insert(loc=iCol + 1, value=newDf, column=newDf.columns[0])
                oldiColName = newDf.columns[0].split('_')[:-1]
                self._updateColNamesAfterInsert(oldiColName, newDf)
                self._updateAllIndices()

        elif self.nanDataHandling == 'mode':
            if newDf is not None:
                for colName in self.dataset.columns:
                    iCol = self.dataset.columns.get_loc(colName)
                    hasNaNs = currDataset.isnull().values.any()
                    if hasNaNs:
                        newDf = self._getIsNaNsColValues(currDataset, iCol)
                        self.dataset.insert(loc=iCol + 1, value=newDf, column=newDf.columns[0])
                        oldiColName = newDf.columns[0].split('_')[:-1]
                        self._updateColNamesAfterInsert(oldiColName, newDf)
                        self.allIndices.insert(iCol + 1, newDf.columns[0])
                        self._updateAllIndices()
        

            if iCol == len(self.dataset.columns) - 1:
                u, indices = np.unique(currDataset.values, return_inverse=True)
                modeVal = u[np.argmax(np.apply_along_axis(np.bincount, 0, indices.reshape(currDataset.values.shape), None, np.max(indices) + 1), axis=0)]
                currDataset = currDataset.replace(np.NaN, modeVal[0])
                

        elif self.nanDataHandling == 'fixed':
            currDataset = currDataset.replace(np.NaN, self.nanFixedValue)
            self.dataset.iloc[:, [iCol]] = currDataset
            if newDf is not None:
                self.dataset.insert(loc=iCol + 1, value=newDf, column=newDf.columns[0])
                oldiColName = newDf.columns[0].split('_')[:-1]
                self._updateColNamesAfterInsert(oldiColName, newDf)
                self._updateAllIndices()


    def _executeCategorical(self, iCol):
        '''
            Execute the categorical data handling on a column on the current
            dataset

            RETURNS -> Void
        '''
        currDataset = self.dataset.iloc[:, iCol].values
        transformed = self.ohEncoder.fit_transform(currDataset.reshape(-1, 1)).toarray()[:, :-1]

        indices = [x for x in range(iCol, iCol + transformed.shape[1], 1)]
        newCols = [self.dataset.columns[[iCol]][0] + "_Categorical_" + str(x) for x in indices]


        dfArray = []
        for dim in range(transformed.shape[1]):
            newDf = pd.DataFrame(data=transformed[:, dim], columns=[newCols[dim]])
            newDf.reset_index(drop=True, inplace=True)
            dfArray.append(newDf)

        dfArrayLen = len(dfArray)
        dfArray = pd.concat(dfArray, axis=1)

        dfArrayConcat = []
        dfBefore = self.dataset.iloc[:, :iCol]
        dfBefore.reset_index(drop=True, inplace=True)
        dfArrayConcat.append(dfBefore)

        dfArrayConcat.append(dfArray)

        dfAfter = self.dataset.iloc[:, (iCol + dfArrayLen):]
        dfAfter.reset_index(drop=True, inplace=True)
        dfArrayConcat.append(dfAfter)
        
        currColName = self.dataset.columns[iCol]

        if currColName in self.xIndicesNames:
            self.xIndicesNames.remove(currColName)
            self.xIndicesNames = self.xIndicesNames + newCols

        if currColName in self.yIndicesNames:
            self.yIndicesNames.remove(currColName)
            self.yIndicesNames = self.yIndicesNames + newCols

        if currColName in self.ordinalIndicesNames:
            self.ordinalIndicesNames.remove(currColName)
            self.ordinalIndicesNames = self.ordinalIndicesNames + newCols

        if currColName in self.categoricalIndicesNames:
            self.categoricalIndicesNames.remove(currColName)
            self.categoricalIndicesNames = self.categoricalIndicesNames + newCols

        self.dataset.drop(currColName, axis=1, inplace=True)
        self.dataset = pd.concat(dfArrayConcat, axis=1)


    def _executeOrdinal(self, iCol):
        '''
            Execute the ordinal data handling on a column on the current
            dataset

            RETURNS -> Void
        '''
        currDataset = self.dataset.iloc[:, iCol].values
        transformed = self.labelEncoder.fit_transform(currDataset.reshape(-1, 1).ravel())

        newDf = pd.DataFrame(data=transformed, columns=[self.dataset.columns[[iCol]][0]])
        newDf.reset_index(drop=True, inplace=True)
        self.dataset.iloc[:, iCol] = newDf


    def _executeScaler(self):
        '''
            Execute the scaling data handling on the current
            dataset

            RETURNS -> Void            
        '''
        currDataset = self.dataset.iloc[:, :].values
        for i in range(len(self.scalers)):
            transformed = self.scalers[i].fit_transform(currDataset)

            newDf = pd.DataFrame(data=transformed, columns=self.dataset.columns)
            newDf.reset_index(drop=True, inplace=True)
            self.dataset.iloc[:, :] = newDf


    def _executeFeatureSelection(self):
        '''
            Execute the feature selection data handling on the current
            dataset

            RETURNS -> Void                   
        '''
        if self.featureSelectionMethod == 'backward':
            self._backwardElimination()
        elif self.featureSelectionMethod == 'recursive':
            self._recursiveElimination()
        elif self.featureSelectionMethod == 'embedded':
            self._embeddedElimination()


    def _delIgnoreRow(self):
        '''
            Delete all rows that arent considered in allRowsIndices

            RETURNS -> Void
        '''
        self.dataset.iloc[:, :] = self.dataset.iloc[self.allRowsIndices, :]
        self.allRowsIndices = [x for x in range(len(self.dataset.iloc[:, :].values))]


    def _updateAllIndices(self):
        '''
            Update all indices parameters given the current data

            RETURNS -> Void
        '''
        self.xIndices = [self.dataset.columns.get_loc(x) for x in self.xIndicesNames]
        self.yIndices = [self.dataset.columns.get_loc(x) for x in self.yIndicesNames]
        self.categoricalIndices = [self.dataset.columns.get_loc(x) for x in self.categoricalIndicesNames]
        self.ordinalIndices = [self.dataset.columns.get_loc(x) for x in self.ordinalIndicesNames]
        self.allIndices = [x for x in range(self.dataset.iloc[:, :].values.shape[1]) if x in self.xIndices or x in self.yIndices]


    def execute(self):
        '''
            Execute the preprocessing given the current AutoProcessor parameters

            RETURNS -> Void
        '''
        self._validateDataset()
        self._validateFeatureSelectionMethod()
        self._validateCategoricalIndices()
        self._validateOrdinalIndices()
        self._validateNanHandlingMethod()
        self._validateScaleData()

        self._delIgnoreRow()

        self.categoricalIndicesNames = [self.dataset.columns[x] for x in self.categoricalIndices]
        self.ordinalIndicesNames = [self.dataset.columns[x] for x in self.ordinalIndices]
        self.xIndicesNames = [self.dataset.columns[x] for x in self.xIndices]
        self.yIndicesNames = [self.dataset.columns[x] for x in self.yIndices]

        i = 0
        iCol = 0
        while i < len(self.allIndices):
            iCol = self.allIndices[i]
            if iCol in self.categoricalIndices:
                self._executeCategorical(iCol)

            if iCol in self.ordinalIndices:
                self._executeOrdinal(iCol)

            if self.nanDataHandling != 'predict':
                self._executeNaN(iCol)

            i = i + 1

        if self.scaleDataType is not None and len(self.scaleDataType) > 0:
            self._executeScaler()
        self._updateAllIndices()

        if self.nanDataHandling == 'predict':
            i = 0
            iCol = 0
            while i < len(self.allIndices):
                iCol = self.allIndices[i]
                self._executeNaN(iCol)
                i = i + 1

        if self.featureSelectionMethod is not None:
            self._executeFeatureSelection()


    def export(self, filePath, fileType='csv'):
        '''
            Export the current dataset to a given file format

            Parameters:

                1. filePath : Path of the file
                2. fileType : Type of the file ('csv'...)

            RETURNS -> Void
        '''
        if fileType not in self.supportedDatasetType:
            raise Exception("Unsupported dataset type")

        if fileType == 'csv':
            self.dataset.to_csv(filePath, index=False)


    def inverseScaleRow(self, rowIndex):
        '''
            Get the original row data representation, without any scaler

            Parameters:

                1. rowIndex : Index of the row to inverse scale

            RETURNS -> Float : value
        '''
        baseRow = self.dataset.iloc[[rowIndex], :].values
        for scaler in reversed(self.scalers):
            i = self.scalers.index(scaler)
            baseRow = self.scalers[i].inverse_transform(baseRow)

        return baseRow