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

from autoAi.AutoTrainer import AutoTrainer

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

