from sklearn.model_selection import train_test_split
import os
import numpy as np

import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
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
from autoAi.AutoTrainer import AutoTrainer
from autoAi.AIModel import AIModel


class AutoPreprocessor():
    '''
        Class that allows automatic data preprocessing.

            Parameters:

                1. datasetPath      : Path of the dataset
                2. datasetType      : Type of the dataset ('csv'...)
                3. yDataNames       : List of names of the dependent variables in the dataset
                4. ignoreColNames   : List of indices of the columns to ignore in the dataset
                5. ignoreRowIndices : List of indices of the rows to ignore in the dataset

        For documentation, refer to https://github.com/FanaticPythoner/AutoAi
    '''
    
    def __init__(self, datasetPath, datasetType, yDataNames, ignoreColNames=[], ignoreRowIndices=[]):
        super().__init__()
        self.supportedNanHandlingMethods = ('delete', 'mean', 'median', 'mode', 'fixed', 'predict')
        self.supportedFeatureSelectionMethod = ('backward', 'recursive', 'embedded', 'vif')
        self.supportedScaleDataType = ('robust', 'minmax', 'standard')
        self.supportedDatasetType = ('csv')
        self.supportedStepAddFunctionApply = (0,1,2,3,4,5)
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
        self.featureSelectionMethodThreshold = None
        self.predictAutoTrainer = None
        self.ohEncoder = OneHotEncoder()
        self.labelEncoder = LabelEncoder()
        self.dicFunctionColumnsApplyList = {}
        self.dicFunctionColumnsApplyListSteps = {
            0:False,
            1:False,
            2:False,
            3:False,
            4:False,
            5:False
        }
        self.updateDataset(datasetPath, datasetType, yDataNames, ignoreColNames, ignoreRowIndices)

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

        self.ignoreColIndicesScaler = []
        self.ignoreColIndicesScalerNames = []


    def _validateFeatureSelectionMethod(self, featureSelectionMethod=None, threshold=None):
        '''
            Validate the current feature selection method

            RETURNS -> Void
        '''
        if featureSelectionMethod is None:
            featureSelectionMethod = self.featureSelectionMethod

        if threshold is None:
            threshold = self.featureSelectionMethodThreshold

        if featureSelectionMethod is not None and featureSelectionMethod not in self.supportedFeatureSelectionMethod:
            raise Exception("Unsupported featureSelectionMethod")

        if threshold is not None and featureSelectionMethod not in ('vif', 'backward'):
            raise Exception("Invalid parameters combination : if threshold is set, featureSelectionMethod must be either 'vif' or 'backward'")

        if threshold is not None and (threshold >= 1 and threshold <= 0) and featureSelectionMethod == 'backward':
            raise Exception("Invalid threshold : The threshold for 'backward' must be between 0 and 1")

        if threshold is not None and threshold <= 1 and featureSelectionMethod == 'vif':
            raise Exception("Invalid threshold : The threshold for 'vif' must be smaller than 1")


    def updateFeatureSelectionMethod(self, featureSelectionMethod, threshold=None):
        '''
            Update the AutoProcessor feature selection method

            Supported methods for featureSelectionMethod:

            1. 'backward'  : Backward Elimination feature selection
            2. 'recursive' : Recursive Elimination feature selection
            3. 'embedded'  : Embedded Elimination feature selection
            4. 'vif'       : VIF (Variable Inflation Factors) Elimination feature selection

            RETURNS -> Void
        '''
        self._validateDataset()
        self._validateFeatureSelectionMethod(featureSelectionMethod, threshold)
        self.featureSelectionMethod = featureSelectionMethod
        self.featureSelectionMethodThreshold = threshold


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
        self.ignoreColIndicesScalerNames = [val for val in self.ignoreColIndicesScalerNames if val in self.xIndicesNames or val in self.yIndicesNames]
        self.ignoreColIndicesScaler = [self.dataset.columns.get_loc(val) for val in self.ignoreColIndicesScalerNames]

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
            self.ignoreColIndicesScaler = [self.dataset.columns.get_loc(val) for val in self.ignoreColIndicesScalerNames if val != deletedXVal]

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
        self.ignoreColIndicesScalerNames = [val for val in self.ignoreColIndicesScalerNames if val in self.xIndicesNames or val in self.yIndicesNames]
        self.ignoreColIndicesScaler = [self.dataset.columns.get_loc(val) for val in self.ignoreColIndicesScalerNames]

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
            self.ignoreColIndicesScaler = [self.dataset.columns.get_loc(val) for val in self.ignoreColIndicesScalerNames if val != deletedXVal]

        self.yIndices = [self.dataset.columns.get_loc(val) for val in self.yIndicesNames]


    def _backwardElimination(self, SL=0.05):
        '''
            Perform automatic backward elimination on independent variables
            with a given threshold (SL)

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
        newIgnoreScalerIndicesNames = self.ignoreColIndicesScalerNames

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
                        newIgnoreScalerIndicesNames = [val for val in newIgnoreScalerIndicesNames if val != deletedXVal]

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
        self.ignoreColIndicesScalerNames = newIgnoreScalerIndicesNames
        self.ignoreColIndicesScaler = [datasetCopy.columns.get_loc(val) for val in self.ignoreColIndicesScalerNames]
        self.allIndices = [x for x in range(datasetCopy.values.shape[1]) if x in self.xIndices or x in self.yIndices]


    def _getVif(self, threshold):
        '''
            Get the current X dataset VIF values

            RETURNS -> Int : deletedIndex
        '''
        deletedIndices = []
        valsVif = []
        newDataset = deepcopy(self.dataset.iloc[:, self.xIndices])
        yDataset = deepcopy(self.dataset.iloc[:, self.yIndices])
        datasetLen = newDataset.shape[1]
        i = 0
        while i < datasetLen:
            currVif = variance_inflation_factor(self.dataset.values, i)
            valsVif.append((newDataset.columns[i], currVif))
            if currVif >= threshold:
                deletedIndices.append((i, currVif))
            i = i + 1
        
        if len(deletedIndices) == 0:
            return None

        deletedIndices = sorted(deletedIndices, key=lambda x: float(x[1]), reverse=True)
        deletedIndex = deletedIndices[0][0]
        deletedName = newDataset.columns[deletedIndex]

        newDataset.drop(newDataset.columns[deletedIndex], axis=1, inplace=True)

        self.xIndicesNames = [x for x in newDataset.columns if x != deletedName]
        self.xIndices = [newDataset.columns.get_loc(val) for val in self.xIndicesNames]

        newDataset = pd.concat([newDataset] + [yDataset], axis=1)
        self.categoricalIndicesNames = [val for val in self.categoricalIndicesNames if val in self.xIndicesNames or val in self.yIndicesNames]
        self.categoricalIndices = [newDataset.columns.get_loc(val) for val in self.categoricalIndicesNames]
        self.ordinalIndicesNames = [val for val in self.ordinalIndicesNames if val in self.xIndicesNames or val in self.yIndicesNames]
        self.ordinalIndices = [newDataset.columns.get_loc(val) for val in self.ordinalIndicesNames]
        self.yIndices = [newDataset.columns.get_loc(val) for val in self.yIndicesNames]
        self.ignoreColIndicesScalerNames = [val for val in self.ignoreColIndicesScalerNames if val in self.xIndicesNames or val in self.yIndicesNames]
        self.ignoreColIndicesScaler = [newDataset.columns.get_loc(val) for val in self.ignoreColIndicesScalerNames]

        self.dataset = newDataset
        return deletedIndex


    def _vifElimination(self, threshold=10):
        '''
            Perform VIF (Variable Inflation Factors) elimination on independent variables
            with a given threshold

            RETURNS -> Void
        '''
        deletedIndices = []
        deletedIndex = -1

        while True:
            deletedIndex = self._getVif(threshold)
            if deletedIndex is None:
                break
            deletedIndices.append(deletedIndex)

        self.allIndices = [x for x in range(self.dataset.iloc[:, :].values.shape[1])]
        

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


    def updateDataset(self, datasetPath, datasetType, yDataNames, ignoreColNames=[], ignoreRowIndices=[]):
        '''
            Update the AutoPreprocessor dataset

            Parameters:

                1. datasetPath      : Path of the dataset
                2. datasetType      : Type of the dataset ('csv'...)
                3. yDataNames       : List of indices of the dependent variables in the dataset
                4. ignoreColNames   : List of indices of the columns to ignore in the dataset
                5. ignoreRowIndices : List of indices of the rows to ignore in the dataset

            RETURNS -> Void
        '''
        if type(ignoreColNames) is not list:
            raise Exception("Invalid ignoreColNames type : must type \"list\"")

        if type(ignoreRowIndices) is not list:
            raise Exception("Invalid ignoreRowIndices type : must type \"list\"")        

        if type(yDataNames) is not list:
            raise Exception("Invalid yDataNames type : must type \"list\"")

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
                raise Exception("Invalid dataset type file.")

        for colName in ignoreColNames:
            self.dataset.drop(colName, axis=1, inplace=True)

        self.allIndices = [x for x in range(self.dataset.iloc[:, :].values.shape[1])]

        self.yIndices = [x for x in range(self.dataset.iloc[:, :].values.shape[1]) if self.dataset.columns[x] in yDataNames]

        for i in self.yIndices:
            if i not in self.allIndices:
                raise Exception("Invalid y data index : " + str(i))

        currRows = self.dataset.iloc[:, :].values.shape[0]
        for i in ignoreRowIndices:
            if i > currRows or i < 0:
                raise Exception("Invalid ignore row index : " + str(i))

        self.datasetType = datasetType
        self.datasetPath = datasetPath
        self.allRowsIndices = [x for x in range(self.dataset.iloc[:, :].values.shape[0]) if x not in ignoreRowIndices]

        self.xIndices = [x for x in self.allIndices if x not in self.yIndices]

        for name in self.dataset.columns.values:
            self.dicFunctionColumnsApplyList[name] = []


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


    def updateCategoricalColumns(self, categoricalNames):
        '''
            Update the dataset indices to be treated as categorical

            Parameters:

                1. categoricalNames : List of names of the categorical columns in the dataset

            RETURNS -> Void
        '''
        self._validateDataset()
        categoricalIndices = [self.dataset.columns.get_loc(x) for x in categoricalNames]
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


    def updateOrdinalColumns(self, ordinalNames):
        '''
            Update the dataset indices to be treated as ordinal

            Parameters:

                1. ordinalNames : List of names of the ordinal columns in the dataset

            RETURNS -> Void
        '''
        self._validateDataset()
        ordinalIndices = [self.dataset.columns.get_loc(x) for x in ordinalNames]
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

            isAutoTrainer = (nanDataHandling == 'predict' and predictAutoTrainer is not None and callable(getattr(predictAutoTrainer, "getModelsTypes", None)))

            if not isAutoTrainer and nanDataHandling == 'predict':
                isAutoTrainer = True
                predictAutoTrainer = AutoTrainer()

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


    def updateNaNHandlingMethod(self, nanDataHandling, nanFixedValue=None, addIsNanFeature=False, predictAutoTrainer=None, 
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


    def _validateScaleData(self, scaleDataType=None, ignoreColIndices=[]):
        '''
            Validate the current scale data

            RETURNS -> Void
        '''
        if scaleDataType is None:
            scaleDataType = self.scaleDataType

        if ignoreColIndices is None:
            ignoreColIndices = self.ignoreColIndicesScaler

        if type(ignoreColIndices) is not list:
            raise Exception("Invalid ignoreColIndices type : must type \"list\"")

        if scaleDataType is not None and (not all(isinstance(x, str) for x in scaleDataType) or not set(scaleDataType).issubset(self.supportedScaleDataType)):
            raise Exception("Invalid scaleDataType: must be a list of str. Each elements must be present in supportedScaleDataType")


    def updateScaleData(self, scaleDataType, ignoreColNames=[]):
        '''
            Update the current scale data type

            Supported scale data types:

            1. 'minmax'   : Subtracts the minimum value then divides by range
            2. 'robust'   : Subtracts the median then divide by interquartile range
            3. 'standard' : Subtracts the mean then scale to unit variance

            RETURNS -> Void
        '''
        ignoreColIndices = []
        try:
            ignoreColIndices = [self.dataset.columns.get_loc(x) for x in self.dataset.columns if x in ignoreColNames]
        except Exception as e:
            pass

        self._validateScaleData(scaleDataType, ignoreColIndices)
        self.scaleDataType = scaleDataType
        self.ignoreColIndicesScaler = ignoreColIndices
        self.ignoreColIndicesScalerNames = [self.dataset.columns[x] for x in self.ignoreColIndicesScaler]
        self.scalers = []

        if self.scaleDataType is not None:
            for element in self.scaleDataType:
                if element == 'minmax':
                    self.scalers.append(MinMaxScaler())
                elif element == 'robust':
                    self.scalers.append(RobustScaler())
                elif element == 'standard':
                    self.scalers.append(StandardScaler())


    def _validateAddApplyFunctionForColumn(self, colName, step):
        '''
            Validate the a function to apply

            RETURNS -> Void
        '''
        if colName not in self.dataset.columns.values:
            raise Exception('Invalid colName')

        if step not in self.supportedStepAddFunctionApply :
            raise Exception('Unsupported step')


    def addApplyFunctionForColumn(self, colName, function, step=0):
        '''
            Add a function to apply to a given column in the current dataset

            Parameters:

                1. colName             : Name of the column to apply the function on
                2. function            : Function to apply
                3. step                : When will the function be applied to the column.
                                            a) 0 : Before doing any preprocessing
                                            b) 1 : After doing the categorical preprocessing
                                            c) 2 : After doing the ordinal preprocessing
                                            d) 3 : After doing the NaN preprocessing (if NaN is 'predict', it's actually after scaling (4).)
                                            e) 4 : After doing the scaling preprocessing
                                            f) 5 : After doing the feature selection preprocessing
        '''
        self._validateAddApplyFunctionForColumn(colName, step)
        self.dicFunctionColumnsApplyListSteps[step] = True
        self.dicFunctionColumnsApplyList[colName].append((function, step))


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
            isNan = False
            try:
                isNan = np.isnan(val)
            except Exception as e:
                pass
            if isNan:
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

            model = AIModel(modelProjectName, baseDumpPath="NanPredictModels", 
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

            RETURNS -> Int : CategoricalIndexModifier
        '''
        currDataset = self.dataset.iloc[:, iCol].values
        transformed = self.ohEncoder.fit_transform(currDataset.reshape(-1, 1)).toarray()
        if transformed.shape[1] > 1:
            transformed = transformed[:, :-1]

        indices = [x for x in range(iCol, iCol + transformed.shape[1], 1)]
        newCols = [self.dataset.columns[[iCol]][0] + "_Categorical_" + str(x) for x in indices]

        categoricalIndexModifier = len(newCols) - 1

        dfArray = []
        for dim in range(transformed.shape[1]):
            newDf = pd.DataFrame(data=transformed[:, dim], columns=[newCols[dim]])
            newDf.reset_index(drop=True, inplace=True)
            dfArray.append(newDf)

        if len(dfArray) > 0:
            dfArray = pd.concat(dfArray, axis=1)

        dfArrayConcat = []
        dfBefore = self.dataset.iloc[:, :iCol]
        dfBefore.reset_index(drop=True, inplace=True)
        dfArrayConcat.append(dfBefore)

        if len(dfArray) > 0:
            dfArrayConcat.append(dfArray)

        dfAfter = self.dataset.iloc[:, (iCol + 1):]
        dfAfter.reset_index(drop=True, inplace=True)
        dfArrayConcat.append(dfAfter)
        
        currColName = self.dataset.columns[iCol]

        if currColName in self.xIndicesNames:
            currColNameIndex = self.xIndicesNames.index(currColName)
            self.xIndicesNames.remove(currColName)
            self.xIndicesNames = self.xIndicesNames[:currColNameIndex] + newCols + self.xIndicesNames[currColNameIndex:]

        if currColName in self.yIndicesNames:
            currColNameIndex = self.yIndicesNames.index(currColName)
            self.yIndicesNames.remove(currColName)
            self.yIndicesNames = self.yIndicesNames[:currColNameIndex] + newCols + self.yIndicesNames[currColNameIndex:]

        if currColName in self.ordinalIndicesNames:
            currColNameIndex = self.ordinalIndicesNames.index(currColName)
            self.ordinalIndicesNames.remove(currColName)
            self.ordinalIndicesNames = self.ordinalIndicesNames[:currColNameIndex] + newCols + self.ordinalIndicesNames[currColNameIndex:]

        if currColName in self.categoricalIndicesNames:
            currColNameIndex = self.categoricalIndicesNames.index(currColName)
            self.categoricalIndicesNames.remove(currColName)
            self.categoricalIndicesNames = self.categoricalIndicesNames[:currColNameIndex] + newCols + self.categoricalIndicesNames[currColNameIndex:]

        if currColName in self.ignoreColIndicesScalerNames:
            currColNameIndex = self.ignoreColIndicesScalerNames.index(currColName)
            self.ignoreColIndicesScalerNames.remove(currColName)
            self.ignoreColIndicesScalerNames = self.ignoreColIndicesScalerNames[:currColNameIndex] + newCols + self.ignoreColIndicesScalerNames[currColNameIndex:]

        for key in list(self.dicFunctionColumnsApplyList.keys()):
            if currColName == key:
                oldVal = self.dicFunctionColumnsApplyList.pop(key)
                for newColName in newCols:
                    self.dicFunctionColumnsApplyList[newColName] = oldVal 

        self.dataset.drop(currColName, axis=1, inplace=True)
        self.dataset = pd.concat(dfArrayConcat, axis=1)
        return categoricalIndexModifier


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
            if len(self.ignoreColIndicesScaler) > 0:
                transformed = self.scalers[i].fit_transform(currDataset[:, [x for x in self.allIndices if x not in self.ignoreColIndicesScaler]])
                transformed = np.concatenate((transformed, currDataset[:, self.ignoreColIndicesScaler]), axis=1)
            else:
                transformed = self.scalers[i].fit_transform(currDataset)

            newCols = [x for x in self.dataset.columns if self.dataset.columns.get_loc(x) not in self.ignoreColIndicesScaler]
            newCols = newCols + [self.dataset.columns[x] for x in self.ignoreColIndicesScaler]

            newDf = pd.DataFrame(data=transformed, columns=newCols)
            newDf.reset_index(drop=True, inplace=True)

            self.dataset.iloc[:, :] = newDf


    def _executeFeatureSelection(self):
        '''
            Execute the feature selection data handling on the current
            dataset

            RETURNS -> Void                   
        '''
        if self.featureSelectionMethod == 'backward':
            if self.featureSelectionMethodThreshold is not None:
                self._backwardElimination(SL=self.featureSelectionMethodThreshold)
        elif self.featureSelectionMethod == 'recursive':
            self._recursiveElimination()
        elif self.featureSelectionMethod == 'embedded':
            self._embeddedElimination()
        elif self.featureSelectionMethod == 'vif':
            if self.featureSelectionMethodThreshold is not None:
                self._vifElimination(threshold=self.featureSelectionMethodThreshold)


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
        self.ignoreColIndicesScaler = [self.dataset.columns.get_loc(val) for val in self.ignoreColIndicesScalerNames]

    
    def _executeStep(self, step, colName=None, index=None):
        '''
            Execute all the function to apply to columns for a given step

            RETURNS -> Void
        '''
        def applyToCol(key, function):
            newDf = getattr(self.dataset, key).apply(function)
            newDf.values
            setattr(self.dataset, key, newDf)

        def getNewIndexCategorical(oldColName, oldIndex):
            partCategorical = '_Categorical_' + str(oldIndex)
            newColName = oldColName + partCategorical
            while newColName in self.dicFunctionColumnsApplyList:
                oldIndex = oldIndex - 1
                newColName = oldColName + '_Categorical_' + str(oldIndex)
            oldIndex = oldIndex + 1
            return oldIndex

        def applyToColLoopIter(val, col):
            function, currStep = val
            if currStep == step:
                applyToCol(col, function)

        if self.dicFunctionColumnsApplyListSteps[step]:
            lst = None
            
            if colName is None:
                for key, lst in list(self.dicFunctionColumnsApplyList.items()):
                    for val in lst:
                        applyToColLoopIter(val, key)
                return

            try:
                partCategorical = '_Categorical_' + str(index)
                if colName.endswith(partCategorical):
                    colName = partCategorical.join(colName.split(partCategorical)[:-1])
                    index = getNewIndexCategorical(colName, index)
                    raise KeyError()
                lst = self.dicFunctionColumnsApplyList[colName]
                for val in lst:
                    applyToColLoopIter(val, colName)

            except KeyError:
                newColName = colName + '_Categorical_' + str(index)
                lst = self.dicFunctionColumnsApplyList[newColName]
                index = getNewIndexCategorical(colName, index)
                newColName = colName + '_Categorical_' + str(index)
                modifier = 0

                while newColName in self.dicFunctionColumnsApplyList:
                    lst = self.dicFunctionColumnsApplyList[newColName]
                    for val in lst:
                        applyToColLoopIter(val, newColName)

                    modifier = modifier + 1
                    newColName = colName + '_Categorical_' + str(index + modifier)
                    

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

        self._executeStep(0)

        i = 0
        iCol = 0
        categoricalIndexModifier = 0
        while i < len(self.allIndices):
            iCol = self.allIndices[i] + categoricalIndexModifier
            if iCol >= self.dataset.columns.size:
                break
            currColName = self.dataset.columns.values[iCol]
            if currColName in self.categoricalIndicesNames:
                newModifCategorical = self._executeCategorical(iCol)
                categoricalIndexModifier = categoricalIndexModifier + newModifCategorical
                self._updateAllIndices()
            
            self._executeStep(1, currColName, iCol)

            if currColName in self.ordinalIndicesNames:
                self._executeOrdinal(iCol)
                self._updateAllIndices()
            
            self._executeStep(2, currColName, iCol)

            if self.nanDataHandling != 'predict':
                self._executeNaN(iCol)
                self._updateAllIndices()
                self._executeStep(3, currColName, iCol)

            i = i + 1

        if self.scaleDataType is not None and len(self.scaleDataType) > 0:
            self._executeScaler()
            self._updateAllIndices()
        
        self._executeStep(4)

        if self.nanDataHandling == 'predict':
            i = 0
            iCol = 0
            while i < len(self.allIndices):
                iCol = self.allIndices[i]
                self._executeNaN(iCol)
                i = i + 1
            self._executeStep(3)

        if self.featureSelectionMethod is not None:
            self._executeFeatureSelection()
        
        self._executeStep(5)


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


    def plotDataset(self):
        '''
            Show a visual representation of the current dataset fetures

            RETURNS -> Void
        '''
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(style="ticks")
        sns.set_palette("husl")
        sns.pairplot(self.getXDataset())
        plt.show()


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