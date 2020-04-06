from autoAi.AutoPreprocessor import AutoPreprocessor

# Creating the custom AutoTrainer class
class CustomAutoTrainer():
    def getModelsTypes(self):
        import sklearn.ensemble
        return [
            sklearn.ensemble.VotingRegressor(estimators=[('lr', sklearn.linear_model.LinearRegression()),
                                                         ('rf', sklearn.ensemble.RandomForestRegressor(n_estimators=50))])
        ]

# Create the AutoPreprocessor object
obj = AutoPreprocessor(datasetPath='Test_Dataset\\iris.csv', 
                       datasetType='csv', yDataNames=['species'])

# Specify the dataset categorical names
obj.updateCategoricalColumns(categoricalNames=['species'])

# Specify the current data scale type
obj.updateScaleData(scaleDataType=['minmax'])

# Specify the dataset data handling method. In this case 'predict', which 
# will use the autoAi.AiModel to build models that will predict the NaNs values
obj.updateNaNHandlingMethod(nanDataHandling='predict', predictAutoTrainer=CustomAutoTrainer(), 
                            predictMaxIter=50, predictBatchSize=10, predictDumpEachIter=25, 
                            predictVerboseLevel=2)

# Execute the preprocessing with the current settings
obj.execute()

# Export the preprocessed dataset
obj.export(filePath="Test_Dataset\\iris_preprocessed_predict_custom.csv", 
           fileType='csv')

# Print the preprocessed data
print(obj.getFullDataset())