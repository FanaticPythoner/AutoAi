from autoAi.AutoPreprocessor import AutoPreprocessor

# Create the AutoPreprocessor object
obj = AutoPreprocessor(datasetPath='Test_Dataset\\iris.csv', 
                       datasetType='csv', yDataIndices=[4])

# Specify the dataset ordinal indices
obj.updateCategoricalIndices(categoricalIndices=[4])

# Specify the current data scale type
obj.updateScaleData(scaleDataType=['minmax'])

# Specify the dataset data handling method. In this case 'predict', which 
# will use the autoAi.AiModel to build models that will predict the NaNs values
obj.updateNaNHandlingMethod(nanDataHandling='predict', predictMaxIter=50, predictBatchSize=10,
                            predictDumpEachIter=25, predictVerboseLevel=2)

# Execute the preprocessing with the current settings
obj.execute()

# Export the preprocessed dataset
obj.export(filePath="Test_Dataset\\iris_preprocessed_predict_all.csv", 
           fileType='csv')

# Print the preprocessed data
print(obj.getFullDataset())