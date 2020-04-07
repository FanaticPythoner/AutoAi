from autoAi.AutoPreprocessor import AutoPreprocessor

# Create the AutoPreprocessor object
obj = AutoPreprocessor(datasetPath='Test_Dataset\\iris.csv', 
                       datasetType='csv', yDataNames=['species'])

# Specify the dataset categorical names
obj.updateCategoricalColumns(categoricalNames=['species'])

# Specify the current data scale type
obj.updateScaleData(scaleDataType=['minmax'])

# Multiplying by 1000 each element in the column 'sepal_length' after doing the
# categorical preprocessing (1)
obj.addApplyFunctionForColumn('sepal_length', lambda x: x * 1000, step=1)

# Dividing by 80 each element in the column 'sepal_length' after every preprocessing steps (5)
obj.addApplyFunctionForColumn('sepal_length', lambda x: x / 80, step=5)

# Execute the preprocessing with the current settings
obj.execute()

# Export the preprocessed dataset
obj.export(filePath="Test_Dataset\\iris_preprocessed_lambdas.csv", 
           fileType='csv')

# Print the preprocessed data
print(obj.getFullDataset())