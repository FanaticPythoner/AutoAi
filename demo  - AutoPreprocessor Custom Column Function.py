from autoAi.AutoPreprocessor import AutoPreprocessor

# Create the AutoPreprocessor object
obj = AutoPreprocessor(datasetPath='Test_Dataset\\iris.csv', 
                       datasetType='csv', yDataNames=['species'])

# Creating a function to apply on each specified element of a given column
def addString(element):
    return element + ' is Amazing!'

# Applying the function to each element in the column 'species' before any preprocessing steps (0)
obj.addApplyFunctionForColumn('species', addString, step=0)

# Execute the preprocessing with the current settings
obj.execute()

# Export the preprocessed dataset
obj.export(filePath="Test_Dataset\\iris_preprocessed_functions.csv", 
           fileType='csv')

# Print the preprocessed data
print(obj.getFullDataset())
