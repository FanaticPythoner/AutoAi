from autoAi.AutoPreprocessor import AutoPreprocessor

# Create the AutoPreprocessor object
obj = AutoPreprocessor(datasetPath='Test_Dataset\\iris.csv', 
                       datasetType='csv', yDataNames=['species'])

# Show a visual representation of the current dataset fetures
obj.plotDataset()