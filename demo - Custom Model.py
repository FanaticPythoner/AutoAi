# Get the dataset
import pandas as pd
df = pd.read_csv("Test_Dataset\\iris2.csv")
x = df.iloc[:, 0:4]
y = df.iloc[:, 4:]

# Create a machine learning model
from sklearn.ensemble import AdaBoostClassifier
mlModel = AdaBoostClassifier()

# Import AIModel
from autoAi import AIModel

# Create the AIModel
model = AIModel("MyModel_CustomModel", baseDumpPath="Output_Models")

# Update the AIModel model
model.updateModel(mlModel)

# Update the AIModel dataset
model.updateDataSet(x, y, test_size=0.2)

# Train the AIModel
model.train(max_iter=50, batchSize=10, dumpEachIter=25, verboseLevel=2)

# Load the best model from all trained models located in "Output_Models/MyModel_CustomModel"
model.loadBestModel()
