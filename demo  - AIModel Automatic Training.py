# Get the dataset
import pandas as pd
df = pd.read_csv("Test_Dataset\\iris_preprocessed_predict_all.csv")
x = df.iloc[:, 0:4]
y = df.iloc[:, 4:]

# Import AIModel
from autoAi.AIModel import AIModel

# Create the AIModel with the auto trainer enabled
model = AIModel("MyModel_AutoTraining", baseDumpPath="Output_Models", autoTrainer=True)

# Update the AIModel dataset
model.updateDataSet(x, y, test_size=0.2)

# Train the AIModel
model.train(max_iter=50, batchSize=10, dumpEachIter=25, verboseLevel=2)

# Load the best model from all trained models located in "Output_Models/MyModel_AutoTraining"
model.loadBestModel()