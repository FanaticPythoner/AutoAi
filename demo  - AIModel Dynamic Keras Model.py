# Get the dataset
import pandas as pd
df = pd.read_csv("Test_Dataset\\iris_preprocessed_predict_all.csv")
x = df.iloc[:, 0:4]
y = df.iloc[:, 4:]

# Import the DynamicKerasWrapper
from autoAi.AutoTrainer import DynamicKerasWrapper

# Import AIModel
from autoAi.AIModel import AIModel

# Create the AIModel
model = AIModel("MyModel_DynamicModelKeras", baseDumpPath="Output_Models")

# Update the AIModel model with a DynamicKerasWrapper
model.updateModel(DynamicKerasWrapper())

# Update the AIModel dataset
model.updateDataSet(x, y, test_size=0.2)

# Train the AIModel
model.train(max_iter=1000, batchSize=10, dumpEachIter=500, verboseLevel=2)

# Load the best model from all trained models located in "Output_Models/MyModel_DynamicModelKeras"
model.loadBestModel()
