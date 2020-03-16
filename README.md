# pyAiTrainer
pyAiTrainer is a high-level AI model handler that allows easy model handling and automatic training for a large amount of differents models.

### Language: ### 

- Tested in Python 3.7.5, should work in all Python 3.7+ versions.

### Limitations: ###

- pyAiTrainer is not able to train multiple models in parallel, it does so sequentially.

- For now, it only works with models from the Scikit-Learn python library. Keras models will soon be supported.

### Table of Contents: ###

- Classes
  - [*AIModel*](https://github.com/FanaticPythoner/pyAiTrainer#aimodel-class)
  - [*AutoTrainer*](https://github.com/FanaticPythoner/pyAiTrainer#autotrainer-class)
  
# Installation

- Download the repository

- Install the requirements in the requirements.txt file (pip install -r requirements.txt)

# AIModel Class

### Description : ###
Class that allows machine learning / neural network model handling with autotraining support.

### Usage / Code sample : ###
*This example create an [*AIModel*](https://github.com/FanaticPythoner/pyAiTrainer#aimodel-class) object with a given machine learning model from the [*scikit-learn*](https://pypi.org/project/scikit-learn/) library, then train it.*
```python
# Get the dataset
import pandas as pd
df = pd.read_csv("Test_Dataset\\iris2.csv")
x = df.iloc[:, 0:4]
y = df.iloc[:, 4:]

# Create a machine learning model
from sklearn.ensemble import AdaBoostClassifier
mlModel = AdaBoostClassifier()

# Import AIModel
from pyAiTrainer import AIModel

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
```

### Usage / Code sample : ###
*This example create a [*AIModel*](https://github.com/FanaticPythoner/pyAiTrainer#aimodel-class) object with the auto trainer enabled, then train it on every model available in the default [*AutoTrainer*](https://github.com/FanaticPythoner/pyMagicBytes#autotrainer-class) class.*
```python
# Get the dataset
import pandas as pd
df = pd.read_csv("Test_Dataset\\iris2.csv")
x = df.iloc[:, 0:4]
y = df.iloc[:, 4:]

# Import AIModel
from pyAiTrainer import AIModel

# Create the AIModel with the auto trainer enabled
model = AIModel("MyModel_AutoTraining", baseDumpPath="Output_Models", autoTrainer=True)

# Update the AIModel dataset
model.updateDataSet(x, y, test_size=0.2)

# Train the AIModel
model.train(max_iter=50, batchSize=10, dumpEachIter=25, verboseLevel=2)

# Load the best model from all trained models located in "Output_Models/MyModel_AutoTraining"
model.loadBestModel()
```

When we run the previous code, we should get a warning similar to this one:

![alt text](https://i.imgur.com/p4lbi1o.png)

#### This is totaly normal, as not all default custom-parameters-less models are compatible with this dataset input shape. #### 


# AutoTrainer Class

### Description : ###
Class that contains all supported auto-training models

### Usage / Code sample : ###
*This example create a custom [*AutoTrainer*](https://github.com/FanaticPythoner/pyMagicBytes#autotrainer-class) object, then feeds it to an [*AIModel*](https://github.com/FanaticPythoner/pyAiTrainer#aimodel-class) object, then train the [*AIModel*](https://github.com/FanaticPythoner/pyMagicBytes#aimodel-class).
```python
# Get the dataset
import pandas as pd
df = pd.read_csv("Test_Dataset\\iris2.csv")
x = df.iloc[:, 0:4]
y = df.iloc[:, 4:]

# Creating the custom AutoTrainer class.
class AutoTrainer():
    def getModelsTypes(self):
        import sklearn.ensemble
        import sklearn.linear_model
        return [
            (sklearn.ensemble.RandomForestClassifier, { "n_estimators": 100,
                                                        "random_state": 42 }),
            (sklearn.ensemble.RandomForestRegressor, {}),
            (sklearn.linear_model.LinearRegression, {}),
        ]

# Import AIModel
from pyAiTrainer import AIModel

# Create the AIModel
model = AIModel("MyModel_CustomTrainer", baseDumpPath="Output_Models", autoTrainer=True, autoTrainerInstance=AutoTrainer())

# Update the AIModel dataset
model.updateDataSet(x, y, test_size=0.2)

# Train the AIModel
model.train(max_iter=50, batchSize=10, dumpEachIter=25, verboseLevel=2)

# Load the best model from all trained models located in "Output_Models/MyModel_CustomTrainer"
model.loadBestModel()
```
