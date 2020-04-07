# AutoAi
AutoAi is a high-level AI automation library that allows things like automatic training for a large amount of differents models and automatic data preprocessing. Support custom class implementation, making it work on any neural network you can imagine.

## Update: ##
- Added support in the [*AutoPreprocessor*](https://github.com/FanaticPythoner/AutoAi#autopreprocessor-class) for applying custom functions (or lambda) on specific columns at specific time using the *addApplyFunctionForColumn* method
- Added example for the *addApplyFunctionForColumn* method in the [*AutoPreprocessor*](https://github.com/FanaticPythoner/AutoAi#autopreprocessor-class)

### Language: ### 

- Tested in Python 3.7.5, should work in all Python 3.7+ versions.

### Limitations: ###

- AutoAi is not able to train multiple models in parallel, it does so sequentially.

### Table of Contents: ###

- Classes
  - [*AIModel*](https://github.com/FanaticPythoner/AutoAi#aimodel-class)
  - [*AutoTrainer*](https://github.com/FanaticPythoner/AutoAi#autotrainer-class)
  - [*AutoPreprocessor*](https://github.com/FanaticPythoner/AutoAi#autopreprocessor-class)
  
- Interfaces
  - [*ICustomWrapper*](https://github.com/FanaticPythoner/AutoAi#icustomwrapper-interface)
  
# Installation

- Download the repository

- Install the requirements in the requirements.txt file (pip install -r requirements.txt)

# AIModel Class

### Description : ###
Class that allows neural network / machine handling model handling with autotraining support.

### Usage / Code sample (Custom Model): ###
*This example create an [*AIModel*](https://github.com/FanaticPythoner/AutoAi#aimodel-class) object with a given machine learning model from the [*scikit-learn*](https://pypi.org/project/scikit-learn/) library, then train it.*
```python
# Get the dataset
import pandas as pd
df = pd.read_csv("Test_Dataset\\iris_preprocessed_predict_all.csv")
x = df.iloc[:, 0:4]
y = df.iloc[:, 4:]

# Create a machine learning model
from sklearn.ensemble import RandomForestClassifier
mlModel = RandomForestClassifier()

# Import AIModel
from autoAi.AIModel import AIModel

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

When we run the previous code, we should get something like this:

![alt text](https://i.imgur.com/TH7qxTd.png)

The image speaks for itself.

### Usage / Code sample (Automatic Training): ###
*This example create a [*AIModel*](https://github.com/FanaticPythoner/AutoAi#aimodel-class) object with the auto trainer enabled, then train it on every model available in the default [*AutoTrainer*](https://github.com/FanaticPythoner/AutoAi#autotrainer-class) class.*
```python
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
```

When we run the previous code, we should get this line telling us that the auto trainer is activated:

![alt text](https://i.imgur.com/kFHeOrB.png)


# AutoTrainer Class

### Description : ###
Class that contains all supported auto-training models

### Usage / Code sample (Automatic Training With Custom Trainer): ###
*This example create a custom [*AutoTrainer*](https://github.com/FanaticPythoner/AutoAi#autotrainer-class) object, then feeds it to an [*AIModel*](https://github.com/FanaticPythoner/AutoAi#aimodel-class) object, then train the [*AIModel*](https://github.com/FanaticPythoner/AutoAi#aimodel-class). In the method *getModelsTypes* in the [*AutoTrainer*](https://github.com/FanaticPythoner/AutoAi#autotrainer-class) class, the second element in every tuple is a dictionary of parameters for the first tuple element, which is a machine learning / neural network model class.
```python
# Get the dataset
import pandas as pd
df = pd.read_csv("Test_Dataset\\iris_preprocessed_predict_all.csv")
x = df.iloc[:, 0:4]
y = df.iloc[:, 4:]


# Creating the custom AutoTrainer class.
class AutoTrainer():
    def getModelsTypes(self):
        import sklearn.ensemble
        import sklearn.linear_model
        return [
            sklearn.ensemble.RandomForestClassifier(n_estimators=100, random_state=42),
            sklearn.ensemble.RandomForestRegressor(),
            sklearn.linear_model.LinearRegression()
        ]

# Import AIModel
from autoAi.AIModel import AIModel

# Create the AIModel
model = AIModel("MyModel_CustomTrainer", baseDumpPath="Output_Models", autoTrainer=True, autoTrainerInstance=AutoTrainer())

# Update the AIModel dataset
model.updateDataSet(x, y, test_size=0.2)

# Train the AIModel
model.train(max_iter=50, batchSize=10, dumpEachIter=25, verboseLevel=2)

# Load the best model from all trained models located in "Output_Models/MyModel_CustomTrainer"
model.loadBestModel()
```

# AutoPreprocessor Class

### Description : ###
Class that allows automatic data preprocessing

### Usage / Code sample (AutoPreprocessor Predict NaN with AiModel and AutoTrainer): ###
*This example uses the [*AIModel*](https://github.com/FanaticPythoner/AutoAi#aimodel-class) and create a model for each column in the dataset that has NaN values to predict those values. The [*AutoTrainer*](https://github.com/FanaticPythoner/AutoAi#autotrainer-class) can either be specified or not, if not like in this case, the default [*AutoTrainer*](https://github.com/FanaticPythoner/AutoAi#autotrainer-class) class is used.*
```python
from autoAi.AutoPreprocessor import AutoPreprocessor

# Create the AutoPreprocessor object
obj = AutoPreprocessor(datasetPath='Test_Dataset\\iris.csv', 
                       datasetType='csv', yDataNames=['species'])

# Specify the dataset categorical names
obj.updateCategoricalColumns(categoricalNames=['species'])

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
```

### Usage / Code sample (AutoPreprocessor Predict NaN with AiModel and custom AutoTrainer): ###
*This example uses the [*AIModel*](https://github.com/FanaticPythoner/AutoAi#aimodel-class) and create a model for each column in the dataset that has NaN values to predict those values. The [*AutoTrainer*](https://github.com/FanaticPythoner/AutoAi#autotrainer-class) is created and passed as a parameter to the [*AutoPreprocessor*](https://github.com/FanaticPythoner/AutoAi#autopreprocessor-class) instance.*
```python
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
```

### Usage / Code sample (AutoPreprocessor with custom column lambdas): ###
*This example apply a first lambda expression on the column 'sepal_length' after the categorical preprocessing, then a second lambda expression on the same column after all preprocessing steps occured*
```python
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
```

# ICustomWrapper Interface

### Description : ###
Interface for custom neural network implementation

### Usage / Code sample (Custom Neural Network Implementation): ###
*This example create an [*AIModel*](https://github.com/FanaticPythoner/AutoAi#aimodel-class) object with a given custom neural nework model from the [*keras*](https://pypi.org/project/Keras/) library using the [*ICustomWrapper*](https://github.com/FanaticPythoner/AutoAi#icustomwrapper-interface) interface, then train it.*
```python
# Get the dataset
import pandas as pd
df = pd.read_csv("Test_Dataset\\iris_preprocessed_predict_all.csv")
x = df.iloc[:, 0:4]
y = df.iloc[:, 4:]

# Import the ICustomWrapper interface
from autoAi.Interfaces import ICustomWrapper

# Import the Keras libraries
from keras.layers import Dense
from keras.models import Sequential

# Create the Wrapper using the ICustomWrapper interface
class CustomKerasWrapper(ICustomWrapper):
    def __init__(self):
        self.model = Sequential()
        self.model.add(Dense(7, input_dim=4, activation='relu'))
        self.model.add(Dense(7, activation='relu'))
        self.model.add(Dense(2, activation='sigmoid'))

        self.model.compile(loss='categorical_crossentropy', 
                      optimizer='adam', 
                      metrics=['mean_squared_error'])

    def fit(self, X, y):
        self.model.fit(X, y, verbose=0)

    def predict(self, X):
        return self.model.predict(X)


# Import AIModel
from autoAi.AIModel import AIModel

# Create the AIModel
model = AIModel("MyModel_CustomModelKeras", baseDumpPath="Output_Models")

# Update the AIModel model with a CustomKerasWrapper
model.updateModel(CustomKerasWrapper())

# Update the AIModel dataset
model.updateDataSet(x, y, test_size=0.2)

# Train the AIModel
model.train(max_iter=1000, batchSize=10, dumpEachIter=500, verboseLevel=2)

# Load the best model from all trained models located in "Output_Models/MyModel_CustomModelKeras"
model.loadBestModel()
```
