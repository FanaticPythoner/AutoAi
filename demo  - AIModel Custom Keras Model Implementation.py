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
    '''
        Class that represent a custom Keras model
    '''
    def __init__(self):
        self.model = Sequential()
        self.model.add(Dense(7, input_dim=4, activation='relu'))
        self.model.add(Dense(7, activation='relu'))
        self.model.add(Dense(2, activation='sigmoid'))

        self.model.compile(loss='categorical_crossentropy', 
                      optimizer='adam', 
                      metrics=['mean_squared_error'])

    def fit(self, X, y):
        '''
            Train the model on a given X and Y dataset

            RETURNS -> Void
        '''
        self.model.fit(X, y, verbose=0)

    def predict(self, X):
        '''
            Predict X values

            RETURNS -> np.Array(Float) : Predictions
        '''
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

# Load the best model from all trained models located in "Output_Models/MyModel_CustomModel"
model.loadBestModel()