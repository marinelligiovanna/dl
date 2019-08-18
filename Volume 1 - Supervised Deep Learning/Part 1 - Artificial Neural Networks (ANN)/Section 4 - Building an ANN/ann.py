import os

os.chdir(
    "D:\Documents\Github\dl\Volume 1 - Supervised Deep Learning\Part 1 - Artificial Neural Networks (ANN)\Section 4 - Building an ANN")

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Loading data ----------------------------------------
dataset = pd.read_csv("Churn_Modelling.csv")
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, -1].values

# Preprocessing ----------------------------------------

# Enconding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])

labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

onehotencoder = OneHotEncoder(categorical_features=[1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Splitting the dataset
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Making the ANN --------------------------------------
import keras
from keras.models import Sequential  # Init NN
from keras.layers import Dense  # Create the layers of NN

# Initializing the ANN (Defining it as a sequence
# of layers)
classifier = Sequential()

# Adding the input the layer and first hidden layer
classifier.add(Dense(units=6,  # Num of Units on Hidden Layer =  Avg of num. nodes on input  + num. nodes on output
                     kernel_initializer='uniform',  # Initialized of the weights
                     activation='relu',  # Rectifier
                     input_dim=11))  # Num. of variables on input layer

# Adding the second hidden layer
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))

# Adding the output layer
classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

# Compiling ANN (adding Gradient Descent Optimizer)
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fitting the ANN to the training set
classifier.fit(X_train, y_train, batch_size=10, epochs=100)

# Predicting the Test set result
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the confusion matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
