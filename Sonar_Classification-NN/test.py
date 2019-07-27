import numpy as np
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
import csv

def read_data(url):
  dataframe = read_csv(url)
  dataset = dataframe.values
  # split into input (X) and output (Y) variables
  X = dataset[:,0:60].astype(float)
  Y = dataset[:,60]
  return X, Y

# baseline
def create_model():  
  model = Sequential()
  # YOUR CODE HERE
  model.add(Dense(units=60, input_dim=60, activation='relu'))
  model.add(Dense(units=30, activation='relu'))
  model.add(Dense(units=1, activation='sigmoid'))
  return model

# load dataset
url_train = 'train_dataset.csv'
url_test = 'test_dataset.csv'

X_train, Y_train = read_data(url_train)
X_test, Y_test = read_data(url_test)

model = create_model()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, Y_train, epochs=1000)
score, acc = model.evaluate(X_test, Y_test)

print("Test accuracy:" + str(acc))

