from __future__ import absolute_import, division, print_function

# TensorFlow and tf.keras
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

#download data set of fashion-MNIST
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
#reshape data to 4D dimesions
train_images = train_images.reshape(60000,28,28,1)
test_images = test_images.reshape(10000,28,28,1)
train_images = train_images / 255.0
test_images = test_images / 255.0 

#show the first 25 examples
# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap=plt.cm.binary)
#     plt.xlabel(class_names[train_labels[i]])
# plt.show()

#visuallization label of first 100 exxamples
# train_labels[0:100]

#Define CNN model: CNN layers and Dense layers
model = keras.Sequential()
model.add(Conv2D(64, kernel_size=(3,3), activation='relu', input_shape=(28,28,1)))
model.add(MaxPooling2D())
model.add(Conv2D(32, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#view model
# model.summary()

#train model with epochs = 15
model.fit(train_images, train_labels, epochs=15)

#test model
test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)