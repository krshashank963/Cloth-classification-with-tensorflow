# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 09:42:45 2020

@author: SHASHANK RAJPUT
"""
# from keras.layers import Dense,Flatten,Convolution2D,MaxPooling2D
# from keras.models import Sequential
import tensorflow as tf
from tensorflow import keras
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

plt.imshow(test_images[10])

plt.colorbar()
plt.show()
#Rescaling test and train images
train_images = train_images / 255.0
test_images = test_images / 255.0

plt.imshow(test_images[10])
plt.colorbar()
plt.show()

for i in range(1,30):
    plt.subplot(5,6,i+1)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
    
#build the model
# model = Sequential()
#     #to make a vector of iage specification
# model.add(Flatten())
# #hiddden layer
# model.add(Dense(128, activation='relu'))

# #output layer
# model.add(Dense(10, activation='sigmoid'))

# model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics =['accuracy'])


    model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10)
])
    
#complile the model
    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    
    
model.fit(train_images, train_labels, epochs=20)

#makng prediction
probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)

img = test_images[7]

print(img.shape)
img = (np.expand_dims(img,0))

print(img.shape)
plt.imshow(test_images[7])
plt.colorbar()
plt.show()
predictions_single = probability_model.predict(img)

print(predictions_single)
plot_value_array(1, predictions_single[0], test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)
np.argmax(predictions_single[0])

