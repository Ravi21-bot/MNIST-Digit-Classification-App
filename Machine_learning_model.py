import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#importing dataset 
mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

#Now processing our dataset
train_images = train_images/255.0
test_images = test_images/255.0

#Adding dimensions to our dataset for later data augmentation 
train_images = np.expand_dims(train_images, axis=3)
test_images = np.expand_dims(test_images, axis=3)

#creating ur model
def create_model():
	model =keras.Sequential([
		keras.layers.InputLayer(input_shape=(28, 28, 1)),
		keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation = tf.nn.relu),
		keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation = tf.nn.relu),
		keras.layers.MaxPooling2D(pool_size=(2, 2)),
		keras.layers.Dropout(0.25),
		keras.layers.Flatten(),
		keras.layers.Dense(10, activation = tf.nn.softmax)
	])
	model.compile(optimizer ='adam',
		loss ='sparse_categorical_crossentropy',
		metrics = ['accuracy'])

	return model



#Now adding augmented value (The Image Augmentation is used in order to improve accuracy of our model.)
datagen = keras.preprocessing.image.ImageDataGenerator(
	rotation_range = 30,
	width_shift_range = 0.25,
	height_shift_range = 0.25,
	shear_range = 0.25,
	zoom_range = 0.2)

train_generator = datagen.flow(train_images, train_labels)
test_generator = datagen.flow(test_images, test_labels)

'''augmented_images, augmented_labels = next(train_generator)
plt.figure(figsize=(10, 10))
for i in range (25):
	plt.subplot(5, 5, i+1)
	plt.xticks([])
	plt.yticks([])
	plt.grid(False)
	plt.imshow(np.squeeze(augmented_images[i], axis = 2), cmap=plt.cm.gray)
	plt.xlabel('Label: %d' % augmented_labels[i])
	plt.show()'''

#Training our Augmented Model OR Improved Model
improved_model = create_model()
improved_model.fit(train_generator, epochs=5, validation_data = test_generator)
