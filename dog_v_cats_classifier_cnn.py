# -*- coding: utf-8 -*-
"""Dog-v-Cats_Classifier_CNN.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1tJRIxQcBP54-TXsiHwTRkRbgVqq8AwbV
"""

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/

!kaggle datasets download -d salader/dogs-vs-cats

import zipfile
zip_ref = zipfile.ZipFile('/content/dogs-vs-cats.zip')
zip_ref.extractall('/content')
zip_ref.close()

import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,BatchNormalization,Dropout
from keras.utils import load_img, img_to_array,image_dataset_from_directory
from keras.preprocessing.image import ImageDataGenerator

batch_size = 32

# this is the augmentation configuration we will use for training
train_ds = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
validation_ds = ImageDataGenerator(rescale=1./255)

# this is a generator that will read pictures found in
# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data
train_generator = train_ds.flow_from_directory(
        '/content/train',  # this is the target directory
        target_size=(256, 256),  # all images will be resized to 150x150
        batch_size=batch_size,
        class_mode='binary')  # since we use binary_crossentropy loss, we need binary labels

# this is a similar generator, for validation data
validation_generator = validation_ds.flow_from_directory(
        '/content/test',
        target_size=(256, 256),
        batch_size=batch_size,
        class_mode='binary')



#create CNN model

model = Sequential()

model.add(Conv2D(16,kernel_size=(3,3),padding='valid',strides=(1,1),activation='relu',input_shape=(256,256,3)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2),padding='valid',strides=2))

model.add(Conv2D(32,kernel_size=(3,3),padding='valid',strides=(1,1),activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2),padding='valid',strides=2))

model.add(Conv2D(64,kernel_size=(3,3),padding='valid',strides=(1,1),activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2),padding='valid',strides=2))

model.add(Conv2D(128,kernel_size=(3,3),padding='valid',strides=(1,1),activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2),padding='valid',strides=2))

model.add(Flatten())

model.add(Dense(128,activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(1,activation='sigmoid'))

model.summary()

model.compile(loss='binary_crossentropy', optimizer='adam', metrics= ['accuracy'])

callback = tf.keras.callbacks.EarlyStopping(
    monitor = 'val_loss',
    patience = 10,
    start_from_epoch = 25,
    mode = 'min',

)

#history = model.fit(train_ds, epochs=5, validation_data=validation_ds, verbose=True)

history = model.fit_generator(
        train_generator,
        steps_per_epoch=2000 // batch_size,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=800 // batch_size,
        callbacks = [callback])

import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'],color='red',label='train')
plt.plot(history.history['val_accuracy'],color='blue',label='validation')
plt.legend()
plt.show()

plt.plot(history.history['loss'],color='red',label='train')
plt.plot(history.history['val_loss'],color='blue',label='validation')
plt.legend()
plt.show()

import cv2

test_img = cv2.imread('/content/cat2.jpg')

plt.imshow(test_img)

test_img.shape

test_img = cv2.resize(test_img,(256,256))
plt.imshow(test_img)

test_input = test_img.reshape((1,256,256,3))

model.predict(test_input)
