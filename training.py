from functools import partialmethod
import re
from keras import callbacks
from keras import optimizers
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import RMSprop, SGD, Adam
import keras  # a library for deep learning
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
import os

from tensorflow.python.training.tracking.util import Checkpoint

num_classes = 5
img_rows, img_columns = 48, 48
batch_size = 5  # number of images that gives for the model to train at once

train_data_dir = r'C:\Users\pahan\Documents\Facial Expressions Recognition\train'
validation_data_dir = r'C:\Users\pahan\Documents\Facial Expressions Recognition\validation'

train_data_generator = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    shear_range=0.3,
    zoom_range=0.3,
    width_shift_range=0.4,
    height_shift_range=0.3,
    horizontal_flip=True,
    vertical_flip=True,
)

validation_data_generator = ImageDataGenerator(rescale=1./255)

train_generator = train_data_generator.flow_from_directory(
    train_data_dir, color_mode='grayscale', target_size=(img_rows, img_columns),
    batch_size=batch_size, class_mode='categorical', shuffle="True"
)

validation_generator = validation_data_generator.flow_from_directory(
    validation_data_dir, color_mode='grayscale', target_size=(img_rows, img_columns),
    batch_size=batch_size, class_mode='categorical', shuffle=True
)


model = Sequential()  # defining the model

# Block 01
model.add(Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal',
          input_shape=(img_rows, img_columns, 1)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal',
          input_shape=(img_columns, img_columns, 1)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))


# Block 02
model.add(Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal',
          input_shape=(img_rows, img_columns, 1)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal',
          input_shape=(img_columns, img_columns, 1)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))


# Block 03
model.add(Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal',
          input_shape=(img_rows, img_columns, 1)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal',
          input_shape=(img_columns, img_columns, 1)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))


# Block 04
model.add(Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal',
          input_shape=(img_rows, img_columns, 1)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal',
          input_shape=(img_columns, img_columns, 1)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))


# Block 05
model.add(Flatten())
model.add(Dense(64, kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))


# Block 06
model.add(Dense(64, kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))


# Block 07
model.add(Dense(num_classes, kernel_initializer='he_normal'))
model.add(Activation('softmax'))

print(model.summary())


checkpoint = ModelCheckpoint(
    r'C:\Users\pahan\Documents\Facial Expressions Recognition\emotions_little_vgg.h5',
    monitor='val_loss',
    mode='min',
    save_best_only=True
)

earlystop = EarlyStopping(
    monitor='val_loss',
    min_delta=0,
    patience=3,
    verbose=1,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=3,
    verbose=1,
    min_delta=0.0001
)

callbacks = [earlystop, checkpoint, reduce_lr]

model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(lr=0.001),
    metrics=['accuracy']
)


nb_train_samples = 24176
nb_validation_samples = 3006
epochs = 25

history = model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples//batch_size,
    epochs=epochs,
    callbacks=callbacks,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples//batch_size

)
