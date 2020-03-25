import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

# %matplotlib inline 
import os
np.random.seed(2)

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools

from keras.utils.np_utils import to_categorical 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

sns.set(style='white', context='notebook', palette='deep')
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
config.gpu_options.allow_growth = True 
session = tf.Session(config=config)

# 设置session
KTF.set_session(session )

train = pd.read_csv("~/machine_l/Database/Digit_Reg/train.csv")

test = pd.read_csv("~/machine_l/Database/Digit_Reg//test.csv")

Y_train = train['label']

X_train = train.drop(labels = ["label"], axis = 1)

del train

g = sns.countplot(Y_train)

Y_train.value_counts()

# plt.show()

X_train.isnull().any().describe()

test.isnull().any().describe()

X_train = X_train / 255.0
test = test / 255.0

X_train = X_train.values.reshape(-1, 28, 28, 1)
test = test.values.reshape(-1, 28, 28, 1)

Y_train = to_categorical(Y_train, num_classes = 10)

random_seed = 2

X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, \
    test_size = 0.1, random_state=random_seed)

g = plt.imshow(X_train[0][:,:,0])

# plt.show()


# Set the CNN model 
# my CNN architechture is In -> [[Conv2D->relu]*2 -> MaxPool2D -> Dropout]*2 -> Flatten -> Dense -> Dropout -> Out

model = Sequential()
model.add(Conv2D(filters = 32, kernel_size = (5, 5), padding = 'Same', \
    activation = 'relu', input_shape = (28, 28, 1)))

model.add(Conv2D(filters = 32, kernel_size = (5, 5), padding = 'Same', \
    activation = 'relu' ))

model.add(MaxPool2D(pool_size = (2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters = 64, kernel_size = (3,3), padding = 'Same', \
    activation = 'relu'))

model.add(Conv2D(filters = 64, kernel_size = (3,3), padding = 'Same', \
    activation = 'relu'))
model.add(MaxPool2D(pool_size = (2,2), strides = (2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation = 'softmax'))
optimizer = RMSprop(lr = 0.001, rho = 0.9, epsilon = 1e-08, decay =0.0)

model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics=['accuracy'])

learning_rate_reduction = ReduceLROnPlateau(monitor = 'val_acc', 
                                            patience=3,
                                            verbose=1,
                                            factor = 0.5,
                                            min_lr = 0.00001)
epochs = 1
batch_size = 86



datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

datagen.fit(X_train)

history = model.fit_generator(datagen.flow(X_train, Y_train, batch_size),
                                        epochs = epochs, validation_data = (X_val, Y_val),
                                        verbose = 2, 
                                        steps_per_epoch = X_train.shape[0]//batch_size,
                                        callbacks = [learning_rate_reduction])


# predict_value = model.predict(test_data, verbose=2)

fig, ax = plt.subplots(2,1)
ax[0].plot(history.history['loss'], color='b', label='Training loss')
ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['acc'], color='b', label="Training accuracy")
ax[1].plot(history.history['val_acc'], color='r',label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)
