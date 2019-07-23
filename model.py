import os
import csv

samples = []
with open('./Training data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

import cv2
import numpy as np
import sklearn
import keras
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten, Cropping2D, Convolution2D

def generator(samples, batch_size=32):
    num_samples = len(samples)
    adjustment = 0.05
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                center_name = './Training data/IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.cvtColor(cv2.imread(center_name), cv2.COLOR_BGR2RGB)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

                flipped_center_image = np.copy(np.fliplr(center_image))
                images.append( flipped_center_image )
                angles.append(-center_angle)

                left_name = './Training data/IMG/'+batch_sample[1].split('/')[-1]
                left_image = cv2.cvtColor(cv2.imread(left_name), cv2.COLOR_BGR2RGB)
                left_angle = center_angle + adjustment
                images.append(left_image)
                angles.append(left_angle)

                flipped_left_image = np.copy(np.fliplr(left_image))
                images.append( flipped_left_image )
                angles.append(-left_angle)

                right_name = './Training data/IMG/'+batch_sample[2].split('/')[-1]
                right_image = cv2.cvtColor(cv2.imread(right_name), cv2.COLOR_BGR2RGB)
                right_angle = center_angle - adjustment
                images.append(right_image)
                angles.append(right_angle)

                flipped_right_image = np.copy(np.fliplr(right_image))
                images.append( flipped_right_image )
                angles.append(-right_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# Set our batch size
batch_size=32

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

row, col, ch  = 160, 320, 3  # Trimmed image format

model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation 
model.add(Lambda(lambda x: x / 255 - 0.5, input_shape=(row, col, ch)))
model.add(Cropping2D(cropping = ((70, 25), (0, 0))))
# Nvidia end To end pipeline
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation = 'relu'))
model.add(Convolution2D(36, 5, 5, subsample =(2, 2), activation = 'relu'))
model.add(Convolution2D(48, 5, 5, subsample =(2, 2),activation = 'relu'))
model.add(Convolution2D(64, 3, 3, activation = 'relu'))
model.add(Convolution2D(64, 3, 3, activation = 'relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
  
# Model.Fit(X_train, y_train, validation_split = 0.2, shuffle = True, nb_epoch = 3)
 
model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator,
            steps_per_epoch=np.ceil(len(train_samples)/batch_size),
            validation_data=validation_generator,
            validation_steps=np.ceil(len(validation_samples)/batch_size),
            epochs=5, verbose=1)

model.save('model.h5')