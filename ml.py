import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import pathlib
from datetime import datetime


batch_size = 32
img_height = 180
img_width = 180
num_classes = 5




def train(data_dir):
    data_dir = pathlib.Path(data_dir)
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,   
    image_size=(img_height, img_width),
    batch_size=batch_size)    
    return train_ds

def validate(data_dir):
    data_dir = pathlib.Path(data_dir)
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)
    return val_ds


def getModel():
    model = Sequential([
      layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
      layers.Conv2D(16, 3, padding='same', activation='relu'),
      layers.MaxPooling2D(),
      layers.Conv2D(32, 3, padding='same', activation='relu'),
      layers.MaxPooling2D(),
      layers.Conv2D(64, 3, padding='same', activation='relu'),
      layers.MaxPooling2D(),
      layers.Flatten(),
      layers.Dense(128, activation='relu'),
      layers.Dense(num_classes)
    ])
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    #model.summary()
    return model
    

def predictNew(predict_url, model, class_names):

    to_predict_file_path = tf.keras.utils.get_file(predict_url, origin=predict_url)
    img = keras.preprocessing.image.load_img(
    to_predict_file_path, target_size=(img_height, img_width)
    )    
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch
    predictions = model.predict(img_array)
    print("predictions:",predictions[0])
    
    score = tf.nn.softmax(predictions[0])
    array = np.array(predictions[0])
    sorted_index = np.argsort(array)
    sorted_array = array[sorted_index]
    
    print("predictions:",predictions[0])
    print("sortedindex",sorted_index)
    print("sortedarray:",sorted_array)
    
    return class_names[np.argmax(score)], 100 * np.max(score)




def predict(predict_url, model, class_names):

    to_predict_file_path = tf.keras.utils.get_file(predict_url, origin=predict_url)
    img = keras.preprocessing.image.load_img(
    to_predict_file_path, target_size=(img_height, img_width)
    )    
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch
    predictions = model.predict(img_array)
    print("predictions:",predictions[0])
    
    score = tf.nn.softmax(predictions[0])
    print("score:",score)

    
                                    
    prediction_array = np.array(predictions[0])
    classes_array = np.array(class_names)
    scores_array = np.array(score)
    
    probabilities_sorted_index = np.argsort(scores_array)
    
    prediction_array_sorted = scores_array[probabilities_sorted_index]
    classes_array_sorted = classes_array[probabilities_sorted_index]
        
    print("prediction_array_sorted",prediction_array_sorted)
    print("classes_array_sorted:",classes_array_sorted)
    #return class_names[np.argmax(score)], 100 * np.max(score)
    return classes_array_sorted, prediction_array_sorted
    
    


def trainPrep(cwd_path_photos):
    data_dir = tf.keras.utils.get_file(cwd_path_photos,'file://' + cwd_path_photos)
    data_dir = pathlib.Path(data_dir)
    print("from",data_dir)
    image_count = len(list(data_dir.glob('*/*.jpg')))
    print("images for training",image_count)
    #now = datetime.now()
    #current_time = now.strftime("%H:%M:%S")
    #print("Current Time", current_time)
    return data_dir