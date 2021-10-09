import matplotlib.pyplot as plt
import numpy as np
import files
import os
import db
import PIL
import tensorflow as tf
import ml
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
import pathlib
from datetime import datetime


AUTOTUNE = tf.data.AUTOTUNE

predict_url = os.path.join(os.getcwd(),'predict\IMG_20210727_213407.jpg')
predict_directory  = os.path.join(os.getcwd(),'predict')
numberOfFiles=0
print(predict_directory)
cwd_path_photos = os.path.join(os.getcwd(),'trainingdata/photosall')
cwd_path_photos_partial = os.path.join(os.getcwd(),'trainingdata/photospartial')


def predict():
    for filename in os.listdir(predict_directory):
        numberOfFiles=numberOfFiles + 1
      
        f = os.path.join(predict_directory, filename)  
        if os.path.isfile(f):
            predict_url = f
           
      
        data_dir = ml.trainPrep(cwd_path_photos)
        train_ds_all = ml.train(data_dir)
        val_ds_all = ml.validate(data_dir)
        class_names = train_ds_all.class_names
        print(class_names)

        train_ds_all = train_ds_all.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
        val_ds_all = val_ds_all.cache().prefetch(buffer_size=AUTOTUNE)

        model = ml.getModel()
        epochs=10
        history = model.fit(
          train_ds_all,
          validation_data=val_ds_all,      
          epochs=epochs,
          verbose=0
        )

        predicted_class = ml.predict(predict_url, model, class_names)
        imageId = db.insertImage(os.path.basename(predict_url),'')

        print(
                "FIRST:This image most likely belongs to {} with a {:.2f} percent confidence."
                .format(predicted_class[0],predicted_class[1])
        )

        db.insertObject(imageId, predicted_class[0], predicted_class[1])


        #LOOKING FOR SECOND OBJECT

        files.copyDirectories(predicted_class[0])

        data_dir_partial = ml.trainPrep(cwd_path_photos_partial)
        train_ds_partial = ml.train(data_dir_partial)
        val_ds_partial   = ml.validate(data_dir_partial)
        class_names =      train_ds_partial.class_names
        print(class_names)

        train_ds_partial = train_ds_partial.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
        val_ds_partial   = val_ds_partial.cache().prefetch(buffer_size=AUTOTUNE)

        model = ml.getModel()
        epochs=10
        history = model.fit(
          train_ds_partial,
          validation_data=val_ds_partial,
          epochs=epochs,
          verbose=0
        )

        predicted_class = ml.predict(predict_url, model, class_names)
        print(
                "SECOND:This image most likely belongs to {} with a {:.2f} percent confidence."
                .format(predicted_class[0],predicted_class[1])
        )
        db.insertObject(imageId, predicted_class[0], predicted_class[1])

    files.moveFiles()
    print('FilesProcessed:')
    print(numberOfFiles)



def menu():

    print_menu()
    user_input = 0
    
    while user_input != 4:
        
        user_input = int(input())

        if user_input == 1:
            print("Searching....")

        elif user_input == 2:
            print("Predicting....")
            try:
                predict()
            except:
                print("Prediction compelted with some errors!")

        elif user_input == 3:
            print("Training the model")        

        elif user_input == 4:
            print("Exiting...")
            exit(0)
        
        else:
            print("Invalid Input. Valid inputs are - (1/2/3/4)")
        


def print_menu():

    print("=====================",
"MENU",
    "=====================\n",
    "1 - Search Objects\n",
    "2 - Predict Objects\n",
    "3 - Train Model\n",    
    "4 - Exit\n",
    "============================================\n",
"Enter a choice and press enter:\n")


def testPredict():

    print("TEST PROGRAM----------------")    
    
    filename = os.listdir(predict_directory)[0]    
    f = os.path.join(predict_directory, filename)  
    if os.path.isfile(f):
        predict_url = f
        
    data_dir = ml.trainPrep(cwd_path_photos)
    train_ds_all = ml.train(data_dir)
    val_ds_all = ml.validate(data_dir)
    class_names = train_ds_all.class_names
    print(class_names)

    train_ds_all = train_ds_all.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds_all = val_ds_all.cache().prefetch(buffer_size=AUTOTUNE)

    model = ml.getModel()
    epochs=10
    history = model.fit(
      train_ds_all,
      validation_data=val_ds_all,      
      epochs=epochs,
      verbose=0
    )

    predicted_class,predicted_confidence = ml.predict(predict_url, model, class_names)      
    
    imageId = db.insertImage(os.path.basename(predict_url),'')
    db.insertObject(imageId, predicted_class[4], 100 * np.max(predicted_confidence[4]))
    db.insertObject(imageId, predicted_class[3], 100 * np.max(predicted_confidence[3]))
    
    print(
            "FIRST:This image most likely belongs to {} with a {:.2f} percent confidence."
            .format(predicted_class[4],100 * np.max(predicted_confidence[4])))
    print(
     "SECOND:This image second-most likely belongs to {} with a {:.2f} percent confidence."
            .format(predicted_class[3],100 * np.max(predicted_confidence[3])))
    
    db.showObjects('2 DAY')
   
def main():
    menu()
    #Menu
    testPredict()
    #predict()
    #print(db.showObjects('1 MINUTE'))
    
    
main()