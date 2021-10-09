import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
import db
import shutil

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import pathlib
from datetime import datetime



def moveFiles ():
    #cwd_path = os.path.join(os.getcwd(),'trainingdata/photos.zip')
    #full_path = os.path.abspath("./trainingdata/photos.zip")
    cwd_predict_path = os.path.join(os.getcwd(),'predict')    
    save_path = os.path.join(os.getcwd(),'predicted')
    data_dir = tf.keras.utils.get_file(cwd_predict_path,'file://' + cwd_predict_path)
    data_dir = pathlib.Path(data_dir)

    list_files = list(data_dir.glob('*.jpg'))
    image_count = len(list_files)
    #print(data_dir)
    #print(image_count)

    for i in range(image_count):
        #print(list_files[i])
        move_file = save_path + '\\' + os.path.basename(list_files[i])    
        pathlib.Path(list_files[i]).rename(move_file)
        #db.insert(move_file,move_file)
        #print('db done')


def copyDirectories (alreadyPredicted):

    trainingAll = os.path.join(os.getcwd(),'trainingdata\\photosall')
    trainingPartial = os.path.join(os.getcwd(),'trainingdata\\photospartial')
    if(os.path.exists(trainingPartial)):
        shutil.rmtree(trainingPartial, ignore_errors=True)
    #if not (os.path.exists(trainingPartial)):
        #os.mkdir(trainingPartial)
    
    #from distutils.dir_util import copy_tree
    #copy_tree(trainingAll, trainingPartial)
    
    shutil.copytree(trainingAll, trainingPartial)
    
    print("removing predicted class",alreadyPredicted)
    remove = os.path.join(trainingPartial,alreadyPredicted)    
    shutil.rmtree(remove, ignore_errors=False)
    
   


def readFiles():    
    predict_directory  = os.path.join(os.getcwd(),'predicted')
    numberOfFiles=0
    print(predict_directory)
    for filename in os.listdir(predict_directory):
        print(filename)
        numberOfFiles = numberOfFiles + 1
        print(numberOfFiles)

#copyDirectories('keys')
#readFiles()
#moveFiles()
