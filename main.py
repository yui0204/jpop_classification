#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 23:29:14 2018

@author: yui-sudo
"""

import os
import datetime
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import pydot, graphviz
import pickle
from keras.optimizers import Adam, SGD
import keras.backend as K
from keras.callbacks import EarlyStopping
from keras.utils import plot_model

import CNN

from sound import WavfileOperate, Stft

import shutil


import tensorflow as tf
from keras.utils import multi_gpu_model

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = "0"
sess = tf.Session(config=config)
K.set_session(sess)


import soundfile as sf
from scipy import signal
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sound import WavfileOperate, Wavedata, Multiwave
import librosa


def normalize(inputs):
    inputs = inputs / inputs.max()
    inputs = np.clip(inputs, 0.0, 1.0)
    
    return inputs



def log(inputs):
    inputs += 10**-7
    inputs = 20 * np.log10(inputs)
    
    return inputs



def load(segdata_dir, n_classes=8, load_number=2911, complex_input=False):   
    print("data loading\n")
    if complex_input == True or VGG > 0:
        input_dim = 3
    else:
        input_dim = 1
        
    inputs = np.zeros((load_number, input_dim, 256, image_size), dtype=np.float32)
    #inputs = np.zeros((load_number, input_dim, 40, image_size), dtype=np.float32)
    labels = np.zeros((load_number, n_classes), dtype=np.float32)
    artist_list = os.listdir(segdata_dir)
    artist_list.sort()
    i = 0
    cls = 0
    for artist in artist_list:
        print(artist)
        album_list = os.listdir(segdata_dir + "/" + artist + "/")
        for album in album_list:
            print(album)
            filelist = os.listdir(segdata_dir + "/" + artist + "/" + album + "/")
            for file in filelist:
#                print(file)
                if file[-4:] == ".wav":
                    #wave = WavfileOperate(segdata_dir + "/" + artist + "/" + album + "/" + file).wavedata
                    #melspectrogram = librosa.feature.melspectrogram(y=wave.norm_sound, 
                    #                                        sr=wave.fs, n_fft=512, 
                    #                                        hop_length=512//2, n_mels=40)    
                    #melspectrogram = 10 * np.log10(melspectrogram)
                    #stft = librosa.feature.mfcc(y=wave.norm_sound, sr=wave.fs, 
                    #                     S = melspectrogram, n_mfcc=40)
                    waveform, fs = sf.read(segdata_dir + "/" + artist + "/" + album + "/" + file)
                    freqs, t, stft = signal.stft(x=waveform, fs=fs, nperseg=512, 
                                                           return_onesided=False)
                    stft = stft[:, 1:len(stft.T) - 1]
                    if len(stft.T) > 512:
                        stft = stft.T[:512].T
                    if complex_input == True:
                        inputs[i][1] = stft[:256].real
                        inputs[i][2] = stft[:256].imag
                    inputs[i][0] = abs(stft[:256])
                    #inputs[i][0] = stft
                    labels[i][cls] = 1
                    #print(artist, labels[i])
                    i += 1
        cls += 1

    
    if complex_input == True:
        sign = (inputs > 0) * 2 - 1
        sign = sign.astype(np.float16)
                    
 
    inputs = log(inputs)   
    inputs = np.nan_to_num(inputs)
    inputs += 120
    inputs = normalize(inputs)

    if complex_input == True:
        inputs = inputs * sign
            
    inputs = inputs.transpose(0, 2, 3, 1)
    
    if VGG > 0:
        inputs = inputs.transpose(3,0,1,2)
        if VGG == 1:
            inputs[1:3] = 0       # R only
        elif VGG == 3:
            inputs[1] = inputs[0]
            inputs[2] = inputs[0] # Grayscale to RGB
        inputs = inputs.transpose(1,2,3,0)
    
    return inputs, labels



def read_model(Model):
    with tf.device('/cpu:0'):
        if Model == "CNN":
            model = CNN.CNN(n_classes=classes, input_height=256, 
                                input_width=image_size, nChannels=1)

        elif Model == "CRNN8":
            model = CNN.CRNN(n_classes=classes, input_height=256, 
                                input_width=image_size, nChannels=1)

        elif Model == "BiCRNN8":
            model = CNN.BiCRNN8(n_classes=classes, input_height=256, 
                                input_width=image_size, nChannels=1)
        
    if gpu_count > 1:
        multi_model = multi_gpu_model(model, gpus=gpu_count)
    else:
        multi_model = model
        
    return model, multi_model



def train(X_train, Y_train, Model):
    model, multi_model = read_model(Model)
    
    if gpu_count == 1:
        model.compile(loss=loss, optimizer=Adam(lr=lr),metrics=["accuracy"])
    else:
        multi_model.compile(loss=loss, optimizer=Adam(lr=lr),metrics=["accuracy"])                

    plot_model(model, to_file = results_dir + model_name + '.png')

    early_stopping = EarlyStopping(monitor="val_loss", patience=20, verbose=1,mode="auto")
    
    model.summary()


    if complex_input == True:
        X_train = [X_train, 
                   X_train.transpose(3,0,1,2)[0][np.newaxis,:,:,:].transpose(1,2,3,0)]
    
    if gpu_count == 1:            
        history = model.fit(X_train, Y_train, batch_size=BATCH_SIZE, 
                            epochs=NUM_EPOCH, verbose=1, validation_split=0.0,
                            callbacks=[early_stopping])        
    else:       
        history = multi_model.fit(X_train, Y_train, batch_size=BATCH_SIZE, 
                            epochs=NUM_EPOCH, verbose=1, validation_split=0.0,
                            callbacks=[early_stopping])                 


    with open(results_dir + "history.pickle", mode="wb") as f:
        pickle.dump(history.history, f)

    model_json = model.to_json()
    with open(results_dir + "model.json", mode="w") as f:
        f.write(model_json)
    
    model.save_weights(results_dir + model_name + '_weights.hdf5')
    
    return history



def plot_history(history, model_name):
    """
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['acc', 'val_acc'], loc='lower right')
    plt.savefig(results_dir + model_name + "_accuracy.png")
    plt.close()
    """
    plt.plot(history.history['loss'])
#    plt.plot(history.history['val_loss'])
    plt.title('loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.xlim(0, 30)
    plt.ylim(0.0, 0.03)
#    plt.legend(['loss', 'val_loss'], loc='upper right')
#    plt.savefig(results_dir + "loss_"+str(np.array((history.history["val_loss"])).min())+".png")
    plt.close()


def predict(X_test, Model):
    model, multi_model = read_model(Model)
    model.load_weights(results_dir + model_name + '_weights.hdf5')

    print("\npredicting...")

    if complex_input == True:
        X_test = [X_test, 
                  X_test.transpose(3,0,1,2)[0][np.newaxis,:,:,:].transpose(1,2,3,0)]
    Y_pred = model.predict(X_test, BATCH_SIZE * gpu_count)
           
    print("prediction finished\n")
    
    return Y_pred



if __name__ == '__main__':
    classes = 3
    image_size = 512
        
    gpu_count = 1
    BATCH_SIZE = 16 * gpu_count
    NUM_EPOCH = 40
    
    lr = 0.0001
    
    loss = "categorical_crossentropy"

    mode = "2019_0619"
    date = mode       
    plot = True
    
    if os.getcwd() == '/home/yui-sudo/document/segmentation/jpop_classification':
        datasets_dir = "/home/yui-sudo/document/dataset/"
    else:
        datasets_dir = "/misc/export2/sudou/"
    
   
    datadir = "jpop_classification/"
    dataset = datasets_dir + datadir    
    segdata_dir = dataset + "train/"
    valdata_dir = dataset + "val/"
            
    Model = "CNN"        
    complex_input = False
    VGG = 0                     #0: False, 1: Red 3: White
    load_number = 1799
    
    model_name = Model+"_"+str(classes)+"class_cin"+str(complex_input)
    dir_name = model_name + "_"+datadir
    date = datetime.datetime.today().strftime("%Y_%m%d")
    results_dir = "./model_results/" + date + "/" + dir_name
    
    if mode == "train":
        print("\nTraining start...")
        if not os.path.exists(results_dir + "prediction"):
            os.makedirs(results_dir + "prediction/")
                            
        X_train, Y_train = load(segdata_dir, n_classes=classes, load_number=load_number,
                                complex_input=complex_input)

        p = np.random.permutation(load_number)
        X_train = X_train[p]
        Y_train = Y_train[p]
        

        print(X_train.shape, Y_train.shape)
        
        history = train(X_train, Y_train, Model)
        plot_history(history, model_name)


    # prediction            
    elif not mode == "train":
        print("Prediction\n")
        date = mode
        results_dir = "./model_results/" + date + "/" + dir_name

        
    load_number = 1127
    """
    X_test, Y_test = load(valdata_dir, n_classes=classes, load_number=load_number, 
                          complex_input=complex_input)
    Y_pred = predict(X_test, Model)
    print(Y_pred)
    Y_pred = np.argmax(Y_pred, axis=1)[:, np.newaxis]
    Y_test = np.argmax(Y_test, axis=1)[:, np.newaxis]
    """
    print(accuracy_score(Y_test, Y_pred))
    print(confusion_matrix(Y_test, Y_pred))
    
    with open(results_dir + 'result.txt','w') as f:
        result = ""
        artist_list = os.listdir(segdata_dir)
        artist_list.sort()
        i = 0
        cls = 0
        for artist in artist_list:
            print(artist)
            album_list = os.listdir(valdata_dir + "/" + artist + "/")
            for album in album_list:
                print(album)
                filelist = os.listdir(valdata_dir + "/" + artist + "/" + album + "/")
                for file in filelist:
                    print(artist, file, Y_test[i] == Y_pred[i])
                    if not Y_test[i] == Y_pred[i]:
                        result += artist + ", " +  file + ", " +  str(Y_test[i])+ str(Y_pred[i])+str(Y_test[i] == Y_pred[i]) + "\n"
                    i += 1
        f.write(result + "\n" + str(accuracy_score(Y_test, Y_pred)) + "\n" + str(confusion_matrix(Y_test, Y_pred)))
        
        
        
    if not os.getcwd() == '/home/yui-sudo/document/segmentation/sound_segtest':
        shutil.copy("main.py", results_dir)
        shutil.copy("CNN.py", results_dir)

        # copy to export2
#        shutil.copytree(results_dir, "/misc/export2/sudou/model_results/" + date + "/" + dir_name)
                                
    #os.remove("CNN.pyc")
    #os.remove("sound.pyc")
