#!/usr/bin/env python2

import os
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import load_img, img_to_array
from keras.utils import np_utils
import matplotlib.pyplot as plt
import glob
from sklearn.model_selection import train_test_split


#IMGSIZE = 224
IMGSIZE = 160

def loadimg_one(DIRPATH, NUM):
    x = []
    y = []

    img_list = os.listdir(DIRPATH)
    img_list = sorted(img_list)
    if (NUM) and (len(img_list) > NUM):
        img_list = img_list[:NUM]
    #print("[loadimg] : img_list : ", end=' ')
    #print(img_list)
    
    with open('categories.txt', 'w') as f:
        f.write('\n'.join(img_list))
        f.write('\n')

    img_count = 0

    for number in img_list:
        dirpath = os.path.join(DIRPATH, number)
        dirpic_list = glob.glob(os.path.join(dirpath, '*.jpg'))
        dirpic_list += glob.glob(os.path.join(dirpath, '*.png'))
        for picture in dirpic_list:
            #img = img_to_array(load_img(picture, color_mode = "grayscale", target_size=(IMGSIZE, IMGSIZE)))
            img = img_to_array(load_img(picture, target_size=(IMGSIZE, IMGSIZE)))
            x.append(img)
            y.append(img_count)
            #print("Load {0} : {1}".format(picture, img_count))
        img_count += 1

    output_count = img_count
    x = np.asarray(x)
    x = x.astype('float32')
    x = x/255.0
    y = np.asarray(y, dtype=np.int32)
    y = np_utils.to_categorical(y, output_count)

    return x, y, output_count


def loadimg(COMMONDIR='./', NUM=None):
    print("########## loadimg ########")

    #COMMONDIR = './make_image'
    #TRAINDIR = os.path.join(COMMONDIR, 'train')
    #TESTDIR = os.path.join(COMMONDIR, 'test')
    x, y, class_count = loadimg_one(COMMONDIR, NUM)
    #x_test,  y_test,  _  = loadimg_one(TESTDIR, NUM)
    #for i in range(0, x_test.shape[0]):
    #    plt.imshow(x_test[i])
    #    plt.show()
    #x = np.concatenate((x_train, x_test))
    #x = np.reshape(x, [-1, 784])
    #y = np.concatenate((y_train, y_test)) 

    print("x_train, y_train, x_test, y_test, class_count")
    print("x_train shape : ", x.shape)

    print("########## END of loadimg ########")
    x_train, x_test, y_train, y_test = train_test_split(x, y,train_size=0.8, test_size=0.2)
    return x_train,  y_train, x_test, y_test, class_count

if __name__ == '__main__':
    loadimg()
