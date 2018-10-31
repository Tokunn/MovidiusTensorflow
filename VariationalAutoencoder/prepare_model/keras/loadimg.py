#!/usr/bin/env python
# coding: utf-8


import os, glob
import numpy as np

from tqdm import tqdm

from keras.preprocessing.image import load_img, img_to_array
from keras.utils import np_utils
from sklearn.model_selection import train_test_split


def write_category(dirs_list, output_catetxt):
    if (output_catetxt):
        with open('categories.txt', 'w') as f:
            f.write('\n'.join(img_list))
            f.write('\n')

def get_dirlist(dirpath):
    files = os.listdir(dirpath)
    files_dir = [f for f in files if os.path.isdir(os.path.join(dirpath, f))]
    return sorted(files_dir)

def load_images_from_onefolder(dirpath, imgsize_x, imgsize_y, grayscale):
    print("dirpath", dirpath, flush=True)
    imgName_list = glob.glob(os.path.join(dirpath, '*.jpg'))
    imgName_list += glob.glob(os.path.join(dirpath, '*.JPG'))
    imgName_list += glob.glob(os.path.join(dirpath, '*.png'))
    imgName_list += glob.glob(os.path.join(dirpath, '*.PNG'))
    x = []
    for imgName in tqdm(imgName_list):
        img = load_img(imgName, target_size=(imgsize_y, imgsize_x), grayscale=grayscale)
        img = img_to_array(img) # Numpy list
        x.append(img)
    return x # Python list

def load_all_images(dirpath, imgsize_x, imgsize_y, isdirs, output_catetxt, grayscale):
    if (isdirs):
        dirs_list = get_dirlist(dirpath)
        write_category(dirs_list, output_catetxt)
        dirs_list = [os.path.join(dirpath, d) for d in dirs_list]
    else:
        dirs_list = [dirpath]
        
    n_class = len(dirs_list)
    
    x = []
    y = []
    y_label = 0
    for current_dirpath in dirs_list:
        x_tmp = load_images_from_onefolder(current_dirpath, imgsize_x, imgsize_y, grayscale)
        x += x_tmp # Python list + Python list
        y += [y_label] * len(x_tmp) # Python list + Python list
        y_label += 1
        
    return np.asarray(x), np.asarray(y), n_class

def loadimg(
    dirpath,
    imgsize=None, imgsize_x=160, imgsize_y=160,
    train_ratio=0.8,
    isdirs=True,
    normalize=True,
    onehot=True,
    output_catetxt=False,
    grayscale=False):
    
    if not (imgsize == None):
        imgsize_x = imgsize
        imgsize_y = imgsize
    
    x, y, n_class = load_all_images(dirpath, imgsize_x, imgsize_y, isdirs, output_catetxt, grayscale)
    
    if normalize:
        x = x/255.0
    if onehot:
        y = np_utils.to_categorical(y, n_class)
        
    if (train_ratio==1): 
        x_train = x
        y_train = y
        x_test = []
        y_test = []
    elif (train_ratio==0):
        x_train = []
        y_train = []
        x_test = x
        y_test = y
    else:
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_ratio, test_size=1-train_ratio)
        
        
    return x_train, x_test, y_train, y_test, n_class

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    x_train, x_test, y_train, y_test, n_class = loadimg('images/AnomalyDetectionPictures',
                                                       imgsize=160,
                                                       isdirs=True,
                                                       normalize=True,
                                                       onehot=True,
                                                       grayscale=True)
    print("Shape", x_train.shape)
    for x,y in zip(x_train, y_train):
        #plt.imshow(x)
        #plt.show()
        print(np.argmax(y))


