# -*- coding: utf-8 -*-
"""Data_preprocessing_Hackerearth_Emotion_Detection_source_code.ipynb
"""

from tqdm import tqdm
import librosa
import numpy as np
import matplotlib.pyplot as plt
import audioread
import cv2
import librosa
import random,os,gc,sys

import warnings
warnings.filterwarnings('ignore')


audio_file_dir = "/content/train_audio"

files_dir = os.listdir(audio_file_dir)

#from https://www.kaggle.com/daisukelab/creating-fat2019-preprocessed-data
def mono_to_color(X, mean=None, std=None, norm_max=None, norm_min=None, eps=1e-6):
    # Stack X as [X,X,X]
#     X = np.stack([X, X, X], axis=-1)

    # Standardize
    mean = mean or X.mean()
    X = X - mean
    std = std or X.std()
    Xstd = X / (std + eps)
    _min, _max = Xstd.min(), Xstd.max()
    norm_max = norm_max or _max
    norm_min = norm_min or _min
    if (_max - _min) > eps:
        # Normalize to [0, 255]
        V = Xstd
        V[V < norm_min] = norm_min
        V[V > norm_max] = norm_max
        V = 255 * (V - norm_min) / (norm_max - norm_min)
        V = V.astype(np.uint8)
    else:
        # Just zero
        V = np.zeros_like(Xstd, dtype=np.uint8)
    return V

def build_spectrogram(path,srate=44100):
    y, sr = librosa.load(path,sr=srate)
    M = librosa.feature.melspectrogram(y=y, sr=srate)
    M = librosa.power_to_db(M)
    x1,y1 = M.shape
    temp_img = np.zeros((128,455))
    if y1<455:
        temp_img[0:x1,0:y1] = M[:]
    else:
        temp_img[:] = M[:128,0:455]
    img = mono_to_color(temp_img)
    return img


OUT_TRAIN = 'SpectogramsFolder.zip'

files = [str(audio_file_dir)+"/"+str(x) for x in files_dir]

x_tot,x2_tot = [],[]
batch = 20
with zipfile.ZipFile(OUT_TRAIN, 'w') as img_out:
    for idx in tqdm(range(0,len(files),batch)):
        names = files[idx:idx+batch]
        out = Parallel(n_jobs=-1)(delayed(build_spectrogram)(i) for i in names)
        for s in range(len(out)):
            img = out[s]
            x_tot.append((img/255.0).mean())
            x2_tot.append(((img/255.0)**2).mean()) 
            name = names[s].split('/')[-1].split('.')[0]
            img = cv2.imencode('.png',img)[1]
            img_out.writestr(name + '.png', img)

