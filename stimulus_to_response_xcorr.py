#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 13:23:58 2023
Calculate EEG & audio correlation in adults for FFR.
Correlation methods: xcorr

@author: tzcheng
"""

import mne
import numpy as np
from numpy import dot
from numpy.linalg import norm
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy import stats,signal
import scipy as sp
import os
import seaborn as sns
import pandas as pd
import scipy.stats as stats
from scipy.io import wavfile

#%%######################################## xcorr between FFR and audio
root_path='/home/tzcheng/Documents/GitHub/Paper0_Paradigm/'

## Load FFR from 0
std = np.load(root_path + 'group_std_ffr_eeg_200.npy')[:,100:]
dev1 = np.load(root_path + 'group_dev1_ffr_eeg_200.npy')[:,100:]
dev2 = np.load(root_path + 'group_dev2_ffr_eeg_200.npy')[:,100:]

## second run
std = np.load(root_path + 'group_02_std_ffr_eeg_150.npy')[:,100:] # use 150 or all trials
dev1 = np.load(root_path + 'group_02_dev1_ffr_eeg_150.npy')[:,100:]
dev2 = np.load(root_path + 'group_02_dev2_ffr_eeg_150.npy')[:,100:]

# plt.figure()
# plt.plot(np.linspace(0,0.13,650),dev2_audio_r)

## Load real audio
fs, std_audio = wavfile.read(root_path + '+10.wav')
fs, dev1_audio = wavfile.read(root_path + '-40.wav')
fs, dev2_audio = wavfile.read(root_path + '+40.wav')
# Downsample
fs_new = 5000
num_std = int((len(std_audio)*fs_new)/fs)
num_dev = int((len(dev2_audio)*fs_new)/fs)  # #sample_new/fs_new=#sample/fs find number of samples in the resampled data
            
std_audio = signal.resample(std_audio, num_std, t=None, axis=0, window=None)
dev1_audio = signal.resample(dev1_audio, num_dev, t=None, axis=0, window=None)
dev2_audio = signal.resample(dev2_audio, num_dev, t=None, axis=0, window=None)

## Change audio0 and EEG0 to corresponding std, dev1, dev2
# Run the corresponding code section below
audio0 = std_audio
EEG0 = std
times0_audio = np.linspace(0,len(audio0)/fs_new,len(audio0))
times0_eeg = np.linspace(0,len(EEG0[0])/fs_new,len(EEG0[0]))

## std: noise burst from 0 ms 
ts = 100 # 0.02s (i.e. 0.02s after noise burst)
te = 500 # 0.1s
audio = audio0[ts:te] # try 0.02 to 0.1 s for std
EEG = EEG0[:,ts:te]
times_audio = times0_audio[ts:te]
times_eeg = times0_eeg[ts:te]
lags = signal.correlation_lags(len(audio),len(EEG[0]))
lags_s = lags/fs_new

## dev 1: noise burst from 40 ms (200th points)
ts = 200 + 100 # .06s (i.e. 0.02s after noise burst)
te = 650 # 0.13s
audio = audio0[ts:te] # try 0.042 to 0.13 s for dev2
EEG = EEG0[:,ts:te]
times_audio = times0_audio[ts:te]
times_eeg = times0_eeg[ts:te]
lags = signal.correlation_lags(len(audio),len(EEG[0]))
lags_s = lags/fs_new

## dev 2: noise burst from 0 ms (100th points)
ts = 100 # 0.02s
te = 650 # 0.13s
audio = audio0[ts:te] # try 0.02 to 0.13 s for dev2
EEG = EEG0[:,ts:te]
times_audio = times0_audio[ts:te]
times_eeg = times0_eeg[ts:te]
lags = signal.correlation_lags(len(audio),len(EEG[0]))
lags_s = lags/fs_new

## select the lag window to be between 7 to 14 ms
lag_window = [-0.007,-0.014]
lag_window_ind = np.where((lags_s<=lag_window[0]) & (lags_s>=lag_window[1]))

a = (audio - np.mean(audio))/np.std(audio)
a = a / np.linalg.norm(a)

## For grand average: a[n] is lagging behind b[n] by k sample periods
b = (EEG.mean(axis=0) - np.mean(EEG.mean(axis=0)))/np.std(EEG.mean(axis=0))
b = b / np.linalg.norm(b)
xcorr = signal.correlate(a,b,mode='full')
xcorr = abs(xcorr)
# xcorr_max = max(xcorr)
# xcorr_maxlag = np.argmax(xcorr)
# print("max lag: ", str(lags_s[xcorr_maxlag]*1000))

xcorr_max = max(xcorr[lag_window_ind]) # only select the xcorr from the time window shift
xcorr_maxlag = np.argmax(xcorr[lag_window_ind])
print("max lag: ", str(lags_s[lag_window_ind][xcorr_maxlag]*1000)) # only select the xcorr from the time window shift
print("max xcorr: ", str(xcorr_max))

## For each individual
xcorr_all_s = []
xcorr_lag_all_s = []

for s in np.arange(0,len(std),1):
    b = (EEG[s,:] - np.mean(EEG[s,:]))/np.std(EEG[s,:])
    
    ## the way matlab do xcorr normalization: the max is 1 if do a zero lag autocorrealtoin
    b = b / np.linalg.norm(b)

    xcorr = signal.correlate(a,b,mode='full')
    xcorr = abs(xcorr)
    xcorr_all_s.append(np.max(xcorr[lag_window_ind]))
    xcorr_lag_all_s.append(np.argmax(xcorr[lag_window_ind]))
    # xcorr_all_s.append(np.max(xcorr))
    # xcorr_lag_all_s.append(np.argmax(xcorr))

# print('abs xcorr between FFR & audio: ' + str(np.array(xcorr_all_s).mean()) + '(' + str(np.array(xcorr_all_s).std()/np.sqrt(len(std))) +')')
# print('abs xcorr lag between FFR & audio (ms): ' + str(np.array(lags_s[xcorr_lag_all_s]*1000).mean())+ '(' + str(np.array(lags_s[xcorr_lag_all_s]*1000).std()/np.sqrt(len(std))) +')')
# arr = np.array(xcorr_all_s)
# rounded_arr = np.around(arr, decimals=3)
# print(rounded_arr)
# print(lags_s[xcorr_lag_all_s]*1000)

print('abs xcorr between FFR & audio: ' + str(np.array(xcorr_all_s).mean()) + '(' + str(np.array(xcorr_all_s).std()/np.sqrt(len(std))) +')')
print('abs xcorr lag between FFR & audio (ms): ' + str(np.array(lags_s[lag_window_ind][xcorr_lag_all_s]*1000).mean())+ '(' + str(np.array(lags_s[lag_window_ind][xcorr_lag_all_s]*1000).std()/np.sqrt(len(std))) +')')
arr = np.array(xcorr_all_s)
rounded_arr = np.around(arr, decimals=3)
print(rounded_arr)
print(lags_s[lag_window_ind][xcorr_lag_all_s]*1000)