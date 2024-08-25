#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 12:51:22 2024
Visualization for Figure 2
@author: tzcheng
"""
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt 
from scipy.io import wavfile

def plot_err(data,color,t):
    group_avg=np.mean(data,axis=0)
   #plt.figure()
    err=np.std(data,axis=0)/np.sqrt(data.shape[0])
    up=group_avg+err
    lw=group_avg-err
    plt.plot(t,group_avg,color=color)
    plt.fill_between(t,up,lw,color=color,alpha=0.5)

#%%####################################### visualize MMR
root_path='/home/tzcheng/Documents/GitHub/Paper0_Paradigm/'
times = np.linspace(-0.1,0.6,3501)

## first run
last_ba = np.load(root_path + 'group_std_eeg.npy')
last_mba = np.load(root_path + 'group_std1_reverse_eeg.npy')
last_pa = np.load(root_path + 'group_std2_reverse_eeg.npy')
first_mba = np.load(root_path + 'group_dev1_eeg.npy')
first_pa = np.load(root_path + 'group_dev2_eeg.npy')

## second run
last_ba = np.load(root_path + 'group_02_std_eeg.npy')
last_mba = np.load(root_path + 'group_02_std1_reverse_eeg.npy')
last_pa = np.load(root_path + 'group_02_std2_reverse_eeg.npy')
first_mba = np.load(root_path + 'group_02_dev1_eeg.npy')
first_pa = np.load(root_path + 'group_02_dev2_eeg.npy')

## Conventional calculation
MMR1 = first_mba - last_ba
MMR2 = first_pa - last_ba

## Controlled calculation
MMR1 = first_mba - last_mba
MMR2 = first_pa - last_pa

plt.figure()
plot_err(MMR1*1e6,'r',times)
plot_err(MMR2*1e6,'b',times)

plt.title('MMRs')
plt.legend(['Nonative MMR','','Native MMR',''])
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (uV)')
plt.xlim([-0.05,0.45])
plt.ylim([-5,4])

## Individual subjects
fig, axs = plt.subplots(18)
fig.suptitle('MMR')
for n in range(18):
    axs[n].plot(times, MMR1[n,:]*1e6,'r')
    axs[n].plot(times, MMR2[n,:]*1e6,'b')
    axs[n].set_xlim([-0.05,0.45])
    axs[n].set_ylim([-13, 13])

#%%####################################### visualize audio and cABR
root_path='/home/tzcheng/Documents/GitHub/Paper0_Paradigm/'
times = np.linspace(-0.02,0.2,1101)

std = np.load(root_path + 'group_std_ffr_eeg_200.npy')
dev1 = np.load(root_path + 'group_dev1_ffr_eeg_200.npy')
dev2 = np.load(root_path + 'group_dev2_ffr_eeg_200.npy')

## second run
std = np.load(root_path + 'group_02_std_ffr_eeg_150.npy')
dev1 = np.load(root_path + 'group_02_dev1_ffr_eeg_150.npy')
dev2 = np.load(root_path + 'group_02_dev2_ffr_eeg_150.npy')

## Audio files
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

## plot std, dev1, dev2
plt.figure()
plot_err(std*1e6,'k',times)
plt.xlim([-0.02,0.2])
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (uV)')
plt.figure()
plt.plot(np.linspace(0,0.1,500),std_audio)
plt.xlim([0,0.1])

plt.figure()
plot_err(dev1*1e6,'k',times)
plt.xlim([-0.02,0.2])
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (uV)')
plt.figure()
plt.plot(np.linspace(0,0.13,650),dev1_audio)
plt.xlim([0,0.13])

plt.figure()
plot_err(dev2*1e6,'k',times)
plt.xlim([-0.02,0.2])
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (uV)')
plt.figure()
plt.plot(np.linspace(0,0.13,650),dev2_audio)
plt.xlim([0,0.13])

## Individual subjects
fig, axs = plt.subplots(18)
fig.suptitle('FFR ba')
for n in range(18):
    axs[n].plot(times, std[n,:]*1e6,'k')
    axs[n].set_xlim([-0.02,0.2])
    axs[n].set_ylim([-1.2, 1.2])

fig, axs = plt.subplots(18)
fig.suptitle('FFR mba')
for n in range(18):
    axs[n].plot(times, dev1[n,:]*1e6,'k')
    axs[n].set_xlim([-0.02,0.2])
    axs[n].set_ylim([-1.2, 1.2])
    
fig, axs = plt.subplots(18)
fig.suptitle('FFR pa')
for n in range(18):
    axs[n].plot(times, dev2[n,:]*1e6,'k')
    axs[n].set_xlim([-0.02,0.2])
    axs[n].set_ylim([-1.2, 1.2])