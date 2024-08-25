#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 12:35:28 2024
SNR analysis for FFR in time and frequency spectra

@author: tzcheng
"""

import mne
import matplotlib.pyplot as plt 
import numpy as np

def plot_err(group_stc,color,t):
    group_avg=np.mean(group_stc,axis=0)
   #plt.figure()
    err=np.std(group_stc,axis=0)/np.sqrt(group_stc.shape[0])
    up=group_avg+err
    lw=group_avg-err
    plt.plot(t,group_avg,color=color)
    plt.fill_between(t,up,lw,color=color,alpha=0.5)
    
#%%####################################### SNR for single channel EEG
root_path='/home/tzcheng/Documents/GitHub/Paper0_Paradigm/'
times = np.linspace(-0.02,0.2,1101) # For FFR

std = np.load(root_path + 'group_std_ffr_eeg_200.npy')
dev1 = np.load(root_path + 'group_dev1_ffr_eeg_200.npy')
dev2 = np.load(root_path + 'group_dev2_ffr_eeg_200.npy')

## second run
std = np.load(root_path + 'group_02_std_ffr_eeg_150.npy')
dev1 = np.load(root_path + 'group_02_dev1_ffr_eeg_150.npy')
dev2 = np.load(root_path + 'group_02_dev2_ffr_eeg_150.npy')

#%%####################################### SNR analysis on the time domain
EEG = dev2
ind_noise = np.where(times<0)
ind_signal = np.where(np.logical_and(times>=0, times<=0.2)) # 0.1 for ba and 0.13 for mba and pa

## Group
rms_noise_s = np.sqrt(np.mean(EEG.mean(0)[ind_noise]**2))
rms_signal_s = np.sqrt(np.mean(EEG.mean(0)[ind_signal]**2))
SNR = rms_signal_s/rms_noise_s
dB_SNR = 20*np.log10(SNR)
print(SNR)
print(dB_SNR)

## Individual
rms_noise_s = []
rms_signal_s = []

for s in range(len(EEG)):
    rms_noise_s.append(np.sqrt(np.mean(EEG[s,ind_noise]**2)))
    rms_signal_s.append(np.sqrt(np.mean(EEG[s,ind_signal]**2)))
SNR = np.array(rms_signal_s)/np.array(rms_noise_s)
dB_SNR_mean = 20*np.log10(SNR.mean())
dB_SNR_se = 20*np.log10(SNR.std()/np.sqrt(len(EEG)))

print('SNR: ' + str(np.array(SNR).mean()) + '(' + str(np.array(SNR).std()/np.sqrt(len(EEG))) +')')
print('dB SNR: ' + str(dB_SNR_mean) + '(' + str(dB_SNR_se) +')')
dB_SNR = 20*np.log10(SNR)
            
#%%####################################### SNR analysis on the frequency domain
fmin = 50
fmax = 150
sfreq = 5000
total_len = len(times)

EEG = dev2

## Group
psds_noise, freqs_noise = mne.time_frequency.psd_array_welch(
    EEG.mean(0)[ind_noise],sfreq, # could replace with label time series
    n_fft=total_len,
    n_overlap=0,
    n_per_seg=total_len,
    fmin=fmin,
    fmax=fmax,)

psds_signal, freqs_signal = mne.time_frequency.psd_array_welch(
    EEG.mean(0)[ind_signal],sfreq, # could replace with label time series
    n_fft=total_len,
    n_overlap=0,
    n_per_seg=total_len,
    fmin=fmin,
    fmax=fmax,)

SNR = np.mean(psds_signal[8:10])/np.mean(psds_noise[8:10]) # find the closest peak to the audio ~90 Hz
print('SNR: ' + str(SNR) +' at ' + str(freqs_noise[8:10]) + ' Hz')

## Individual
psds_noise, freqs_noise = mne.time_frequency.psd_array_welch(
    np.squeeze(EEG[:,ind_noise]),sfreq, # could replace with label time series
    n_fft=total_len,
    n_overlap=0,
    n_per_seg=total_len,
    fmin=fmin,
    fmax=fmax,)
psds_signal, freqs_signal = mne.time_frequency.psd_array_welch(
    np.squeeze(EEG[:,ind_signal]),sfreq, # could replace with label time series
    n_fft=total_len,
    n_overlap=0,
    n_per_seg=total_len,
    fmin=fmin,
    fmax=fmax,)

SNR = np.mean(psds_signal[:,8:10],axis=1)/np.mean(psds_noise[:,8:10],axis=1) # find the closest peak to the audio ~90 Hz
# dB_SNR = 10*np.log10(SNR)
print('SNR: ' + str(np.array(SNR).mean()) + '(' + str(np.array(SNR).std()/np.sqrt(len(EEG))) +')' +' at ' + str(freqs_noise[8:10]) + 'Hz')

plt.figure()
plot_err(psds_noise*1e12,'grey',freqs_noise) # 1V  = 1e6 uV
plot_err(psds_signal*1e12,'k',freqs_signal)
plt.xlim([60, 140])
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power (uV^2)')
plt.legend(['Noise','','Signal',''])
plt.title('pa')