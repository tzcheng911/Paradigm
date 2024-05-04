#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 12:31:26 2023

Decoding for 
1. Native vs. Nonnative MMR
2. cABR to three speech sounds

Permutation test for statistical significance.
 
@author: tzcheng
""" 
import numpy as np
import copy
import random
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from mne.decoding import (
    cross_val_multiscore,
    LinearModel,
    get_coef,
)
#%%####################################### decoding for single channel EEG
root_path='/home/tzcheng/Documents/GitHub/Paradigm/'

#%% MMR
times = np.linspace(-0.1,0.6,3501) # For MMR
ts = 1000
te = 1750

last_ba = np.load(root_path + 'group_std_eeg.npy')
last_mba = np.load(root_path + 'group_std1_reverse_eeg.npy')
last_pa = np.load(root_path + 'group_std2_reverse_eeg.npy')
first_mba = np.load(root_path + 'group_dev1_eeg.npy')
first_pa = np.load(root_path + 'group_dev2_eeg.npy')

## Conventional calculation
MMR1 = first_mba - last_ba
MMR2 = first_pa - last_ba

## Controlled calculation
MMR1 = first_mba - last_mba
MMR2 = first_pa - last_pa

## preserve the subject MMR1 and MMR2 relationship but randomize the order across subjects
rand_ind = np.arange(0,len(MMR1))
random.Random(2).shuffle(rand_ind)
X = np.concatenate((MMR1[rand_ind,:],MMR2[rand_ind,:]),axis=0)
X = X[:,ts:te]
y = np.concatenate((np.repeat(0,len(MMR1)),np.repeat(1,len(MMR2))))

clf = make_pipeline(
    StandardScaler(),  # z-score normalization
    LogisticRegression(solver="liblinear")  # liblinear is faster than lbfgs
)
scores = cross_val_multiscore(clf, X, y, cv=5, n_jobs=None) # takes about 10 mins to run
score = np.mean(scores, axis=0)
print("Accuracy: %0.1f%%" % (100 * score,))

## complete randomization
X = np.concatenate((MMR1,MMR2),axis=0)
X = X[:,ts:te]
y = np.concatenate((np.repeat(0,len(MMR1)),np.repeat(1,len(MMR2))))
rand_ind = np.arange(0,len(X))
random.Random(2).shuffle(rand_ind)
X = X[rand_ind,:]
y = y[rand_ind]
    
clf = make_pipeline(
    StandardScaler(),  # z-score normalization
    LogisticRegression(solver="liblinear")  # liblinear is faster than lbfgs
)
scores = cross_val_multiscore(clf, X, y, cv=5, n_jobs=None) # takes about 10 mins to run
score = np.mean(scores, axis=0)
print("Accuracy: %0.1f%%" % (100 * score,))

#%% FFR 
std = np.load(root_path + 'group_std_ffr_eeg_200.npy')
dev1 = np.load(root_path + 'group_dev1_ffr_eeg_200.npy')
dev2 = np.load(root_path + 'group_dev2_ffr_eeg_200.npy')

X = np.concatenate((std,dev1,dev2),axis=0)
y = np.concatenate((np.repeat(0,len(std)),np.repeat(1,len(dev1)),np.repeat(2,len(dev2))))

rand_ind = np.arange(0,len(X))
random.Random(2).shuffle(rand_ind)
X = X[rand_ind,:]
y = y[rand_ind]

clf = make_pipeline(
    StandardScaler(),  # z-score normalization
    LogisticRegression(solver="liblinear")  # liblinear is faster than lbfgs
)
scores = cross_val_multiscore(clf, X, y, cv=5, n_jobs=None) 
score = np.mean(scores, axis=0)
print("Accuracy: %0.1f%%" % (100 * score,))

## Run permutation on MMR or FFR
n_perm=5000
scores_perm=[]
for i in range(n_perm):
    yp = copy.deepcopy(y)
    random.shuffle(yp)
    clf = make_pipeline(
        StandardScaler(),  # z-score normalization
        LogisticRegression(solver="liblinear")  
        )
    # Run cross-validated decoding analyses:
    scores = cross_val_multiscore(clf, X, yp, cv=5, n_jobs=None) # X can be MMR or cABR
    scores_perm.append(np.mean(scores,axis=0))
    print("Iteration " + str(i))
scores_perm_array=np.asarray(scores_perm)

plt.figure()
plt.hist(scores_perm_array,bins=15,color='grey')
plt.vlines(score,ymin=0,ymax=500,color='r',linewidth=2)
plt.vlines(np.percentile(scores_perm_array,95),ymin=0,ymax=500,color='grey',linewidth=2)
plt.ylabel('Count',fontsize=20)
plt.xlabel('Accuracy',fontsize=20)
