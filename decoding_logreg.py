#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 12:31:26 2023

Decoding for 
1. Native vs. Nonnative MMR
2. FFR to three speech sounds

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
from sklearn.svm import SVC

from mne.decoding import (
    cross_val_multiscore,
    LinearModel,
    get_coef,
)
#%%####################################### decoding for single channel EEG
root_path='/home/tzcheng/Documents/GitHub/Paper0_Paradigm/'

#%% MMR
times = np.linspace(-0.1,0.6,3501) # For MMR
# times = np.linspace(-0.02,0.2,1101) # For FFR

ts = 1000
te = 1750

## first run
# last_ba = np.load(root_path + 'group_std_eeg.npy')
# last_mba = np.load(root_path + 'group_std1_reverse_eeg.npy')
# last_pa = np.load(root_path + 'group_std2_reverse_eeg.npy')
# first_mba = np.load(root_path + 'group_dev1_eeg.npy')
# first_pa = np.load(root_path + 'group_dev2_eeg.npy')

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
## first run
std = np.load(root_path + 'group_std_ffr_eeg_200.npy')
dev1 = np.load(root_path + 'group_dev1_ffr_eeg_200.npy')
dev2 = np.load(root_path + 'group_dev2_ffr_eeg_200.npy')

## second run
# std = np.load(root_path + 'group_02_std_ffr_eeg_150.npy') # use 150 or all trials
# dev1 = np.load(root_path + 'group_02_dev1_ffr_eeg_150.npy')
# dev2 = np.load(root_path + 'group_02_dev2_ffr_eeg_150.npy')

X = np.concatenate((std,dev1,dev2),axis=0)
y = np.concatenate((np.repeat(0,len(std)),np.repeat(1,len(dev1)),np.repeat(2,len(dev2))))

## preserve the subject ba, mba, pa relationship but randomize the order across subjects
rand_ind = np.arange(0,len(std))
random.Random(2).shuffle(rand_ind)
X = np.concatenate((std[rand_ind,:],dev1[rand_ind,:],dev2[rand_ind,:]),axis=0)

clf = make_pipeline(
    StandardScaler(),  # z-score normalization
    LogisticRegression(solver="liblinear")  # liblinear is faster than lbfgs
)
scores = cross_val_multiscore(clf, X, y, cv=5, n_jobs=None) # takes about 10 mins to run
score = np.mean(scores, axis=0)
print("Accuracy: %0.1f%%" % (100 * score,))

## complete randomization
rand_ind = np.arange(0,len(X))
random.Random(2).shuffle(rand_ind)
X = X[rand_ind,:]
y = y[rand_ind]

clf = make_pipeline(
    StandardScaler(),  # z-score normalization
    LogisticRegression(solver="liblinear")  # liblinear is faster than lbfgs
)

clf = make_pipeline(
    StandardScaler(),  # z-score normalization
    SVC(kernel='rbf',gamma='auto')  
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

#%%
import mne
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import StratifiedKFold

## load my emr recording
std_unplug = mne.read_evokeds('cbs_zoe_unplug_evoked_substd_cabr_200.fif',allow_maxshield=True)[0]
dev1_unplug = mne.read_evokeds('cbs_zoe_unplug_evoked_dev1_cabr_200.fif',allow_maxshield=True)[0]
dev2_unplug = mne.read_evokeds('cbs_zoe_unplug_evoked_dev2_cabr_200.fif',allow_maxshield=True)[0]

std_noise = np.squeeze(std_unplug.get_data())
dev1_noise = np.squeeze(dev1_unplug.get_data())
dev2_noise = np.squeeze(dev2_unplug.get_data())

std = np.load(root_path + 'group_std_ffr_eeg_200.npy')
dev1 = np.load(root_path + 'group_dev1_ffr_eeg_200.npy')
dev2 = np.load(root_path + 'group_dev2_ffr_eeg_200.npy')

addnoise_std = np.append(std,np.array([std_noise]),axis=0)
addnoise_dev1 = np.append(dev1,np.array([dev1_noise]),axis=0)
addnoise_dev2 = np.append(dev2,np.array([dev2_noise]),axis=0)

X = np.concatenate((addnoise_std,addnoise_dev1,addnoise_dev2),axis=0)
y = np.concatenate((np.repeat(0,len(addnoise_std)),np.repeat(1,len(addnoise_dev1)),np.repeat(2,len(addnoise_dev2))))

addnoise_X = np.append(X,np.array([std_noise,dev1_noise,dev2_noise]),axis=0)
addnoise_y = np.append(y,np.array([0,1,2]))

scores = cross_val_multiscore(clf, X, y, cv=StratifiedKFold(n_splits=19), n_jobs=None)
scores

scores = cross_val_multiscore(clf, addnoise_X, addnoise_y, cv=19, n_jobs=None)
scores = cross_val_multiscore(clf, addnoise_X, addnoise_y, cv=StratifiedKFold(n_splits=19), n_jobs=None)
scores = cross_val_multiscore(clf, addnoise_X, addnoise_y, cv=LeaveOneOut(), n_jobs=None)

## visualization
fig, axs = plt.subplots(19)
fig.suptitle('FFR mba')
for n in range(19):
    axs[n].plot(times, addnoise_dev2[n,:]*1e6,'k')
    axs[n].set_xlim([-0.02,0.2])
    axs[n].set_ylim([-1.2, 1.2])
    
## Some ML testing
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=1)

X_train = X_train.transpose()
X_test = X_test.transpose()
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
scaler = preprocessing.StandardScaler().fit(X_test)
X_test = scaler.transform(X_test)
X_train = X_train.transpose()
X_test = X_test.transpose()

logreg = LogisticRegression(random_state=16)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
cnf_matrix
target_names = ['ba', 'mba','pa']
print(classification_report(y_test, y_pred, target_names=target_names))

