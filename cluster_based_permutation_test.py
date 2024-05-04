#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 11:13:47 2023
Statistical test on MMR1 vs. MMR2
non-parametric cluster-based permutation t-test

@author: tzcheng
"""

import numpy as np
import mne
import scipy.stats as stats

#%% non-paramatric permutation test on EEG
root_path='/home/tzcheng/Documents/GitHub/Paradigm/'

## MMR
times = np.linspace(-0.1,0.6,3501)
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

ts = 1000 # 0 s
te = 1750 # 0.25 s

X = MMR1-MMR2
X = X[:,ts:te]

T_obs, clusters, cluster_p_values, H0 = mne.stats.permutation_cluster_1samp_test(X, seed=0)
good_cluster_inds = np.where(cluster_p_values < 0.05)[0]
for i in np.arange(0,len(good_cluster_inds),1):
    print("The " + str(i+1) + "st significant cluster")
    print(times[ts:te][clusters[good_cluster_inds[i]]])