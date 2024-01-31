# Paradigm
 
This repo includes the data and codes for the project titled "Validating a new paradigm for simultaneously assessing mismatch response and complex auditory brainstem response". The authors are Tzu-Han Zoe Cheng and Tian Christina Zhao. Please contact Zoe at tzcheng@uw.edu if you have any questions! 

 
**Data** 

EEG data are stored in .npy format, which can be loaded with NumPy package in Python. There are six data files. Three of them (i.e. group_std_eeg.npy, group_dev1_eeg.npy and group_dev2_eeg.npy) are time series of MMR across all 18 subjects, and the other three are cABR data (i.e. group_std_cabr_eeg_200.npy, group_dev1_cabr_eeg_200.npy and group_dev2_cabr_eeg_200.npy).  

  

**Code** 

The codes for data analyses and visualization are included. See below for the details.  

The decoding analysis for both MMR and cABR can be found in decoding.py 
The cluster-based permutation test for the MMR can be found in cluster_based_permutation_test.py 
The cross-correlation analysis for cABR can be found in stimulus-to-response_xcorr.py 
See vis.py for visualization of each subplot in Figure 2  
