# -*- coding: utf-8 -*-
"""
Created on Tue May 25 16:27:59 2021

@author: Christina Zhao
"""
''' this script is to create the soa files for the CBS project'''
''' the soa list will be of n_train long'''
'''each elememt will have a list of soa numbers'''

import numpy as np
import matplotlib.pyplot as plt

n_train=200

train_length=[8,10,12,14]

train_length_seq=[]

# randomly select how long each train will be
for i in range(n_train):
    rand_select=train_length[np.random.randint(0,len(train_length))]
    train_length_seq.append(rand_select)
    
# draw soa functions
# f1
def f1(x):
    if (x>=3) & (x<=6):
        #y=0.05           # the lowest soa number
        y=0.15
    elif (x<3)==1:
        #y=0.5-0.15*x     # the intercept is the highest soa number
        #y=0.6-0.15*x
        y=0.5-(0.35/3)*x
    elif (x>6)==1:
        #y=-0.85+0.15*x
        #y=-0.75+0.15*x
        y=-0.55+(0.35/3)*x
    return y

def f2(x):
    if (x>=2) & (x<=6):
        #y=0.05
        y=0.15
    elif (x<2)==1:
        #y=0.5-0.225*x
        #y=0.6-0.225*x
        y=0.5-(0.35/2)*x
    elif (x>6)==1:
        #y=-0.85+0.15*x
        #y=-0.75 + 0.15*x
        y=-0.55 + (0.35/3) *x
    return y

def f3(x):
    if (x>=3) & (x<=7):
        #y=0.05
        y=0.15
    elif (x<3)==1:
        #y=0.5-0.15*x
        #y=0.6-0.15*x
        y=0.5-(0.35/3)*x
    elif (x>7)==1:
        #y=-1.525+0.225*x
        #y=-1.425 +0.225*x
        y=-1.075 + (0.35/2)*x
    return y

x=np.arange(0,9,0.01)

y1=np.zeros(len(x))
y2=np.zeros(len(x))
y3=np.zeros(len(x))

for i in range(len(x)):
    y1[i]=f1(x[i])
    y2[i]=f2(x[i])
    y3[i]=f3(x[i])
    
#theoretical distributions
plt.plot(x,y2)
plt.plot(x,y3)
plt.plot(x,y1,'k')


def scale_train_len(train_len):
    x=np.arange(0,train_len,1)
    m=9/(train_len-1)
    x_scaled=m*x
    return x_scaled

def normal_dist(x , mean , sd):
    prob_density = (np.pi*sd) * np.exp(-0.5*((x-mean)/sd)**2)
    return prob_density

def add_noise(x):
    if  (x<=0.15):    
        noise_interval=np.random.normal(0,0.01,1000)
        noise=noise_interval[np.random.randint(1,1000)]
    elif (x>0.15) & (x<0.3):
        noise_interval=np.random.normal(0,0.05,1000)
        noise=noise_interval[np.random.randint(1,1000)]
    elif (x>=0.3):
        noise_interval=np.random.normal(0,0.08,1000)
        noise=noise_interval[np.random.randint(1,1000)]
    return x+noise
                
# function to draw soa value
def draw_soa(train_len):
    #select function
    func_select=np.random.randint(1,3)
    # normalize train_len to 0-9
    train_scaled=scale_train_len(train_len)
    soa=np.zeros(len(train_scaled))
    if func_select==1:
        for i in range(len(train_scaled)):
            soa[i]=f1(train_scaled[i])
            soa[i]=add_noise(soa[i])
            if soa[i]<0.15:
                soa[i]=0.15
    elif func_select==2:
        for i in range(len(train_scaled)):
            soa[i]=f2(train_scaled[i])
            soa[i]=add_noise(soa[i])
            if soa[i]<0.15:
                soa[i]=0.15
    elif func_select==3:
        for i in range(len(train_scaled)):
            soa[i]=f3(train_scaled[i])
            soa[i]=add_noise(soa[i])
            if soa[i]<0.15:
                soa[i]=0.15
    return soa

### main part
soas=[]
for i in train_length_seq:
    soa=draw_soa(i)
    soas.append(soa)
    plt.plot(soa)
plt.hlines(y=0.14,xmin=0,xmax=12)
    

#np.save('/Users/Christina Zhao/Documents/ResearchProjects/CBS/stimuli/soas6_200.npy',soas,allow_pickle=True)