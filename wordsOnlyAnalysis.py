import numpy as np
import scipy.stats
from scipy.ndimage import gaussian_filter1d
from numba import njit
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from omegaconf import OmegaConf

import sys
sys.path.insert(0, "./NeuralDecoder")
from neuralDecoder.neuralSequenceDecoder import NeuralSequenceDecoder
import os

@njit
def meanResamples(trlConcat, nResamples):
    resampleMeans = np.zeros((nResamples, trlConcat.shape[1], trlConcat.shape[2]))
    for rIdx in range(nResamples):
        resampleIdx = np.random.randint(0,trlConcat.shape[0],trlConcat.shape[0])
        resampleTrl = trlConcat[resampleIdx,:,:]
        resampleMeans[rIdx,:,:] = np.sum(resampleTrl, axis=0)/trlConcat.shape[0]

    return resampleMeans

def triggeredAvg(features, eventIdx, eventCodes, window, smoothSD=0, computeCI=True, nResamples=100):
    winLen = window[1]-window[0]
    codeList = np.unique(eventCodes)
    
    featAvg = np.zeros([len(codeList), winLen, features.shape[1]])
    featCI = np.zeros([len(codeList), winLen, features.shape[1], 2])
    allTrials = []
    
    for codeIdx in range(len(codeList)):
        trlIdx = np.squeeze(np.argwhere(eventCodes==codeList[codeIdx]))
        trlSnippets = []
        for t in trlIdx:
            if (eventIdx[t]+window[0])<0 or (eventIdx[t]+window[1])>=features.shape[0]:
                continue
            trlSnippets.append(features[(eventIdx[t]+window[0]):(eventIdx[t]+window[1]),:])
        
        trlConcat = np.stack(trlSnippets,axis=0)
        allTrials.append(trlConcat)
            
        if smoothSD>0:
            trlConcat = gaussian_filter1d(trlConcat, smoothSD, axis=1)

        featAvg[codeIdx,:,:] = np.mean(trlConcat, axis=0)
        
        if computeCI:
            tmp = np.percentile(meanResamples(trlConcat, nResamples), [2.5, 97.5], axis=0)   
            featCI[codeIdx,:,:,:] = np.transpose(tmp,[1,2,0]) 
        
    return featAvg, featCI, allTrials

def plotPreamble():
    import matplotlib.pyplot as plt

    SMALL_SIZE=5
    MEDIUM_SIZE=6
    BIGGER_SIZE=7

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    plt.rcParams['svg.fonttype'] = 'none'

#gaussian naive bayes classifier with variable time window and channel set
def gnb_loo(trials_input, timeWindow, chanIdx):
    unroll_Feat = []
    for t in range(len(trials_input)):
        for x in range(trials_input[t].shape[0]):
            unroll_Feat.append(trials_input[t][x,:,:])

    unroll_Feat = np.concatenate(unroll_Feat, axis=0)
    mn = np.mean(unroll_Feat, axis=0)
    sd = np.std(unroll_Feat, axis=0)
    
    unroll_X = []
    unroll_y = []

    for t in range(len(trials_input)):
        for x in range(trials_input[t].shape[0]):
            tmp = (trials_input[t][x,:,:] - mn[np.newaxis,:])/sd[np.newaxis,:]
            b1 = np.mean(tmp[timeWindow[0]:timeWindow[1],chanIdx], axis=0)
            
            unroll_X.append(np.concatenate([b1]))
            unroll_y.append(t)

    unroll_X = np.stack(unroll_X, axis=0)
    unroll_y = np.array(unroll_y).astype(np.int32)
    
    from sklearn.naive_bayes import GaussianNB

    y_pred = np.zeros([unroll_X.shape[0]])
    for t in range(unroll_X.shape[0]):
        X_train = np.concatenate([unroll_X[0:t,:], unroll_X[(t+1):,:]], axis=0)
        y_train = np.concatenate([unroll_y[0:t], unroll_y[(t+1):]])

        gnb = GaussianNB()
        gnb.fit(X_train, y_train)
        gnb.var_ = np.ones(gnb.var_.shape)*np.mean(gnb.var_)

        pred_val = gnb.predict(unroll_X[np.newaxis,t,:])
        y_pred[t] = pred_val
        
    return y_pred, unroll_y

def bootCI(x,y):
    nReps = 10000
    bootAcc = np.zeros([nReps])
    for n in range(nReps):
        shuffIdx = np.random.randint(len(x),size=len(x))
        bootAcc[n] = np.mean(x[shuffIdx]==y[shuffIdx])
        
    return np.percentile(bootAcc,[2.5, 97.5])