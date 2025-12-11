import scipy.io
import numpy as np

import numpy.ma as ma
from pykalman import KalmanFilter
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import FactorAnalysis
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


baseDir = "/Users/apple/Documents/MATLAB/EE 675/Willett Data/tuning-tasks-all/"
fiftyWordDat = scipy.io.loadmat(baseDir+'tuningTasks/t12.2022.05.03_fiftyWordSet.mat')

# sqrt transform -- makes spike bins look more gaussian
fiftyWordDat['feat'] = np.concatenate([fiftyWordDat['tx2'][:,:32].astype(np.float32), fiftyWordDat['tx2'][:,96:128].astype(np.float32), fiftyWordDat['spikePow'][:,:32].astype(np.float32), fiftyWordDat['spikePow'][:,96:128].astype(np.float32)], axis=1)
fiftyWordDat['feat'] = np.sqrt(fiftyWordDat['feat'])

# collect the bins for yes and no trials

yesIdx = np.where(fiftyWordDat['cueList'] == 'yes')[1][0]
noIdx = np.where(fiftyWordDat['cueList'] == 'no')[1][0]

noTrials = np.where(fiftyWordDat['trialCues'] == noIdx)[0]
yesTrials = np.where(fiftyWordDat['trialCues'] == yesIdx)[0]

noTrialEpochs = [fiftyWordDat['goTrialEpochs'][trialNum] for trialNum in noTrials]
yesTrialEpochs = [fiftyWordDat['goTrialEpochs'][trialNum] for trialNum in yesTrials]

noTrialBins = [fiftyWordDat['feat'][epoch[1] - 50:epoch[1]] for epoch in noTrialEpochs] # the last 50 bins were found to be the most informative
yesTrialBins = [fiftyWordDat['feat'][epoch[1] - 50:epoch[1]] for epoch in yesTrialEpochs] # (20, 50, 256) 

yesNoData = np.concatenate((noTrialBins, yesTrialBins))
yesNoLabels = np.concatenate((np.zeros(20),np.ones(20)))

# define GKF

# use a for loop here to reproduce graphs. I recommend collecting all mean accuracies in an array and printing it to make your life easier.
LATENT_DIM = 20 # see paper for accuracy with different dimensionalities
EM_ITERS = 2 # see paper for accuracy with different iteration counts and overfitting

X_with_nans = []
for trial in yesNoData:
    X_with_nans.append(trial)
    X_with_nans.append(np.full((1, 128), np.nan))

X_stacked = np.vstack(X_with_nans)[:-1]
X_masked = ma.masked_invalid(X_stacked) # need to mask to make the A matrix work. See report for more details.

# factor analysis -- used to initialize C & R matrices for EM convergence
fa = FactorAnalysis(n_components=LATENT_DIM)
fa.fit(yesNoData.reshape(-1, 128))
C_init = fa.components_.T
R_init = np.diag(fa.noise_variance_)

# global KF
kf_global = KalmanFilter(
    n_dim_state=LATENT_DIM,
    n_dim_obs=128,
    observation_matrices=C_init,
    observation_covariance=R_init,
    em_vars=['transition_matrices', 'transition_covariance', 
             'initial_state_mean', 'initial_state_covariance']
)

kf_global = kf_global.em(X_masked, n_iter=EM_ITERS) # see paper for accuracy with different iteration counts

# feature extraction step
def extract_features(data, kf_model):
    features = []
    for trial in data:
        (smoothed, _) = kf_model.smooth(trial) # apply Kalman smoother
        
        # split word into windows -- could be adapted in the future to work on live data
        n_windows = 5
        window_size = 50 // n_windows
        
        trial_feats = []
        for w in range(n_windows):
            window_mean = np.mean(smoothed[w*window_size : (w+1)*window_size], axis=0) # mean of latent state in this window
            trial_feats.append(window_mean)
            
        features.append(np.concatenate(trial_feats))
        
    return np.array(features)

X_latent = extract_features(yesNoData, kf_global)
print(f"Feature Matrix Shape: {X_latent.shape}") # should be (40 trials, 5 windows*latent_dim_size)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) # see paper for accuracy with different fold counts
lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')

accuracies = []

y = yesNoLabels.ravel().astype(int)

# run cross-validation
for fold, (train_idx, test_idx) in enumerate(skf.split(X_latent, y)): # skf automatically balances classes in train/test splits
    X_train, X_test = X_latent[train_idx], X_latent[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    lda.fit(X_train, y_train)
    
    pred = lda.predict(X_test)
    acc = accuracy_score(y_test, pred)
    accuracies.append(acc)
    
    print(f"Fold {fold+1} Accuracy: {acc*100:.1f}%")

print("\nClassification On Tuning Data:")
print(f"\nAverage Accuracy: {np.mean(accuracies)*100:.1f}%")

# now it's time to try our GKF on data from other days

baseDir2 = "/Users/apple/Documents/MATLAB/EE 675/Willett Data/sentences/"

compDat1 = scipy.io.loadmat(baseDir2+'t12.2022.05.17_sentences.mat') # data from May 17th 2022
compDat2 = scipy.io.loadmat(baseDir2+'t12.2022.05.19_sentences.mat') # data from May 19th 2022

# does not actually subtract the mean; replicates step one of the tuning data pre-processing, also returns the indices of the relevant (50-word sentence) blocks
def meanSubtract2(dat):
    # sqrt transform -- makes spike bins look more gaussian
    dat['feat'] = np.concatenate([dat['tx2'][:,:32].astype(np.float32), dat['tx2'][:,96:128].astype(np.float32), dat['spikePow'][:,:32].astype(np.float32), dat['spikePow'][:,96:128].astype(np.float32)], axis=1)
    dat['feat'] = np.sqrt(dat['feat'])
        
    blockList = dat['blockList'][np.where(dat['blockTypes'] == 'OL Chang')[0]]
    return dat, blockList

# returns the bins for each sentence in the relevant blocks
def changSentences(data, blockList):
    sentences = []
    binRange = []
    for block in blockList:
        rangeTemp = np.where(data['blockNum'] == block)[0]
        trialMin = np.where(data['goTrialEpochs'] == rangeTemp[0])[0][0] + 1
        trialMax = np.where(data['goTrialEpochs'] == rangeTemp[-1] + 1)[0][0]
        sentences.append(data['sentences'][trialMin:trialMax + 1])
        binRange.append(data['goTrialEpochs'][trialMin:trialMax + 1])
    return np.concatenate(np.squeeze(sentences)), np.concatenate(binRange)

# doesn't normalize data; returns data for the 'yes' and 'no' trials
def normalizeYesNo(data):
    changDatTemp, blockList = meanSubtract2(data)
    sentenceList, binRange = changSentences(changDatTemp, blockList)

    noId = np.where(sentenceList == "No")[0].item()
    yesId = np.where(sentenceList == "Yes")[0].item()

    noData = data['feat'][binRange[noId, 0]:binRange[noId, 1], :]
    yesData = data['feat'][binRange[yesId, 0]:binRange[yesId, 1], :]

    return noData, yesData

noData1, yesData1 = normalizeYesNo(compDat1)
noData2, yesData2 = normalizeYesNo(compDat2)

# combine yes/no data for the two days
X_test_otherDay = [noData1, noData2, yesData1, yesData2]
y_otherDay = [0, 0, 1, 1] # labels
trial_otherDay = extract_features(X_test_otherDay, kf_global) # apply the GKF we tuned to extract features from the unseen sessions' data 
lda.fit(X_latent, y) # fit LDA to all of the tuning data, this will be tested on feature-extracted unseen sessions' data

# test LDA on unseen sessions' data
pred_otherDay = lda.predict(trial_otherDay)
acc_otherDay = accuracy_score(y_otherDay, pred_otherDay)

print("\Classification On Sentences Data:")
print(f"\nAverage Accuracy: {np.mean(acc_otherDay)*100:.1f}%") # should return 100% !!!