import scipy.io
import numpy as np

import numpy.ma as ma
from pykalman import KalmanFilter
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import FactorAnalysis
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


baseDir = "/home1/berkan/ee675-mind-reader/tuning-tasks-all/" # adjust to your directory
fiftyWordDat = scipy.io.loadmat(baseDir+'tuningTasks/t12.2022.05.03_fiftyWordSet.mat')

# sqrt transform -- makes spike bins look more gaussian
fiftyWordDat['feat'] = np.concatenate([fiftyWordDat['tx2'][:,:32].astype(np.float32), fiftyWordDat['tx2'][:,96:128].astype(np.float32), fiftyWordDat['spikePow'][:,:32].astype(np.float32), fiftyWordDat['spikePow'][:,96:128].astype(np.float32)], axis=1)
fiftyWordDat['feat'] = np.sqrt(fiftyWordDat['feat'])

# make an array containing the data

fiftyWordSubset = []
labels = []

for cue in fiftyWordDat['cueList'][0]:
    cueIdx = np.where(fiftyWordDat['cueList'] == cue)[1]
    cueTrials = np.where(fiftyWordDat['trialCues'] == cueIdx)[0]
    cueTrialEpochs = [fiftyWordDat['goTrialEpochs'][trialNum] for trialNum in cueTrials]
    cueTrialBins = [fiftyWordDat['feat'][epoch[1] - 50:epoch[1]] for epoch in cueTrialEpochs] # the last 50 bins were found to be the most informative

    fiftyWordSubset.append(cueTrialBins)
    labels.append(np.ones(20)*cueIdx)

fiftyWordSubset = np.concatenate(fiftyWordSubset[1:], axis=0) # (1000 trials, 50 bins, 128 channels) array
labels = np.concatenate(labels[:50]) # (1000,) array

# use a for loop here to reproduce graphs. I recommend collecting all mean accuracies in an array and printing it to make your life easier.
LATENT_DIM = 60 # see paper for accuracy with different dimensionalities and convergence
EM_ITERS = 2 # see paper for accuracy with different iterations and overfitting

X_with_nans = []
for trial in fiftyWordSubset:
    X_with_nans.append(trial)
    X_with_nans.append(np.full((1, 128), np.nan))

X_stacked = np.vstack(X_with_nans)[:-1]
X_masked = ma.masked_invalid(X_stacked) # need to mask to make the A matrix work. See report for more details.

# factor analysis -- used to initialize C & R matrices for EM convergence
fa = FactorAnalysis(n_components=LATENT_DIM)
fa.fit(fiftyWordSubset.reshape(-1, 128))
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

kf_global = kf_global.em(X_masked, n_iter=EM_ITERS)  # see paper for accuracy with different iteration counts

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
            # Mean of latent state in this window
            window_mean = np.mean(smoothed[w*window_size : (w+1)*window_size], axis=0)
            trial_feats.append(window_mean)
            
        features.append(np.concatenate(trial_feats))
        
    return np.array(features)

# extract features
X_latent = extract_features(fiftyWordSubset, kf_global)
print(f"Feature Matrix Shape: {X_latent.shape}") # should be (1000 trials, 5 windows*latent_dim_size)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) # see paper for accuracy with different fold counts
lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')

accuracies = []

y = labels.ravel().astype(int)

for fold, (train_idx, test_idx) in enumerate(skf.split(X_latent, y)): # skf automatically balances classes in train/test splits
    X_train, X_test = X_latent[train_idx], X_latent[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    lda.fit(X_train, y_train)
    
    pred = lda.predict(X_test)
    acc = accuracy_score(y_test, pred)
    accuracies.append(acc)
    
    print(f"Fold {fold+1} Accuracy: {acc*100:.1f}%")

print("\nChance Level: 2%")
print(f"\nAverage Accuracy: {np.mean(accuracies)*100:.1f}%")