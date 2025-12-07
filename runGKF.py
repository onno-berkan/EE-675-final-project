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

LATENT_DIM = 16

X_with_nans = []
for trial in fiftyWordSubset:
    X_with_nans.append(trial)
    X_with_nans.append(np.full((1, 128), np.nan))

X_stacked = np.vstack(X_with_nans)[:-1]
X_masked = ma.masked_invalid(X_stacked)

# factor analysis
fa = FactorAnalysis(n_components=LATENT_DIM)
fa.fit(fiftyWordSubset.reshape(-1, 128))
C_init = fa.components_.T
R_init = np.diag(fa.noise_variance_)

# global 'meta' KF
kf_global = KalmanFilter(
    n_dim_state=LATENT_DIM,
    n_dim_obs=128,
    observation_matrices=C_init,
    observation_covariance=R_init,
    em_vars=['transition_matrices', 'transition_covariance', 
             'initial_state_mean', 'initial_state_covariance']
)
kf_global = kf_global.em(X_masked, n_iter=10) # 10 iters is enough

print("Global Model Trained. Extracting Features...")
# takes 40+ minutes with 10/12 latent dimensions
# takes 303 minutes with 16 dims

# feature extraction step
def extract_features(data, kf_model):
    features = []
    for trial in data:
        (smoothed, _) = kf_model.smooth(trial)
        
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
print(f"Feature Matrix Shape: {X_latent.shape}") # (1000, 5*latent_dim_size)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) # 5 folds -- gets slightly better acc at 10
lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')

accuracies = []

# Ensure y is 1D
y = labels.ravel().astype(int)

for fold, (train_idx, test_idx) in enumerate(skf.split(X_latent, y)):
    X_train, X_test = X_latent[train_idx], X_latent[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    lda.fit(X_train, y_train)
    
    pred = lda.predict(X_test)
    acc = accuracy_score(y_test, pred)
    accuracies.append(acc)
    
    print(f"Fold {fold+1} Accuracy: {acc*100:.1f}%")

print("\nChance Level: 2%")
print(f"\nAverage Accuracy: {np.mean(accuracies)*100:.1f}%")