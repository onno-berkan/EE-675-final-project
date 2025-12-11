# Code for My EE 675 Final Project - Onno Berkan
This repository contains the code used in my EE 675 final project, titled "Kalman Filter-Based Neural Alignment for Speech Prediction". I nicknamed the project "Mind Reader", which may be seen in a few places within the code. The project used data from ([Willett et al., 2023](https://www.nature.com/articles/s41586-023-06377-x)), who built an RNN to predict what a participant was attempting to say. I replaced the RNN with a Kalman Filter + LDA to track track latent neural states corresponding to what word the participant wanted to say and run predictions on them. This course was instructed by Prof. Maryam Shanechi, taken in Fall of 2025.

## To Reproduce Willett et al.'s Results
All of Willett et al.'s code and data can be found through ([here](https://github.com/fwillett/speechBCI/tree/main)). This repository contains parts of their code that I've used, which are indicated by comments within files. I replicated their RNN results as well, the code for which can be found in their repository.

## Description of Files
initialAnalysis.ipynb: Run on your local machine. Uses the Tuning and Sentences datasets from Willett et al. Reproduces their confusion matrix, trains and tests LDA on their Tuning data. Extracts "Yes"/"No" trials from the Tuning data, trains a new LDA and tests on them. Tests both LDAs on "Yes"/"No" data from May 17th and 19th data from Sentences, shows that LDA trained on one day does not predict above chance-level on data from another day.

wordsOnlyAnalysis.py: Modification of Willett et al.'s analysis.py to only contain the functions used in initialAnalysis.py, such as the GNB.

runYesNoGKF.py: Run using CARC or other server. Uses "tuningTasks/t12.2022.05.03_fiftyWordSet.mat" from Willett et al.'s dataset. Preprocesses the data and extracts "Yes" and "No" trials, as well as labels. Defines GKF, FA, and feature extraction functions. Defines cross-validation folds and prints accuracies for each one. This file as is will tune a GKF with 20 latent dimensions and two EM iterations, with 5-fold cross-validation. It will also extract the "Yes" and "No" trials from two other days' data, apply the same pre-processing steps, run them through the GKF trained on the Tuning data, and classify them using LDA. I show that GKF + LDA can achieve high accuracy on sessions where just LDA fails.

runGKF.py: Run using CARC or other server. Same as runYesNoGKF.py but extracts trials for every word. This file as is will tune a GKF with 60 latent dimensions and two EM iterations, with 5-fold cross-validation. It will also extract the "Yes" and "No" trials from two other days' data, apply the same pre-processing steps, run them through the GKF trained on the Tuning data, and classify them using LDA. I show that GKF + LDA can achieve high accuracy on sessions where just LDA fails.

plotData.ipynb: Because I ran the above two files on a server, I couldn't output graphs from them. I outputted arrays of results and used this notebook to graph them. See below for how to modify them to atatin my results.

## To Reproduce My Results

For Tables 1 and 2: Table 1 can be obtained by commenting out the pre-processing steps in initialAnalysis.py for the no-preprocessing columns and by simply running the cells for the rest. Table 2 can be obtained by running the cells further down the same notebook. Note that the notebook does not output accuracy metrics, but they can clearly be seen from the output.

For Graph 1 and 2 (Accuracy vs EM Iterations and Latent State Dimensionality in Yes/No): Manipulate runYesNoGKF.py by adding for loops to vary the EM_ITERS and LATENT_DIM variables. See comments on code. Run on a server. Paste data into plotData.ipynb.

For Graph 3 (Accuracy vs Latent State Dimensionality in 50 Word): Same as above, manipulate runGKF.py this time. See comments on code. This needs to be run on a server; 10+ GB of memory is needed if you want to go above ~40 dimensions, and the amount of memory needed only goes up. Paste data into plotData.ipynb.

For Table 3: Uncomment the last parts of runYesNoGKF.py and runGKF.py
