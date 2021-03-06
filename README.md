# Multimodal-Few-Shot-Learning-for-Gait-Recognition
This repository is the official implementation of Multimodal-Few-Shot-Learning-for-Gait-Recognition paper, which proposes a system that learns a mapping from a multimodal time series collected using insole to a latent space to address the open set gait recognition problem. The system maps unit steps to embedding vectors using an ensemble consisting of a convolutional neural network and a recurrent neural network. To recognize each individual, the system learns a decision function using a one-class support vector machine from a few embedding vectors (few-shot) of the person in the latent space, then the system determines whether an unknown unit step is recognized as belonging to a known individual or not.

# Requirements
Some of the main packages used for this project are Tensorflow-gpu 1.14, Keras 2.2.4, and scikit-learn 0.23.2.
It is recommended to create a new environment and install the packages listed in requirements.txt:
```
pip install -r requirements.txt
```

# Experiments
This repository contains the code to perform experiments with the CNN, RNN, and the Ensemble individually. Each experiment is divided in two steps:

- <b>1. Train the encoder</b>: this script trains the encoder (CNN, RNN, or Ensemble). It saves the trained encoder and the predicted embeddings for the next step.
- <b>2. Train and test the classifier</b>: it trains the OSVM classifier with the few-shot learning method and test it with known-unknown-test and unknown-unknown-test sets. It requires the predicted embeddings obtained in the previous step. The results are saved in a CSV file for later plot and analysis.

# Datasets
As is shown in the following image, the data was collected from 30 subjects and it was split into three sets: 

<img src="images/split.png" width="70%">

- <b>Training set</b>: used to train the CNN, RNN, and ensemble models independently. It consists of all the unit steps of 16 individuals selected randomly.
- <b>Unknown-Known test set</b>: it contains the unit steps of 7 individuals selected randomly from the 14 remaining people after selecting the training set. This dataset is divided in two subsets. The first subset consists of 10 unit steps for each individual and it is used for training the OSVM classifier. The second subset is the remaining steps of the same 7 individuals and it is used to test the classifier as known data in the open set gait recognition problem.
- <b>Unknown-unknown test set</b>: it contains all the unit steps of the remaining 7 subjects which were not used in any training process, therefore they are unknown subjects. It is used for testing the classifier as unknown data in the open set gait recognition problem.

# Evaluation
The system is evaluated in terms of Accuracy (ACC), True Positive Rate (TPR), and True Negative Rate (TNR) defined as follows:

- ![equation one](https://latex.codecogs.com/gif.latex?%5Cinline%20%5Cdpi%7B100%7D%20%5Csmall%20ACC%20%3D%20%5Cfrac%7BTP%20&plus;%20TN%7D%7BTP%20&plus;%20FN%20&plus;%20TN%20&plus;%20FP%7D)

- ![equation two](https://latex.codecogs.com/gif.latex?%5Cinline%20%5Cdpi%7B100%7D%20%5Csmall%20TPR%3D%20%5Cfrac%7BTP%7D%7BTP%20&plus;%20FN%7D)

- ![equation three](https://latex.codecogs.com/gif.latex?%5Cinline%20%5Cdpi%7B100%7D%20%5Csmall%20TNR%3D%20%5Cfrac%7BTN%7D%7BTN%20&plus;%20FP%7D)

Where, 
- TP stands for True Positive and it is the total unit steps in the known test set that were classified correctly. 
- FN stands for False Negative and it is the total unit steps in the known test set that were classified incorrectly. 
- TN stands for True Negative and it is the total unit steps in the unknown test set that were classified correctly as an unknown participant.
- FP stands for False Negative and it is the total unit steps in the unknown test set that were classified incorrectly as a known participant.

# Results

The following countour plots show the obtained distributions of ACC as a function of ?? and ?? for the CNN, RNN, and ensemble models respectivelly. A comparison of the area in which the rates are greater than 90% (light green to yellow areas) indicates that the region of the ensemble model is broader than that of the regions of the CNN or RNN model. This means that the ensemble model has a weak dependency when selecting ?? and ??, which affects the robustness of the recognition result.<br/>
<img src="images/acc.png" width="70%">

The distribution of the TPR is shown in the following plot. A comparison of the area in which the rates are greater than 93% (yellow), the region of the RNN model is slightly broader than that of the CNN model. The overall distribution of the ensemble model is similar to that of the RNN model.<br/>

<img src="images/tpr.png" width="70%">

The distribution of the TNRs is shown below. Contrary to the distributions of the TPR, the overall distribution of the ensemble model is almost identical to the distribution of the CNN model. In particular, a comparison of the area in which the rates are greater than 93% (yellow) reveals that the region of the CNN model is significantly broader than that of the RNN model.<br/>

<img src="images/tnr.png" width="70%">

To determine the effect of ??, we specified separate values of ?? and ?? for the different models in the experiment. We used ?? = 1.9 and ?? = 0.06 for the ensemble model, ?? = 1.8 and ?? = 0.06 for the CNN model, and ?? = 2.2 and ?? = 0.08 for the RNN model. In the figure, we see that choosing a ?? value smaller than 0 significantly improves the TPR and ACC. 

<img src="images/tau.png" width="70%">

# Contributors
Nelson Minaya nelson.minaya@student.csulb.edu <br/>
Nhat Le nhat.le01@student.csulb.edu
