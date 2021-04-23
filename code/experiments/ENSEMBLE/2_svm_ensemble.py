import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # use id from $ nvidia-smi


import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time


#Developed classes
import sys
sys.path.append("../../classes")
from Recognition_Dataset_V1 import Dataset
from Recognition_EncoderModel_V2 import EncoderModel
from Recognition_EmbeddingsDataset_V1 import EmbeddingsDataset
from Recognition_SVM_V4 import SVM


def saveHeader(saveAs):
    #save the header
    values = [["iteration", "gamma", "nu", "tau", "noise", "TPR", "TNR", "accuracy", "TP", "FP", "FN", "TN", "total in train", "total not in train", "fp-3-3", "fn-2-3", "time"]]
    data = pd.DataFrame(data= values)    
    data.to_csv(saveAs, header=None, mode="a") 

def save(saveAs, iteration, gamma, nu, threshold, noise, results, delta):
    #get values:
    acc1 = results[0]
    acc2 = results[1]    
    accuracy = (results[2][2] + results[3][1]) / (results[2][0] + results[3][0])
    tp = results[2][2]
    fp = results[3][2] + results[3][3]
    fn = results[2][1] + results[2][3]
    tn = results[3][1]
    total_in_train = results[2][0]
    total_not_in_train = results[3][0]
    fp_3_3 = results[3][3]
    fn_2_3 = results[2][3]
                
    values = [[iteration, gamma, nu, threshold, noise, acc1, acc2, accuracy, tp, fp, fn, tn, total_in_train, total_not_in_train, fp_3_3, fn_2_3, delta]]
    data = pd.DataFrame(data= values)    
    data.to_csv(saveAs, header=None, mode="a")  

    
#CSV file
saveAs = "./summary_svm_ensemble.csv"
saveHeader(saveAs)    
    
#load the dataset
pathData = "../../data"
fileData = "data_cond1_c2"

print("--> loading the datasets")
dataset = Dataset(pathData, fileData+".csv")
print("--> datasets ready")

#parameters
path = "./"
gammas = np.linspace(0.1, 5, 50)
nus = np.linspace(0.05, 0.4, 36)
thresholds = np.linspace(0.0, 0.2, 21) * -1
iterations=20


#Repeat training 
for i in range(1, iterations+1):
    print("\n-->Iteration: ", i)
        
    #load the embeddings
    print("loading embeddings")
    embeddings0 = EmbeddingsDataset(None, dataset)
    embeddings0.load(path, "iter_" + str(i) + "_embeddings.npy")
        
    #load the subsets
    print ("loading the subsets")
    dataset.loadSets(path,"iter_" + str(i) + "_dataset.npy")
                            
    #Get the random training set
    randomTrainingSet, randomKnownSet = dataset.selectRandomUnits(dataset.validationSet, 10)

    #Train SVM with different gammas
    for gamma in gammas:
        for nu in nus:
            t0 = time.time()
            print("--> Train SVM: gamma=", gamma, " nu=", nu)
            svm = SVM(embeddings0)
            svm.fit(randomTrainingSet, gamma=gamma, nu = nu)
                      
            for tau in thresholds:  
                #noise 0%
                svm.embeddings = embeddings0
                results = svm.accuracy(randomKnownSet, dataset.unseenSet, tau)
                t1 = time.time()
                save(saveAs, i, gamma, nu, tau, 0.00, results, t1-t0)
    