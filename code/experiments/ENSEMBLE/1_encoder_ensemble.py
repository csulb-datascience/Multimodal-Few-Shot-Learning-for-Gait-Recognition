# csulb-datascience
#
# Program to train the models.
#   the results are saved in a csv file "summary.csv"
#
# Authors: 
#      Nelson Minaya, email: nelson.minaya@student.csulb.edu
#      Nhat Anh Le,   email: nhat.le01@student.csulb.edu
#
# Date: June 2020
#
# Include a reference to this site if you will use this code.

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # use id from $ nvidia-smi



import tensorflow as tf
import numpy as np
import pandas as pd
import time

#classes
import sys
sys.path.append("../../classes")
from Recognition_Dataset_V1 import Dataset
from Recognition_EncoderModel_V2 import EncoderModel
from Recognition_EmbeddingsDataset_V1 import EmbeddingsDataset


#Save the header for the CSV file
def saveHeader(saveAs):
    #save the header
    values = [["iteration","lossTrain","lossValid", "samplesTrain", "samplesValid", "time"]]
    data = pd.DataFrame(data= values)    
    data.to_csv(saveAs, header=None, mode="a") 
    
#Save the results of testing 
def save(saveAs, iteration, lossTrain, lossValid, samplesTrain, samplesValid, delta):
    #get values:               
    values = [[iteration, lossTrain, lossValid, samplesTrain, samplesValid, delta]]
    data = pd.DataFrame(data= values)    
    data.to_csv(saveAs, header=None, mode="a")     
    
#CSV file
saveAs = "./summary_ensemble.csv"
saveHeader(saveAs)    
    
#load the dataset
pathData = "../../data"
fileData = "data_cond1_c2"

print("--> loading the datasets")
dataset = Dataset(pathData, fileData+".csv")
print("--> datasets ready")

#parameters
numberPeopleTraining = 16
numberPeopleKnown = 7
learningRate = 0.001
alpha = 1.0
beta = 1.0
epochs =  20
batchSize = 64
iterations=20

print("Training encoders")    
for i in range(1, iterations+1):
    print("\n--> Iteration: ", i)
    
    #Split the dataset for: training, validation, unknown, test
    dataset.split(numberPeopleTraining, numberPeopleKnown)
    dataset.saveSets(".","iter_" + str(i) + "_dataset.npy")
    x_train, y_train, m_train = dataset.getDataset(dataset.trainingSet, batchSize=batchSize)
    #x_valid, y_valid, m_valid = dataset.getDataset(dataset.validationSet, batchSize=batchSize)
        
    # Creates a session 
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
            
    with session.as_default():                    
        t0 = time.time()
        print("--> training...")
        encoderModel = EncoderModel()
        encoder = encoderModel.getCompiledEnsemble(dataset.unitSize(), alpha, beta, learningRate)
        history = encoder.fit(x_train, y_train, batch_size = batchSize, epochs = epochs)        
        #lossTrain, lossValid = encoderModel.getResults(history)

        #Save results for encoder
        print("\n--> saving ")
        #encoder.save("iter_" + str(i) + "_model.h5")
        embeddings = EmbeddingsDataset(encoder, dataset)
        embeddings.predictEmbeddings()
        embeddings.save(".", "iter_" + str(i) + "_embeddings.npy")
                
        t1 = time.time()
        save(saveAs, i, 0.0, 0.0, len(y_train), 0, t1-t0)

    tf.compat.v1.reset_default_graph()
    session.close()            





