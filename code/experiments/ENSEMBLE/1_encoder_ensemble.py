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
        
    # Creates a session 
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
            
    with session.as_default():                    
        print("--> training...")
        encoderModel = EncoderModel()
        encoder = encoderModel.getCompiledEnsemble(dataset.unitSize(), alpha, beta, learningRate)
        history = encoder.fit(x_train, y_train, batch_size = batchSize, epochs = epochs)        

        #Save results for encoder
        print("\n--> saving ")
        encoder.save("iter_" + str(i) + "_model.h5")
        embeddings = EmbeddingsDataset(encoder, dataset)
        embeddings.predictEmbeddings()
        embeddings.save(".", "iter_" + str(i) + "_embeddings.npy")

    tf.compat.v1.reset_default_graph()
    session.close()            





