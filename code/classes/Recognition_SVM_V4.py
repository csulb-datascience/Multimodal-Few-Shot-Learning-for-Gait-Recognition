# csulb-datascience
#
# Authors: 
#      Nelson Minaya, email: nelson.minaya@student.csulb.edu
#      Nhat Anh Le,   email: nhat.le01@student.csulb.edu
#
# Class version: 4.0
# Date: June 2020
#
# Include a reference to this site if you will use this code.

import numpy as np
from sklearn.svm import OneClassSVM
from Recognition_EmbeddingsDataset_V1 import EmbeddingsDataset

class SVM:
    #constructor
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.classifier = {}
        self.centroids = {}
        self.stdDeviation={}
        self.results = None
        
    #trains instances of SVM.svc according to dataIndex information
    #Turn in to a binary classification problem, label 1 for current_id, and -1 for the rest    
    def fit(self, dataIndex, gamma=0.1, nu = 0.5):
        self.results=None
        self.classifier.clear()        
        self.centroids = self.embeddings.getCentroids(dataIndex)
        self.stdDeviation = self.getStdDeviation(dataIndex)
        x, y = self.embeddings.getDataset(dataIndex)
        for key in dataIndex:
            yBinary = np.asarray([1 if userId==key else -1 for userId in y])
            x_current = x[yBinary == 1]
            svc = OneClassSVM(kernel='rbf', gamma=gamma, nu = nu)
            self.classifier[key] = svc.fit(x_current)
        return(self.classifier, self.centroids)        

    #Calculates the standard deviation all points to the centroid
    def getStdDeviation(self, dataIndex):
        std={}
        for key in dataIndex:
            if len(dataIndex[key]) > 0:
                vectors = np.asarray([self.embeddings.getEmbedded(key, unitId)-self.centroids[key] 
                                        for unitId in dataIndex[key]])
                distances = np.linalg.norm(vectors, axis=1)
                std[key] = np.std(distances)
        return std
    
    ##find the closest centroid. returns the user id who has the closest centroid
    def getClosestCentroid(self, embeddedVector):
        closest = -1
        minDist = float("inf")
        for key in self.centroids:
            dist = np.linalg.norm(embeddedVector - self.centroids[key]) 
            if dist < minDist:
                minDist = dist
                closest = key
        return(closest)
        
    #returns the closest user Id if it is an inlier, otherwise -1
    def predictClosest(self, key, unitId, threshold):
        embeddedVector = self.embeddings.getEmbedded(key, unitId)                
        closest = self.getClosestCentroid(embeddedVector)         
        predicted = self.classifier[closest].decision_function([embeddedVector])[0]
        if predicted < threshold: closest = -1
        return(closest)
        
    #returns the accuracy of the trained network
    def accuracy(self, dataIndexSeen, dataIndexUnseen, threshold):
        score1 = self.test(dataIndexSeen, threshold)
        score2 = self.test(dataIndexUnseen, threshold)  
        acc1 = score1[2] / score1[0]
        acc2 = score2[1] / score2[0]
        self.results = [acc1, acc2, score1, score2]
        return(self.results)
            
    #Test the results of SVM.svc
    def test(self, dataIndexTest, threshold):
        score = [0,0,0,0]
        for key in dataIndexTest:
            for unitId in dataIndexTest[key]:
                prediction = self.predictClosest(key, unitId, threshold)
                case=1 if prediction==-1 else 2 if prediction==key else 3
                score[case] +=1  #counter of each case
                score[0]+=1      #total of tests
        return score        

    #Prints the accuracy according to the last obtained results
    def printAccuracy(self):
        print("Accuracy validation dataset=", self.results[0])
        print(" - Total in train=", self.results[2][0])
        print(" - in train incorrect classify=", self.results[2][1])
        print(" - in train correct identify=", self.results[2][2])
        print(" - in train incorrect identify=", self.results[2][3])

        print("Accuracy unseen dataset =", self.results[1])
        print(" - total not in train=", self.results[3][0])
        print(" - not in train correct classify=", self.results[3][1])
        print(" - not in train incorrect classify=", self.results[3][2]+self.results[3][3])
            
    #Save the accuracy as a nampy array
    def saveAccuracy(self, path, fileName):
        np.save(path+"/"+fileName, self.results)
        