# csulb-datascience
#
# Authors: 
#      Nelson Minaya, email: nelson.minaya@student.csulb.edu
#      Nhat Anh Le,   email: nhat.le01@student.csulb.edu
#
# Class version: 1.0
# Date: June 2020
#
# Include a reference to this site if you will use this code.

import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.manifold import TSNE
from Recognition_Dataset_V1 import Dataset

class EmbeddingsDataset:
    #Constructor
    def __init__(self, network, dataset):
        self.network = network
        self.dataset = dataset
        self.embeddingTable = dict()
    
    #Given an image of shape Height * 28, return the image that is feasible for the network
    def prepareImage(self, image):
        height = len(image)        
        press = image[:, 0:16]
        acc   = image[:, 16:22]
        gyro  = image[:, 22:]        
        press = press.reshape(1, height, 16)
        acc   = acc.reshape(1, height, 6)
        gyro  = gyro.reshape(1, height, 6)
        return [press, acc, gyro]

    #returns the encoded embedded vector
    def encode(self, unitStep):
        if self.network == None: return None
        embedded = self.network.predict(self.prepareImage(unitStep))
        return embedded.reshape(embedded.shape[1])
    
    #returns the embedded vector of a unit step
    def getEmbedded(self, personId, unitId):
        if (personId, unitId) not in self.embeddingTable:
            unitStep = self.dataset.getUnitStep(personId, unitId)
            self.embeddingTable[(personId, unitId)] = self.encode(unitStep)
        return(self.embeddingTable[(personId, unitId)])
        
    #predict all the embeddings
    def predictEmbeddings(self, noisePercent=0, masking=False):
        allDataset = self.dataset.getDataIndex(self.dataset.user, 1.0)
        x, y, _ = self.dataset.getDataset(allDataset, noisePercent=noisePercent, interleaved=False, 
                                          shuffled=False, normalized=True, masking=masking)
        predicted = self.network.predict(x)
                
        unitId = 0
        lastPersonId = y[0]
        for i in range(len(y)):
            personId = y[i]
            if personId != lastPersonId: 
                unitId = 0
                lastPersonId = personId
                
            if (personId, unitId) not in self.embeddingTable:
                self.embeddingTable[(personId, unitId)] = predicted[i]
                
            unitId = unitId+1
                
    #returns the embedded vectors and Ids, related to the dataindex
    def getDataset(self, dataIndex):
        x, y = [], []
        for key in dataIndex:
            for unitId in dataIndex[key]:
                x.append(self.getEmbedded(key, unitId))
                y.append(key)
        return(np.asarray(x), np.asarray(y))
    
    #returns the embedded vectors, units and Ids, related to the dataindex
    def getDatasetAugmented(self, dataIndex):
        unitMean = self.dataset.getMeanUnit(dataIndex, normalized=True)
        x, y, u, m = [], [], [], []
        for key in dataIndex:
            for unitId in dataIndex[key]:
                x.append(self.getEmbedded(key, unitId))
                u.append(self.dataset.getUnitStep(key, unitId))
                m.append(unitMean[key])
                y.append(key)
        return(np.asarray(x), np.asarray(y), np.asarray(u), np.asarray(m))
    
    #Calculates the Centroid of embedded vectors from and index dataset
    def getCentroids(self, dataIndex):
        centroids = {}
        for key in dataIndex:
            if len(dataIndex[key]) > 0:                
                embeddings = np.asarray([self.getEmbedded(key, unitId) for unitId in dataIndex[key]])
                centroids[key] = np.sum(embeddings, axis=0) / len(dataIndex[key])
        return centroids
    
    #Calculates Sigma of embedded vectors from and index dataset
    def getSigma(self, dataIndex):
        sigma = {}
        for key in dataIndex:
            if len(dataIndex[key]) > 0:
                embeddings = np.asarray([self.getEmbedded(key, unitId) for unitId in dataIndex[key]])
                standard = np.std(embeddings, axis = 0)
                sigma[key] = np.sum(standard) / len(standard)
        return sigma
    
    #Save the embeddings dataset
    def save(self, path, fileName):
        #for key in self.dataset.unitsMap:
        #    for i in range(len(self.dataset.unitsMap[key])):
        #        update = self.getEmbedded(key, i)  #be sure all the embeddings were created
        np.save(path + "/" + fileName, self.embeddingTable)
        
    #Load the embeddings dataset
    def load(self, path, fileName):        
        self.embeddingTable = np.load(path + "/" + fileName, allow_pickle='TRUE').item()
                
    #Returns the mean encoded
    def getMeanEncoded(self, dataIndex):
        mean = self.dataset.getMeanUnit(dataIndex, normalized=False)
        x, y =[], []
        for key in mean:
            x.append(mean[key])
            y.append(key)
        #Encode the mean
        x = np.array(x)
        press_x =np.asarray(x[:, :, 0:16])
        acc_x = np.asarray(x[:, :, 16:22])
        gyro_x = np.asarray(x[:, :, 22:])
        x = [press_x, acc_x, gyro_x]
        encoded = self.network.predict(x)
        return(encoded, y)
        
    #Compute the Tsne that represents the embedding vectors of dataIndex
    def getTsne(self, dataIndex):
        #dataset of centroids
        centroids = self.getCentroids(dataIndex)
        c, yc = [],[]
        for key in centroids:
            c.append(centroids[key])
            yc.append(key)
        yc = np.array(yc)

        #Dataset of embeddings
        x, y  = self.getDataset(dataIndex)        
        x_train = np.concatenate((x, c), axis=0)                
        
        #Dataset of mean
        m, ym = np.array([]), np.array([])
        if self.network != None: 
            m, ym = self.getMeanEncoded(dataIndex)
            x_train = np.concatenate((x_train, m), axis=0)        
            
        x_tsne = TSNE(n_components=2, perplexity=10, init='pca', n_iter=1000, random_state=0).fit_transform(x_train)
        x_min, x_max = np.min(x_tsne, 0), np.max(x_tsne, 0)
        x_tsne = (x_tsne - x_min) / (x_max - x_min)        
        return(x_tsne[0:len(x),:], y, x_tsne[len(x):len(x)+len(c),:], yc, x_tsne[len(x)+len(c):len(x_tsne),:], ym)
                
    # Scale and visualize the embedding vectors
    def plotTsne(self, dataIndex, title=None):
        X, Yx, C, Yc, M, Ym = self.getTsne(dataIndex)        
        cmap = plt.cm.get_cmap('gist_rainbow')
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))  # setup the plot  
        index = np.linspace(0, 256, max(Yx)).astype("int")
        color = dict(zip(np.unique(Yx), index))         
        
        for i in range(len(Yx)):
            plt.text(X[i, 0], X[i, 1], str("."), color=cmap(color[Yx[i]]),fontdict={'weight': 'bold', 'size': 8})

        #for i in range(len(Yc)):
        #    plt.text(C[i, 0], C[i, 1], str("x"), color="blue",fontdict={'weight': 'bold', 'size': 8})
            
        #for i in range(len(Ym)):
        #    plt.text(M[i, 0], M[i, 1], str("*"), color="black",fontdict={'weight': 'bold', 'size': 12})
        
        plt.title(title)        
        plt.axis('off')
        plt.show()        
        
    #Compute the Tsne that represents the embedding vectors of dataIndex
    def getTsneCompared(self, dataIndex, dataIndex2):
        x, y = self.getDataset(dataIndex)
        m, ym = self.getMeanEncoded(dataIndex)
        x2, y2 = self.getDataset(dataIndex2)
        x_train = np.concatenate((x, x2), axis=0)
        x_train = np.concatenate((x_train, m), axis=0)
        
        x_tsne = TSNE(n_components=2, perplexity=10, init='pca', n_iter=1000, random_state=0).fit_transform(x_train)
        x_min, x_max = np.min(x_tsne, 0), np.max(x_tsne, 0)
        x_tsne = (x_tsne - x_min) / (x_max - x_min)        
        limit2 = len(x) + len(x2)
        return(x_tsne[0:len(x),:], y,  x_tsne[len(x):limit2,:], y2,  x_tsne[limit2:len(x_tsne),:], ym)
    
    
    # Scale and visualize the embedding vectors
    def plotTsneCompared(self, dataIndex, dataIndex2, title=None, prototype="*"):
        X, Yx, X2, Y2, M, Ym = self.getTsneCompared(dataIndex, dataIndex2)
        #print(len(X), len(Yx), len(X2), len(Y2), len(M), len(Ym))
        cmap = plt.cm.get_cmap('gist_rainbow')
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))  # setup the plot  
        index = np.linspace(0, 256, len(Ym)).astype("int")
        color = dict(zip(Ym, index))         

        for i in range(len(Yx)):
            plt.text(X[i, 0], X[i, 1], str("."), color=cmap(color[Yx[i]]),fontdict={'weight': 'bold', 'size': 8})
        
        #cmap = plt.cm.get_cmap('Dark2')
        for i in range(len(Y2)):
            plt.text(X2[i, 0], X2[i, 1], str("*"), color="Red",fontdict={'weight': 'bold', 'size': 8})

        for i in range(len(Ym)):
            plt.text(M[i, 0], M[i, 1], str(prototype), color="black",fontdict={'weight': 'bold', 'size': 12})
        
        plt.title(title)        
        plt.axis('off')
        plt.show()        