# csulb-datascience
#
# Authors: 
#      Nelson Minaya, email: nelson.minaya@student.csulb.edu
#      Nhat Anh Le,   email: nhat.le01@student.csulb.edu
#      Sella Bae,     email: sella.bae@student.csulb.edu
#
# Class version: 1.1
# Date: Nov 2020
#
# Include a reference to this site if you will use this code.

import pandas as pd
import numpy as np
import random

class Dataset:
    def __init__(self, path=None, fileName=None):
        self.user = []
        self.unitsMap = {}
        self.trainingSet = {}
        self.validationSet = {}
        self.unseenSet = {}
        self.testSet = {}
        self.mask = []
        
        #upload the data if required
        if fileName != None: self.loadData(path, fileName)

    #return the size of the unit step if there is at least a user in the dataset
    def unitSize(self):
        size = 0 if len(self.user)==0 else len(self.unitsMap[self.user[0]][0])
        return(size)
    
    #return the indicated unit step 
    def getUnitStep(self, personId, unitId):
        if personId in self.unitsMap:
            if unitId < len(self.unitsMap[personId]):
                return(self.unitsMap[personId][unitId])
        return(None)

    #Save the datasets on disk
    def saveSets(self, path, fileName, includeMap = False):
        #By default save all subsets
        summary = {"training":self.trainingSet,"validation":self.validationSet,
                   "unseen":self.unseenSet, "test":self.testSet}
        #Add the map if required
        if includeMap:
            summary["map"] = self.unitsMap
            summary["user"] = self.user            
            
        #save it as a numpy array
        np.save(path + "/" + fileName, summary)
                
    #load the dataset index saved with saveSets
    def loadSets(self, path, fileName):
        summary = np.load(path + "/" + fileName, allow_pickle='TRUE').item()
        self.trainingSet = summary["training"]
        self.validationSet = summary["validation"]
        self.unseenSet = summary["unseen"]
        self.testSet = summary["test"]
        
        #Load the map if it is available
        if "map" in summary.keys():
            self.user = summary["user"]
            self.unitsMap = summary["map"]
                 
    #Reads the dataset from disk and creates it in memory as a dictionary
    def loadData (self, path, fileName):
        data = pd.read_csv(path + "/" + fileName)
        self.initSets()
        self.user = np.array(data['user_id'].unique()).astype(int)
        self.unitsMap = self.get_data_dictionary(data, self.user)
        
    #clean the variables related to the dataset
    def initSets(self):
        self.unitsMap.clear()
        self.trainingSet.clear()
        self.validationSet.clear()
        self.unseenSet.clear()
        self.testSet.clear()
    
    #Given a data set, and a list of ids, return a dictionary 
    #that map user id to their list of images
    def get_data_dictionary(self, data, ids):
        table = {}
        for person_id in ids:
            list_of_images = []
            current_person = data[data['user_id'] == person_id]
            unique_unit_ids = current_person['unit_id'].unique()
            for unit_id in unique_unit_ids:
                current_unit = current_person[current_person['unit_id'] == unit_id]
                current_image = current_unit.iloc[:, 4:]
                img = np.array(current_image.values)
                unit = np.concatenate((img[:,0:8],img[:,14:22],img[:,8:11],img[:,22:25],img[:,11:14],img[:,25:28]), axis=1)                
                list_of_images.append(unit)
            table[person_id] = np.asarray(list_of_images)
        return table

    #splits the dataset by selecting randomly the people for training and the percentage
    #training set -> % units of numberPeopleTraining;  (% = percentTraining)
    #validation set -> (1-percentTraining) units of numberPeopleTraining
    #test set -> (1-percentTraining) units of numberPeopleTraining + 100% of remaining users            
    def randomSplit(self, numberPeopleTraining, percentUnits):
        trainingPeople = set(random.sample(list(self.user), numberPeopleTraining))
        self.trainingSet = self.getDataIndex(trainingPeople, percentUnits)
        self.validationSet = self.getDataIndex(trainingPeople, percentUnits, complement=True)
        self.unseenSet = self.getDataIndex(self.notTrainingPeople(), 1.0) #100%
        self.testSet = self.validationSet.copy()
        self.testSet.update(self.unseenSet)  #test set includes validation and unseen data

    def split(self, numberPeopleTraining, numberPeopleValidation):
        """Splits the dataset in three parts
        training set -> 100% units of numberPeopleTraining;
        validation set -> 100% units of numberPeopleValidation;
        unseen set ->  100% units of remaining people;
        test set -> validation + unseen
        """
        trainingPeople = set(random.sample(list(self.user), numberPeopleTraining))
        self.split_by_id(trainingPeople, numberPeopleValidation)
        
    def split_by_id(self, trainingPeopleIds, numberPeopleValidation):
        """Splits the dataset in three parts by specified ids for training set
        training set -> 100% units of number of trainingPeopleIds
        validation set -> 100% units of numberPeopleValidation;
        unseen set ->  100% units of remaining people;
        test set -> validation + unseen
        """
        for id_ in trainingPeopleIds:
            if id_ not in self.user:
                raise ValueError("trainingPeopleIds contains invalid user id")
            
        trainingPeople = set(trainingPeopleIds)
        self.trainingSet = self.getDataIndex(trainingPeople, 1.0) #100%
        
        remainder = set(self.user) - trainingPeople
        validationPeople = set(random.sample(list(remainder), numberPeopleValidation))
        self.validationSet = self.getDataIndex(validationPeople, 1.0) #100%
        
        remainder = (set(self.user) - trainingPeople) - validationPeople
        self.unseenSet = self.getDataIndex(remainder, 1.0) #100%
        
        self.testSet = self.validationSet.copy()
        self.testSet.update(self.unseenSet)  #test set includes validation and unseen data
        
    #return the set of people included in the training dataset        
    def trainingPeople(self):
        return(set(self.trainingSet.keys()))
               
    #return the set of people that is not included in the training dataset
    def notTrainingPeople(self):
        return(set(self.user) - self.trainingPeople()) 

    #return a percentage of indexes to the units of users, according to complement:
    #Complement=False returns [0..percentage] indexes from the list of units
    #Complement=True returns [percentage..length] indexes from the list of units
    def getDataIndex (self, users, percentage, complement=False):
        dataIndex = {}
        for person in users:
            limit = int(percentage * len(self.unitsMap[person]))
            endIndex = limit if not complement else len(self.unitsMap[person])
            iniIndex = 0 if not complement else limit
            dataIndex[person] = list(range(iniIndex, endIndex))
        return dataIndex                
    
    #return the total number of units in the dataset index and 
    #the number of units per Id
    def dataIndexSize(self, dataIndex):
        byId = {}
        size = 0
        for key in dataIndex:
            byId[key] = len(dataIndex[key])
            size = size + len(dataIndex[key])
        return(size, byId)
    
    #return the mean unit for each person Id from a dataset index
    def getMeanUnit(self, dataIndex, normalized=True, masking=False):
        results={}
        for key in dataIndex:
            units = [self.unitsMap[key][i] * (1 if masking==False else self.mask)  for i in dataIndex[key]]
            mean = np.mean(units, axis=0)
            if normalized:
                n = np.linalg.norm(mean, axis=0)
                mean = np.divide(mean, n, out=np.zeros_like(mean), where=n!=0)                
            results[key] = mean
        return(results)

    #return the mean unit for all people in a dataset index
    def getMeanUnitAll(self, dataIndex):
        units=[]
        for key in dataIndex:
            for i in dataIndex[key]:
                units.append(self.unitsMap[key][i])
                
        mean = np.mean(units, axis=0)
        return(mean)
    
    #returns a random binary matrix with a percentage of zeroes
    def getNoiseMatrix(self, shape2D, noisePercent):
        N= shape2D[0] * shape2D[1]
        K = (N*noisePercent)//100
        arr = np.array([0] * K + [1] * (N-K))
        np.random.shuffle(arr)
        return(arr.reshape(shape2D))        
        
    #Transform the dataset to a noisy dataset
    def turnToNoisy(self, noisePercent=0):
        for key in self.unitsMap:
            for i in range(len(self.unitsMap[key])):
                unit = np.array(self.unitsMap[key][i])
                noise = self.getNoiseMatrix(unit.shape, noisePercent)
                self.unitsMap[key][i] = unit * noise
        
    #Convert a index dataset to array dataset        
    def toArray(self, dataIndex, noisePercent=0, normalized=True, shuffled=False, masking=False):
        unitMean = self.getMeanUnit(dataIndex, normalized, masking)
        data = []
        for key in dataIndex.keys():
            for i in dataIndex[key]:
                unit = np.array(self.unitsMap[key][i])
                noise = self.getNoiseMatrix(unit.shape, noisePercent)
                if masking==True: unit = unit * self.mask
                data.append((key, unit*noise, unitMean[key]))
        
        data = np.array(data)
        if shuffled==True: np.random.shuffle(data)
        y = np.array([key for key in data[:,0]])
        x = np.array([unit for unit in data[:,1]])
        m = np.array([mean for mean in data[:, 2]])
        return(x, y, m)

    
    # Returns the dataset where the units are interleaved: 
    # one unit person 1, one unit person 2, ..., one unit person n, one unit person 1, ...
    def toArrayInterleaved(self, dataIndex, noisePercent=0, normalized=True, masking=False):
        unitMean = self.getMeanUnit(dataIndex, normalized)
        x, y, m = [], [], []
        counter, numUnits = 0, 0
        totalUnits, _ = self.dataIndexSize(dataIndex)        
        while (numUnits < totalUnits):
            for key in dataIndex.keys():
                if counter < len(dataIndex[key]):
                    unit = np.array(self.unitsMap[key][counter])
                    noise = self.getNoiseMatrix(unit.shape, noisePercent)   
                    if masking==True: unit = unit * self.mask                    
                    x.append(unit * noise)
                    y.append(key)
                    m.append(unitMean[key])
                    numUnits +=1                    
            counter += 1 
        return(np.asarray(x), np.asarray(y), np.asarray(m))

    #return the dataset generated by an index dataset
    def getDataset(self, dataIndex, noisePercent=0, batchSize=1, shuffled=True, interleaved=False, normalized=True, masking=False):
        x, y, m = self.toArrayInterleaved(dataIndex, noisePercent, normalized, masking) \
               if interleaved else self.toArray(dataIndex, noisePercent, normalized, shuffled, masking) 
    
        #Drop the last batch if it is not enough big
        length = batchSize * (len(y) // batchSize)
        if len(y)-length >= 2 * len(np.unique(y)): length = len(y)
            
        press_x = np.asarray(x[:length, :, 0:16])
        acc_x = np.asarray(x[:length, :, 16:22])
        gyro_x = np.asarray(x[:length, :, 22:])
        return([press_x, acc_x, gyro_x], y[:length], m[:length])
    
        #x = x.reshape((x.shape[0], x.shape[1], x.shape[2], 1))
        #x = np.stack([x]*3, axis=-1)
        #return(x[:length], y[:length], m[:length])

    
    #return the dataset with an additional Y added to X
    def getDatasetAugmented(self, dataIndex, noisePercent=0, batchSize=1, shuffled=True, interleaved=False):
        x, y, m = self.toArrayInterleaved(dataIndex, noisePercent) \
               if interleaved else self.toArray(dataIndex, noisePercent, shuffled=shuffled) 
        
        #Drop the last batch if it is not enough big
        length = batchSize * (len(y) // batchSize)
        if len(y)-length >= 2 * len(np.unique(y)): length = len(y)
                
        press_x = np.asarray(x[:length, :, 0:16])
        acc_x = np.asarray(x[:length, :, 16:22])
        gyro_x = np.asarray(x[:length, :, 22:])
        return([press_x, acc_x, gyro_x, y[:length], m[:length]], y[:length])
    
    
    #return the training dataset
    def getTrainingDataset(self, noisePercent=0, batchSize=1):
        return(self.getDataset(self.trainingSet, noisePercent, batchSize))
                
    #return the validation dataset
    def getValidationDataset(self, noisePercent=0, batchSize=1):
        return(self.getDataset(self.validationSet, noisePercent, batchSize))
    
    #Return k random units step indexes for each id
    def selectRandomUnits(self, dataIndex, k):
        result = dict()
        complement = dict()
        for key in dataIndex:
            if k >= len(dataIndex[key]):
                result[key] = dataIndex[key].copy()
            else:
                units = random.sample(dataIndex[key], k)
                result[key] = units
                complement[key] = list(set(dataIndex[key]) - set(units))
        return result, complement

    #Return k units step indexes for each id
    def selectKUnits(self, dataIndex, k):
        result = dict()
        complement = dict()
        for key in dataIndex:
            if k >= len(dataIndex[key]):
                result[key] = dataIndex[key].copy()
            else:
                units = dataIndex[key][0:k]
                result[key] = units.copy()
                complement[key] = list(set(dataIndex[key]) - set(units))
        return result, complement

    
    #return the training people with k random unit step indexes
    def selectRandomTrainingUnits (self, k):
        training, _ = self.selectRandomUnits(self.trainingSet, k)
        return(training)
    