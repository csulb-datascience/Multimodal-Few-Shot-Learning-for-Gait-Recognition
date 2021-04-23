# csulb-datascience
#
# Authors: 
#      Nelson Minaya, email: nelson.minaya@student.csulb.edu
#      Nhat Anh Le,   email: nhat.le01@student.csulb.edu
#
# Class version: 2.0
# Date: July 2020
#
# Include a reference to this site if you will use this code.

import keras as keras
from keras.layers import LSTM
from keras.layers import Conv1D
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import concatenate
from keras.layers import Lambda
from keras.models import Model
from keras.optimizers import Adam
from Recognition_TripletLoss_V2 import TripletSemiHardLoss

class EncoderModel:
    #constructor
    def __init__(self):
        self.embeddingSize = 128
        self.featuresPress = 16
        self.featuresAcc = 6
        self.featuresGyro = 6        
                
    #Load a model from disk in h5 format
    def loadModel(self, fileName):
        #model = keras.models.load_model(fileName, compile=False)        
        model = keras.models.load_model(fileName, custom_objects={'keras':keras}) #custom_objects={'TripletSemiHardLoss':TripletSemiHardLoss})
        return(model)
        
    #****************************************************************************************************
    # CNN
    #****************************************************************************************************
        
    #Network used for each sensor individualy
    def getBranchCNN(self, inputs):
        x = Conv1D(filters=32, kernel_size=20, padding='Same', activation="relu")(inputs)
        x = BatchNormalization()(x)
        x = Conv1D(filters = 64, kernel_size=20, padding = 'Same', activation ='relu')(x)
        x = BatchNormalization()(x)
        x = Conv1D(filters = 128, kernel_size=20, padding = "Same", activation = "relu")(x)
        x = BatchNormalization()(x)
        x = Flatten()(x)
        return(x)
    
    #Merge the branchs of each sensor in a unique network
    def getNetworkEncoderCNN(self, inputs):
        CNN_press = self.getBranchCNN(inputs[0])
        CNN_acc = self.getBranchCNN(inputs[1])
        CNN_gyro = self.getBranchCNN(inputs[2])
    
        # Combine the outputs of the CNNs and complete other layers
        combinedOutput = concatenate([CNN_press, CNN_acc, CNN_gyro])
        x = Dense(256, activation="relu", name="dense_combined")(combinedOutput)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)    
        
        #normalize embedding
        x = Dense(self.embeddingSize, name="embedding")(x)        
        x = Lambda(lambda x: keras.backend.l2_normalize(x, axis=1), name="encoder")(x)        
        return(x)
        
    #Merge the branchs of each sensor in a unique network
    def getUniModalEncoderCNN(self, inputs, sensor):
        branchOutput = self.getBranchCNN(inputs[sensor])
        x = Dense(256, activation="relu", name="dense_combined")(branchOutput)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)    
        
        #normalize embedding
        x = Dense(self.embeddingSize, name="embedding")(x)        
        x = Lambda(lambda x: keras.backend.l2_normalize(x, axis=1), name="encoder")(x)        
        return(x)
    
    #Returns the model of the encoder
    def getEncoderCNN(self, unitSize, sensor=None):
        inputPress = Input(shape = (unitSize, self.featuresPress), name="input_press")
        inputAcc = Input(shape = (unitSize, self.featuresAcc), name="input_acc")
        inputGyro = Input(shape = (unitSize, self.featuresGyro), name="input_gyro")
        inputs = [inputPress, inputAcc, inputGyro]
        embeddings = self.getNetworkEncoderCNN(inputs)
        if sensor != None: embeddings = self.getUniModalEncoderCNN(inputs, sensor)            
        model = Model(inputs=inputs, outputs=embeddings)
        return(model)
    
    #Return the compiled model using the triplet loss function
    def getCompiledCNN(self, unitSize, alpha, beta , learningRate, sensor=None):
        model = self.getEncoderCNN(unitSize, sensor)
        model.compile( optimizer=Adam(learningRate), loss=TripletSemiHardLoss(alpha, beta))
        return(model)
    

    
    #****************************************************************************************************
    # LSTM
    #****************************************************************************************************
        
    #Network used for each sensor individualy
    def getBranchLSTM(self, inputs):
        hidden_cells = 128        
        x = LSTM(hidden_cells, return_sequences=True, recurrent_dropout=0.2)(inputs)
        x = LSTM(hidden_cells)(x)
        return(x)
    
    #Merge the branchs of each sensor in a unique network    
    def getNetworkEncoderLSTM(self, inputs):
        LSTM_press = self.getBranchLSTM(inputs[0])
        LSTM_acc = self.getBranchLSTM(inputs[1])
        LSTM_gyro = self.getBranchLSTM(inputs[2])

        # Combine the outputs of the LSTMs and complete other layers
        combinedOutput = concatenate([LSTM_press, LSTM_acc, LSTM_gyro])
        x = Dense(128, activation="relu", name="dense_combined_LSTM")(combinedOutput)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)

        #Embedding
        x = Dense(self.embeddingSize, name="embedding_LSTM")(x)    
        x = Lambda(lambda x: keras.backend.l2_normalize(x, axis=1), name="encoder_LSTM")(x)            
        return x

    #Merge the branchs of each sensor in a unique network    
    def getUniModalEncoderLSTM(self, inputs, sensor):
        branchOutput = self.getBranchLSTM(inputs[sensor])
        x = Dense(128, activation="relu", name="dense_combined_LSTM")(branchOutput)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)

        #Embedding
        x = Dense(self.embeddingSize, name="embedding_LSTM")(x)    
        x = Lambda(lambda x: keras.backend.l2_normalize(x, axis=1), name="encoder_LSTM")(x)            
        return x
    
    #Returns the model of the encoder
    def getEncoderLSTM(self, unitSize, sensor=None):
        inputPress = Input(shape = (unitSize, self.featuresPress), name="input_press")
        inputAcc = Input(shape = (unitSize, self.featuresAcc), name="input_acc")
        inputGyro = Input(shape = (unitSize, self.featuresGyro), name="input_gyro")
        inputs = [inputPress, inputAcc, inputGyro]
        embeddings = self.getNetworkEncoderLSTM(inputs)        
        if sensor != None: embeddings = self.getUniModalEncoderLSTM(inputs, sensor)
        
        model = Model(inputs=inputs, outputs=embeddings)
        return(model)
        
    #Return the compiled model using the triplet loss function        
    def getCompiledLSTM(self, unitSize, alpha, beta, learningRate, sensor=None):
        # Return the final Model
        model = self.getEncoderLSTM(unitSize, sensor)
        model.compile( optimizer=Adam(learningRate), loss=TripletSemiHardLoss(alpha, beta))
        return(model)    
        
    
    #****************************************************************************************************
    # ENSEMBLE
    #****************************************************************************************************

    def getEnsemble(self, unitSize, sensor=None):                
        #Common Input
        inputPress = Input(shape = (unitSize, self.featuresPress), name="input_press")
        inputAcc = Input(shape = (unitSize, self.featuresAcc), name="input_acc")
        inputGyro = Input(shape = (unitSize, self.featuresGyro), name="input_gyro")
        inputEnsemble = [inputPress, inputAcc, inputGyro]
        
        #CNN encoder
        emsembleCNN = self.getNetworkEncoderCNN(inputEnsemble)
        if sensor != None: emsembleCNN = self.getUniModalEncoderCNN(inputEnsemble, sensor)            
        
        #LSTM encoder
        emsembleLSTM = self.getNetworkEncoderLSTM(inputEnsemble)
        if sensor != None: emsembleLSTM = self.getUniModalEncoderLSTM(inputEnsemble, sensor)            
        
        #Concatenated and normalize the output        
        #combinedOutput = concatenate([encoderCNN.output, encoderLSTM.output])        
        combinedOutput = concatenate([emsembleCNN, emsembleLSTM])
        embedding = Lambda(lambda x: keras.backend.l2_normalize(x, axis=1))(combinedOutput)

        #return the model
        #model = Model(inputs=[encoderCNN.input, encoderLSTM.input], outputs=combinedOutput)        
        model = Model(inputs=inputEnsemble, outputs=embedding)
        return(model)
    
    #Return the compiled model using the triplet loss function            
    def getCompiledEnsemble(self, unitSize, alpha, beta, learningRate, sensor=None):
        # Return the final Model
        model = self.getEnsemble(unitSize, sensor)
        model.compile(optimizer=Adam(learningRate), loss=TripletSemiHardLoss(alpha, beta))   
        return(model)    
    
    
    #****************************************************************************************************
    # OTHERS
    #****************************************************************************************************
    
    def getResults(self, history):
        #get the values
        lossTrain, lossValid = -1, -1
        if "loss" in history.history.keys(): lossTrain = history.history["loss"][-1]
        if "val_loss" in history.history.keys(): lossValid = history.history["val_loss"][-1]        
        return(lossTrain, lossValid)    
        