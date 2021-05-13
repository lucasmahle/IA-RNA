import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class RedeNeural:
    def setupTrainData(self, fileName):
        self.dataSet = self.getInputaData(fileName)
        self.removeOutliers()
        self.splitDataIntoTrainAndValidation()

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def sigmoid_prime(self, x):
        s = self.sigmoid(x)
        return s * (1 - s)

    def getInputaData(self, fileName):    
        dataset = pd.read_csv(fileName)
        dataset.drop(['Hora','Tamanho','Referencia'],axis=1,inplace=True)
        return dataset

    def removeOutliers(self):
        z_scores = zscore(self.dataSet)
        abs_z_scores = np.abs(z_scores)
        filtered_entries = (abs_z_scores < 3).all(axis=1)
        self.dataSet = self.dataSet[filtered_entries]

    def splitDataIntoTrainAndValidation(self):
        scaler=StandardScaler()
        DataScaled=scaler.fit_transform(self.dataSet)
        DataSetScaled=pd.DataFrame(np.array(DataScaled),columns = ['NumAmostra', 'Area', 'Delta', 'Output1','Output2'])
        X = DataSetScaled.drop(['Output1', 'Output2'],axis=1)
        y = self.dataSet[['Output1','Output2']]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)
        self.trainingDataInput = X_train
        self.trainingDataOutput = y_train
        
        self.validatingDataInput = X_test
        self.validatingDataOutput = y_test

    def setupNeuralNetwork(self):
        self.n_input = 3
        self.n_hidden = 12
        self.n_output = 2
        self.learnrate = 0.05
        self.epochs = 15000

        self.weights_input_hidden = np.random.normal(0, scale=0.1, size=(self.n_input, self.n_hidden))
        self.weights_hidden_output = np.random.normal(0, scale=0.1, size=(self.n_hidden, self.n_output))


    def validateTrainingData(self):
        if not hasattr(self, 'trainingDataInput'):
            raise Exception("Training data was not seted")
        
    def runBackPropagation(self, verbose):
        self.validateTrainingData()

        n_records, n_features = self.trainingDataInput.shape
        last_loss=None

        for e in range(self.epochs):
            delta_w_i_h = np.zeros(self.weights_input_hidden.shape)
            delta_w_h_o = np.zeros(self.weights_hidden_output.shape)

            for xi, yi in zip(self.trainingDataInput.values, self.trainingDataOutput.values):
                hidden_layer_input = np.dot(xi, self.weights_input_hidden)
                hidden_layer_output = self.sigmoid(hidden_layer_input)
            
                output_layer_in = np.dot(hidden_layer_output, self.weights_hidden_output)
                output = self.sigmoid(output_layer_in)
            
                error = yi - output
                output_error_term = error * self.sigmoid_prime(output_layer_in)

                hidden_error = np.dot(self.weights_hidden_output,output_error_term)
                hidden_error_term = hidden_error * self.sigmoid_prime(hidden_layer_input)
            
                delta_w_h_o += output_error_term * hidden_layer_output[:, None]
                delta_w_i_h += hidden_error_term * xi[:, None]
                
            self.weights_input_hidden += self.learnrate * delta_w_i_h / n_records
            self.weights_hidden_output += self.learnrate * delta_w_h_o / n_records
            
            if  verbose and e % (epochs / 20) == 0:
                hidden_output = self.sigmoid(np.dot(xi, self.weights_input_hidden))
                out = self.sigmoid(np.dot(hidden_output, self.weights_hidden_output))
                loss = np.mean((out - yi) ** 2)

                if last_loss and last_loss < loss:
                    print("Erro quadrático no treinamento: ", loss, " Atenção: O erro está aumentando")
                else:
                    print("Erro quadrático no treinamento: ", loss)
                last_loss = loss

    def getWeightsHidden(self):
        return self.weights_input_hidden
        
    def getWeightsOutput(self):
        return self.weights_hidden_output                
    
    def showAccuracy(self): 
        n_records, n_features = self.validatingDataInput.shape
        predictions=0

        for xi, yi in zip(self.validatingDataInput.values, self.validatingDataOutput.values):
            hidden_layer_input = np.dot(xi, self.weights_input_hidden)
            hidden_layer_output = self.sigmoid(hidden_layer_input)
         
            output_layer_in = np.dot(hidden_layer_output, self.weights_hidden_output)
            output = self.sigmoid(output_layer_in)

            if (output[0] > output[1]):
                if (yi[0] > yi[1]):
                    predictions += 1
                    
            if (output[1] >= output[0]):
                if (yi[1] > yi[0]):
                    predictions += 1

        acuracia = predictions/n_records
        print("A Acurácia da Predição é de: {:.3f}".format(acuracia))

def run():
    obj = RedeNeural()
    obj.setupNeuralNetwork()
    obj.setupTrainData('arruela_.csv')
    obj.runBackPropagation(not True)
    obj.showAccuracy()

run()