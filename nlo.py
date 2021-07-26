import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
from sklearn.model_selection import train_test_split
data = pd.read_csv (r'C:\Users\idont\Downloads\diabetes.csv') # leemos archivo

class Logistica_Neuron:
    def __init__(self, n_input, learning_rate=0.1):
        self.w = -1 + 2 * np.random.rand(n_input)
        self.b = -1 + 2 * np.random.rand()
        self.eta = learning_rate
    
    def predict_proba(self, X):
        Z= np.dot(self.w,X)+self.b
        Y_est=1/(1+ np.exp(-Z))
        return Y_est
    
    def predict(self, X):#predicir de categoria discreta
        Z= np.dot(self.w,X)+self.b
        Y_est=1/(1+ np.exp(-Z))
        return 1 * (Y_est > 0.5)
    
    def train(self,X,Y, epochs=200):
        p=X.shape[1]
        for _ in range(epochs):
            Y_est= self.predict_proba(X)
            YT=(Y-Y_est)
            self.w += (self.eta/p) *np.dot((Y-Y_est),X.T).ravel()
            self.b += (self.eta/p) * np.sum(Y -Y_est)
def compare(n,m,long):
    i=0
    c=0 
    for i in range(len(long)):
        if(n[i] == m[i]):
            c=c+1
        i=i+1
    return c

#inicio
X0= data.drop("Outcome", axis=1).values                        
Y0 =  np.array(pd.DataFrame(data, columns=["Outcome"]))       
xtrain, xtest, ytrain, ytest = train_test_split(X0,Y0,  test_size=0.3, random_state=0, shuffle=True)

v1=ytest
v2=ytrain

neuron = Logistica_Neuron(8,1)
print("\nPredicciones:") 
print("\n-Test:") 
a1=neuron.predict(xtest.T)
print(a1)

print("\n-Entrenamiento:") 
a2=neuron.predict(xtrain.T)
print(a2)

neuron.train(xtrain.T,ytrain.T, epochs=2000)
print("\nValores Verdaderos Testing:") 
print(v1.reshape(1,-1))

print("\nValores Verdaderos Training:") 
print(v2.reshape(1,-1))

#porcentaje del testing
c=compare(a1,v1,ytest)
print("\nTesting:")     
print(c, "/", len(ytest)) 
print("Precision: ", c/len(ytest))

#porcentaje del training
c=compare(a2,v2,ytrain)
print("\nTraining:")         
print(c, "/", len(ytrain)) 
print("Precision: ", c/len(ytrain))