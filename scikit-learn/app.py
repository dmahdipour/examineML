import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.svm import SVC

df=pd.read_csv('../Datasets/data.csv')
print('=========================================================================================================================')
print('          Data set Basicly Status         ')
print('=========================================================================================================================')
print(df.describe())


#Remove unormal FBS data 
temp=df[df['FBS']>115].index
df=df.drop(temp)

#Remove unormal Cholesterol data 
temp=df[df['Cholesterol']>200].index
df=df.drop(temp)

#Remove unormal Triglycerides data 
temp=df[df['Triglycerides']>150].index
df=df.drop(temp)
print('=========================================================================================================================')
print('          Data set Status After cleaning         ')
print('=========================================================================================================================')
print(df.describe())

#seprate X , Y from all data
X=df[['Age','FBS','Cholesterol','Triglycerides','HDL Cholesterol','HGB','HCT']]
Y=df[['Creatinine']]

#Seprate X, Y to train and test 
Y=data_binary=preprocessing.Binarizer(threshold=1.2).transform(Y.values)
x_train, x_test, y_train, y_test=train_test_split(X,Y, test_size=0.2, random_state=1)
print('=========================================================================================================================')
print('          Train and Test Shapes         ')
print('=========================================================================================================================')
print('x_train shape: ', x_train.shape,'x_test shape: ', x_test.shape, 'y_train shape: ', y_train.shape, 'y_test shape: ', y_test.shape)

#KNN
classifier_knn = KNeighborsClassifier(n_neighbors = 2)
classifier_knn.fit(x_train, y_train.reshape(-1))

y_pred = classifier_knn.predict(x_test)
Accuracy=metrics.accuracy_score(y_test, y_pred)

print('=========================================================================================================================')
print('          KNN         ')
print('=========================================================================================================================')
print('Accuracy: ', Accuracy)

