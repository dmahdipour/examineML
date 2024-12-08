import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing

df=pd.read_csv('../Datasets/data.csv')
print('--------------------Describe---------------------')
print(df.describe())


print('---------------------Cratine Describe--------------------')
y=pd.Series(df['Creatinine'])
print(y.describe())
print('------------------------Cratine Head-----------------')
print(y.head(10))

#print(df[3:5])
#print(df.loc[3:5])
#print(df.iloc[3:5])

#print(df[df['Creatinine']>15])

FBS=df['FBS'][0:250]
Cholesterol=df['Cholesterol'][0:250]
Triglycerides=df['Triglycerides'][0:250]
HDLCholesterol=df['HDL Cholesterol'][0:250]
HGB=df['HGB'][0:250]
HCT=df['HCT'][0:250]


Creatinine=df['Creatinine'][0:250]

x=np.arange(len(FBS))

plt.figure(figsize=(9, 3))

plt.subplot(241)
plt.plot(x,FBS)
plt.ylabel('FBS')

plt.subplot(242)
plt.plot(x,Cholesterol,'r')
plt.ylabel('Cholesterol')

plt.subplot(243)
plt.plot(x,Triglycerides,'g')
plt.ylabel('Triglycerides')

plt.subplot(244)
plt.plot(x,HDLCholesterol,'y')
plt.ylabel('HDLCholesterol')

plt.subplot(245)
plt.plot(x,HGB,'b')
plt.ylabel('HGB')

plt.subplot(246)
plt.plot(x,HCT,'r')
plt.ylabel('HCT')



plt.subplot(248)
plt.plot(x,Creatinine,'y')
plt.ylabel('Creatinine')


plt.show()


print('------------------------Creatinine classes-----------------')
cratineClasses=preprocessing.Binarizer(threshold=1.2).transform(df['Creatinine'].array.reshape(-1, 1))
print(cratineClasses)


