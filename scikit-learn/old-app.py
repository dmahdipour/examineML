import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics  import f1_score,accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier

data=pd.read_excel(open('../Datasets/data.xlsx', 'rb'), sheet_name='Export')  
y = data.Creatinine
for i in range(len(y)):
    if y[i]>1.2:
        y[i]=1
    else:
        y[i]=0
data.drop(['Creatinine'], axis=1, inplace=True)
x=data
print(x, y)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=2)

DT= SGDClassifier()
DT.fit(X_train,y_train)

pred=DT.predict(X_test)


print('Accuracy: ', accuracy_score(y_test,pred))



