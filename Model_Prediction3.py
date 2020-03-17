import pandas as pd
import numpy as np
dataset1=pd.read_csv('C:\\Users\\arpit\\Downloads\\K_Nearest_Neighbors\\Social_Network_Ads.csv')
print(dataset1)
#Split the data into independent and dependent variable
x=dataset1.iloc[:,2:4].values
print(x)
y=dataset1.iloc[:,4:].values
print(y)
#Perform Feature Scaling on the independent variable
from sklearn.preprocessing import StandardScaler
standard_scaler=StandardScaler()
x=standard_scaler.fit_transform(x)
print(x)
#Split the dataset into training and test data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=3)
#Perform Model_Prediction using Support Vector Machine
from sklearn.svm import SVC
sc=SVC(kernel='rbf',random_state=1)
sc.fit(x_train,y_train)
y_pred=sc.predict(x_test)
print(y_pred)