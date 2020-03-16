#Logistic Regression
import numpy as np
import pandas as pd
#Read Dataset from the main memory using csv file
dataset1=pd.read_csv('C:\\Users\\arpit\\Downloads\\Logistic_Regression\\Social_Network_Ads.csv')
print(dataset1)
#Split the dataset into independent and dependent variable
x=dataset1.iloc[:,2:4].values
print(x)
y=dataset1.iloc[:,4:].values
print(y)
#Do Feature Scaling on independent variable
from sklearn.preprocessing import StandardScaler
standard_scaler=StandardScaler()
x=standard_scaler.fit_transform(x)
print(x)
#Split data into training and test set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)
#Model_Prediction using logistic Regression
from sklearn.linear_model import LogisticRegression
regressor1=LogisticRegression()
regressor1.fit(x_train,y_train)
y_pred=regressor1.predict(x_test)