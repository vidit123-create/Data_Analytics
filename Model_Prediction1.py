import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset=pd.read_csv('C:\\Users\\arpit\\Desktop\\medExpenses.csv')
print(dataset)
x=dataset.iloc[:,1:2].values
print(x)
y=dataset.iloc[:,2:].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)
#Graphical Plotting
plt.scatter(x_train,y_train,color="Red")
plt.scatter(x_test,y_test,color="Green")
plt.plot(x_test,regressor.predict(x_test),color="Blue")
plt.xlabel('family_size')
plt.ylabel('medExpenses')
plt.show