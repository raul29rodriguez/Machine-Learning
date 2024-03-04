'''Exercise 2
Given the following data set: avgHigh_jan_1895-2018.csv, split your data into training and
testing. Perform Linear Regression on the training data set and predict temperature values
using the testing data set. The test size should be given as input by the user. Print the actual
temperatures from the test data set as well as the predicted ones. Compute the Root Mean
Square Error between the actual temperatures and the predicted ones from the test data set.
Your output should look like the Figure below
Note 1: Ignore anomaly column. It is the difference between the temperature for the given
date and average temperatures for all dates
Note 2: You can use any built-in functions you wish apart from functions to split the data set'''
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats
df=pd.read_csv('avgHigh_jan_1895-2018.csv')
df=df.drop(['Anomaly'],axis=1)
p=float(input("Enter test size percentage "))
print(len(df))
n=round(p*len(df))
print(n)
trainDF=df.iloc[:n]
testDF=df.iloc[n:]
xTrain=np.array(trainDF.iloc[:,0])
yTrain=np.array(trainDF.iloc[:,1])
xTest=np.array(testDF.iloc[:,0])
yTest=np.array(testDF.iloc[:,1])
slope,intercept,r,p,std_err=stats.linregress(xTrain,yTrain)
#print('Slope: ',slope,'y-intercept: ',intercept,'Correlation(r): ',r,'p-value: ',p,'Standard Error: ',std_err)
mymodel=(slope*xTrain)+intercept
prediction=(slope*xTest)+intercept
print(f'predicted temps {prediction}')
print(f'actual temp {yTest}')
plt.scatter(xTrain,yTrain,c='b',label='Train')
plt.scatter(xTest,yTest,c='g',label='Test')
plt.plot(xTrain,mymodel,c='r',label='Model')
plt.xlabel('Years')
plt.ylabel('Temperatures')
plt.title(f'Janurary avg high temp. slope: {round(slope,3)} intercept: {round(intercept,2)}')
plt.legend(loc='lower right')
plt.show()

