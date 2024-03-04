'''Exercise 1
Given the following data set: avgHigh_jan_1895-2018.csv, perform Linear Regression using
the first two columns of the data set. Predict temperatures for the following three dates: Jan 2019,
Jan 2023, Jan 2024. You can use any built-in functions you wish. Your output should look like the
Figure below
Note: Ignore anomaly column. It is the difference between the temperature for the given date
and average temperatures for all dates'''
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats

df=pd.read_csv('avgHigh_jan_1895-2018.csv')
#print(df)

df=df.drop(['Anomaly'],axis=1)
#print(df)

x=np.array(df['Date'])
y=np.array(df['Value'])

slope,intercept,r,p,std_err=stats.linregress(x,y)
#print('Slope: ',slope,'y-intercept: ',intercept,'Correlation(r): ',r,'p-value: ',p,'Standard Error: ',std_err)
mymodel=(slope*x)+intercept
p=np.array([201901,202301,202401])
prediction=(slope*p)+intercept
print(f'predicted temps {prediction}')

plt.scatter(x,y,c='b',label='Data Points')
plt.scatter(p,prediction,c='g',label='Predicted')
plt.plot(x,mymodel,c='r',label='Model')
plt.xlabel('Years')
plt.ylabel('Temperatures')
plt.title(f'Janurary avg high temp. slope: {round(slope,3)} intercept: {round(intercept,2)}')
plt.legend(loc='lower right')
plt.show()
