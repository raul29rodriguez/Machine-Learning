'''Exercise 3
In continuation of Ex. 2, find how much each one of the three coefficients contributes to the
goodness-of-fit of the model (you will have to use R2)'''
import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_csv('auto-mpg.csv')
df=df.replace('?',np.nan)
df=df.dropna()
df=df.drop("origin",axis=1)
df=df.drop("car name",axis=1)
df=df.astype(float)
print(df)
y=np.array(df['mpg'])
x=np.array(df.loc[:,'cylinders':'model year'])

y = np.array(df['mpg'])
x1 = np.array(df['weight'])
x2 = np.array(df['displacement'])
x3 = np.array(df['horsepower'])
slope, intercept, r, p, std_error = stats.linregress(x1, y)
print('Slope: ', slope, 'y-intercept: ', intercept, 'r: ', r)
slope, intercept, r, p, std_error = stats.linregress(x2, y)
print('Slope: ', slope, 'y-intercept: ', intercept, 'r: ', r)
slope, intercept, r, p, std_error = stats.linregress(x3, y)
print('Slope: ', slope, 'y-intercept: ', intercept, 'r: ', r)