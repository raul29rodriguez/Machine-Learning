import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

#Load dataset
df= pd.read_csv("usedcars.csv")

#extract target variable price
y=np.array(df['price'])

#initiaize LabelEncoder
le = preprocessing.LabelEncoder()

#extract features from the DataFrame
year=df['year']
model=df['model']
mileage=df['mileage']
color=df['color']
transmission=df['transmission']

#Encode categorical features
model_encoded=np.array(le.fit_transform(model))
color_encoded=np.array(le.fit_transform(color))
transmission_encoded=np.array(le.fit_transform(transmission))

#create new DataFrame with encoded features
df2 = pd.DataFrame({'year' : year, 'model' : model_encoded, 'mileage' :mileage, 'color' :color_encoded, 'transmission' : transmission_encoded})

#convert the features DataFrame into a NumPy array
X=np.array(df2[['year','model','mileage','color','transmission']])

#split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.1)

#initialize a list to store RMSE values
li=[]

#Loop to test different numbers of estimators in the RandomForestRegressor for i in range(10)
for i in range(10):
    #Train RandomForestRegressor model
    model = RandomForestRegressor(n_estimators=100*i+1).fit(X_train, y_train)

    #make predictions on test set
    predictions = model.predict(X_test)

    #calculate RMSE
    RMSE=(sum((y_test-predictions)**2)/len(X_test))**.5

    #append RMSE to our list
    li.append(RMSE)

#find index of the minimum RMSE
index=np.argmin(li)

#train model with optimal number of estimators
model = RandomForestRegressor(n_estimators=100*index+1).fit(X_train, y_train)

#print actual vs predicted values for the test set
for i in range(len(predictions)):
    print(f'Actual: {y_test[i]}\tPredicted: {predictions[i]}')

#prepare sample input for prediction
p=np.array([2017,0,11307,1,0])
p=p.reshape(1,-1)

#print prediction for the sample imput
print(f'predction for values p: {model.predict(p)}')

#print feature importances
print(f'Feature Importance: [year,model,mileage,color,transmission]{model.feature_importances_}')
