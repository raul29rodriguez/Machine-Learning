import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
df= pd.read_csv("usedcars.csv")
y=np.array(df['price'])
le = preprocessing.LabelEncoder()
year=df['year']
model=df['model']
mileage=df['mileage']
color=df['color']
transmission=df['transmission']
model_encoded=np.array(le.fit_transform(model))
color_encoded=np.array(le.fit_transform(color))
transmission_encoded=np.array(le.fit_transform(transmission))
df2 = pd.DataFrame({'year' : year, 'model' : model_encoded, 'mileage' :mileage, 'color' :color_encoded, 'transmission' : transmission_encoded})
X=np.array(df2[['year','model','mileage','color','transmission']])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.1)
li=[]
for i in range(10):
    model = RandomForestRegressor(n_estimators=100*i+1).fit(X_train, y_train)
    predictions = model.predict(X_test)
    RMSE=(sum((y_test-predictions)**2)/len(X_test))**.5
    li.append(RMSE)
index=np.argmin(li)
model = RandomForestRegressor(n_estimators=100*index+1).fit(X_train, y_train)
for i in range(len(predictions)):
    print(f'Actual: {y_test[i]}\tPredicted: {predictions[i]}')
p=np.array([2017,0,11307,1,0])
p=p.reshape(1,-1)
print(f'predction for values p: {model.predict(p)}')
print(f'Feature Importance: [year,model,mileage,color,transmission]{model.feature_importances_}')