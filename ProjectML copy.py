#spam SMS detection 
import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

df = pd.read_csv('spam.csv')
X=np.array(df['Message'])
y=np.array(df['Category'])
#print(X)
#print(y)
for i in range(len(y)):
    if y[i]=='ham':
        y[i]='not spam'
#print(y) 

#splittting data and using coutVectorizer to get word count per training message
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2)
#print(X_train)
cv=CountVectorizer() 
X_train_WordCount=cv.fit_transform(X_train)
#print(X_train_WordCount.toarray()) #doesn't show full matrix due to its size...
print(f'shape of vector{X_train_WordCount.shape}')

#creating model with MultinomialNB since it works better with discrete values
model=MultinomialNB()
model.fit(X_train_WordCount,y_train)

#testing model with our own messages
pSpam1=['click here for your money reward']
pSpam1C=cv.transform(pSpam1)
print(f'{pSpam1}\t{model.predict(pSpam1C)}')
pSpam2=['Get your free prize as soon as possible']
pSpam2C=cv.transform(pSpam2)
print(f'{pSpam2}\t{model.predict(pSpam2C)}')
pNotSpam1=['Hi how are you doing?']
pNotSpam1C=cv.transform(pNotSpam1)
print(f'{pNotSpam1}\t{model.predict(pNotSpam1C)}')
pNotSpam2=['Meet me at 8']
pNotSpam2C=cv.transform(pNotSpam2)
print(f'{pNotSpam2}\t{model.predict(pNotSpam2C)}')
userMessage=input("Enter a message to check: ")
userMessage=[userMessage]
userMessage=cv.transform(userMessage)
print(model.predict(userMessage))

#testing rest of dataset and getting accuracy score and confusion matrix
X_test_WordCount=cv.transform(X_test)
y_pred=model.predict(X_test_WordCount)
print(f'Accuracy score {accuracy_score(y_test,y_pred)}')
confusionMatrix = confusion_matrix(y_test, y_pred)
print(confusionMatrix)
print(classification_report(y_test, y_pred))

for i in range(len(y_pred)):
    if y_pred[i] != y_test[i]:
        print(f'{X_test[i]}\tPredicted: {y_pred[i]}\tActual: {y_test[i]}')