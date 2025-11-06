import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

#Define names of columns in dataset
names = ['Sample code number', 'Clump Thickness','Uniformity of Cell Size','Uniformity of Cell Shape','Marginal Adhesion',
'Single Epithelial Cell Size','Bare Nuclei','Bland Chromatin','Normal Nucleoli','Mitoses','Class' ]

#Read data from CSV file into pandas DataFrame
df = pd.read_csv('breast-cancer-wisconsin.data.csv', names=names)

#replace missing values which are represented as '?' with NaN (not a number)
df=df.replace('?',np.nan)

#remove rows with NaN values
df=df.dropna()

#drop column which is not needed for our model
df=df.drop("Sample code number",axis=1)

#Extract features (independent variables) from the DataFrame using list slicing
X = np.array(df.loc[:, 'Clump Thickness' : 'Mitoses'])

#Extract target variable (dependent variable) from DataFrame
y = np.array(df['Class'])

#Standardize features by scaling
#Used so PCA and SVM perform well
X = StandardScaler().fit_transform(X)

#apply PCA to reduce dimensionality of the data, in this case we reduced to 2 principal components
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(X)

#assign principal components to pc1, pc2
pc1=principalComponents[0:,1]
pc2=principalComponents[0:,0]

#split data into training and testing sets
#we use a 10% test size and 90% training size from our data
#random_state=0 ensures the split is reproducible
X_train, X_test, y_train, y_test = train_test_split(principalComponents, y,test_size=0.1, random_state=0)

#create an SVM model with a linear kernel and fit it to the training data
modelSVC = SVC(kernel='linear').fit(X_train, y_train)

#use the trained model to make predictions on the test set
y_pred = modelSVC.predict(X_test)

#Generate confusion matrix to evaluate performance of the model
confusionMatrix = confusion_matrix(y_test, y_pred)
print(confusionMatrix)

#Calculate and print accuracy score of the model
print(f'Accuracy score: {accuracy_score(y_test, y_pred)}')

#create a scatter plot of the data points, colored by class
#red for class 2 and green for the other classes
for i in range(len(y)):
    if y[i]==2:
        plt.scatter(pc1[i],pc2[i],color='r')
    else:
        plt.scatter(pc1[i],pc2[i],color='g')

#set labels for x and y axis
plt.xlabel("pc1")
plt.ylabel("pc2")

#display plot
plt.show()
