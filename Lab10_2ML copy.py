import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
names = ['Sample code number', 'Clump Thickness','Uniformity of Cell Size','Uniformity of Cell Shape','Marginal Adhesion',
'Single Epithelial Cell Size','Bare Nuclei','Bland Chromatin','Normal Nucleoli','Mitoses','Class' ]
df = pd.read_csv('breast-cancer-wisconsin.data.csv', names=names)
df=df.replace('?',np.nan)
df=df.dropna()
df=df.drop("Sample code number",axis=1)
X = np.array(df.loc[:, 'Clump Thickness' : 'Mitoses'])
y = np.array(df['Class'])
X = StandardScaler().fit_transform(X)
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(X)
pc1=principalComponents[0:,1]
pc2=principalComponents[0:,0]
X_train, X_test, y_train, y_test = train_test_split(principalComponents, y,test_size=0.1, random_state=0)
modelSVC = SVC(kernel='linear').fit(X_train, y_train)
y_pred = modelSVC.predict(X_test)
confusionMatrix = confusion_matrix(y_test, y_pred)
print(confusionMatrix)
print(f'Accuracy score: {accuracy_score(y_test, y_pred)}')
for i in range(len(y)):
    if y[i]==2:
        plt.scatter(pc1[i],pc2[i],color='r')
    else:
        plt.scatter(pc1[i],pc2[i],color='g')

plt.xlabel("pc1")
plt.ylabel("pc2")
plt.show()