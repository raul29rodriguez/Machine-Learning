import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from sklearn.decomposition import PCA

df= pd.read_csv("Skin_NonSkin.csv")
X=np.array(df.iloc[::100,:3])
y=np.array(df.iloc[::100,3])
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(X)
principalDf = pd.DataFrame(data = principalComponents, columns = ['pc1', 'pc2'])
pc1=np.array(principalDf['pc1'])
pc2=np.array(principalDf['pc2'])
X_train, X_test, y_train, y_test = train_test_split(principalComponents, y,test_size=0.2)
modelSVC = SVC(kernel='linear').fit(X_train, y_train)
y_pred= modelSVC.predict(X_test)
print(f'Accuracy score {accuracy_score(y_test,y_pred)}')
print(classification_report(y_test, y_pred))
confusionMatrix = confusion_matrix(y_test, y_pred)
print(confusionMatrix)
sns.heatmap(confusion_matrix(y_test, y_pred))
plt.show()
w = modelSVC.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(-200, 200)
yy = a *xx - (modelSVC.intercept_[0] / w[1])
b = modelSVC.support_vectors_[0]
b2 = modelSVC.support_vectors_[1]
yy_down = a * xx + (b[1] - a * b[0])
b = modelSVC.support_vectors_[-1]
yy_up = a * xx + (b[1] - a * b[0])
for i in range(len(pc1)):
    if y[i]==1:
        plt.scatter(pc1[i],pc2[i],color='purple')
    else:
        plt.scatter(pc1[i],pc2[i],color='yellow')
plt.plot(xx, yy)
plt.plot(xx, yy_down, 'r--')
plt.plot(xx, yy_up, 'g--')
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Support Vector Machine")
plt.show()