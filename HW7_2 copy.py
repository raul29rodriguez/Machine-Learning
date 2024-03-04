'''Exercise 2
Based on the Ex. 1, and after having created a Panda Series, create 5 lists from the Panda
Series, one for each grade, that is, A to F. For instance, a list that holds B grades taken from the
grades Panda Series would be: B = list(grades[(grades >= 80) & (grades < 90)]). Create a pie chart
where the slices are the number of elements in each one of the lists of A, B, C, D, F. For colors,
use r, g, b, y, m, start at 90◦, use shadow, and explode the F grades in the pie chart (see second
Figure in the next page)'''
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

gradesList=[]
while True:
    grade=float(input("Enter a grade: "))
    if grade<0:
        break
    else:
        gradesList.append(grade)
gradesArray=np.array(gradesList)
gradesSeries=pd.Series(gradesArray,index=[i+1 for i in range(len(gradesArray))])
a=list(gradesSeries[(gradesSeries>=90)])
b=list(gradesSeries[(gradesSeries<90)&(gradesSeries>=80)])
c=list(gradesSeries[(gradesSeries<80)&(gradesSeries>=70)])
d=list(gradesSeries[(gradesSeries<70)&(gradesSeries>=60)])
f=list(gradesSeries[(gradesSeries<60)])
numElements=[len(a),len(b),len(c),len(d),len(f)]
plt.pie(numElements,labels= ['a','b','c','d','f'],colors=['r','g','b','y','m'],startangle=90,shadow=True,explode=(0,0,0,0,0.1),autopct='%1.1f%%')
plt.show()