'''Exercise 1
Using an infinite loop, enter your homework grades (enter at least 10 grades) of float data type
and append them into a grades list. Break the loop when the user enters a grade smaller than 0.
Create a NumPy array out of the grades list; create a Panda Series out of the NumPy array and
rename the indices to begin from 1 instead of 0 (since you know the length of the list you can
create a new list using list comprehension that begins from 1). Using a built-in method, print
the descriptive statistics of the grades entered (e.g., mean, std, max, min, 25% percentile, etc.).
Create three plots within a single graph, namely a plot, a scatter, and a bar superimposed one
over the other; the x-axis is the indices beginning with 1 and the y-axis is the grades entered (see
first Figure in the next page)
Note: Do not hard code the name of indices beginning with 1 as you do not know in advance
how many grades the user will enter, that is why you are advised to use a list comprehension'''
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
print(gradesSeries.describe())
x=gradesSeries.index.values
y=gradesSeries.values
plt.plot(x,y)
plt.scatter(x,y)
plt.bar(x,y)
plt.show()