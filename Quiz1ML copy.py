'''Exercise 1
Given the following dataset:
x = [2.0, 2.9, 1.5, 3.1, 2.4, 3.2, 2.8, 1.9, 3.0, 2.2]
y = [5.3, 8.9, 8.8, 4.4, 9.7, 8.6, 5.2, 9.8, 5.7, 5.4]
and a data point with coordinates: Cx = 2.7, Cy = 4.7. Find the smallest Euclidean distance
between the data point and the data set. Optional: Print the index of the element that has the
smallest Euclidean distance to the data point
Note 1: Your algorithm should be able to work for any data set irrespective of its length
Note 2: The Euclidean distance formula in 2D is : ED =
q
(Cx −xi)2 + (Cy −yi)2
Note 3: Do not use any built-in functions for min/max, ED, sorting, etc. Use only basic functions
such as len() and range()'''
x = [2.0, 2.9, 1.5, 3.1, 2.4, 3.2, 2.8, 1.9, 3.0, 2.2]
y = [5.3, 8.9, 8.8, 4.4, 9.7, 8.6, 5.2, 9.8, 5.7, 5.4]
Cx = 2.7
Cy = 4.7
min=(((Cx-x[0])**2)+((Cy-y[0])**2))**(1/2)
pos=0
for i in range(len(x)):
    if min>(((Cx-x[i])**2)+((Cy-y[i])**2))**(1/2):
        min=(((Cx-x[i])**2)+((Cy-y[i])**2))**(1/2)
        pos=i
print(min)
print(f'position of closest data point is {pos}')