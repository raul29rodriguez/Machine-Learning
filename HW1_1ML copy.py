'''Exercise 1
Solve the locker riddle which reads as follows:
Imagine 100 lockers numbered 1 to 100 with 100 students lined up in front of those 100
lockers:
The first student opens every locker.
The second student closes every 2nd locker.
The 3rd student changes every 3rd locker; if it’s closed, she opens it; if it’s open, she closes it.
The 4th student changes every fourth locker (e.g., 4th, 8th, etc.).
The 5th student changes every 5th locker (e.g., 5th, 10th, etc.).
That same pattern continues for all 100 students.
Here’s the question: "Which lockers are left open after all 100 students have walked the row
of lockers?"'''
import numpy as np
locker_list = list(range(1,101))
lockers=np.zeros(len(locker_list),dtype=bool)
for i in range(1,len(lockers)+1):
    for j in range(i,len(lockers)+1,i):
        lockers[j-1]=not lockers[j-1] #j-1 since i starts at 1 
#print(lockers)
pos=1
for i in range(1,len(lockers)+1):
    if i==1:
        if lockers[i-1]:
            print(i)
    if i==100:
        break
    if lockers[i]:
        print(i+1)