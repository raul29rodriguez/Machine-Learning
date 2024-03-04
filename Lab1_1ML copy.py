'''Exercise 1 Similar to the built-in methods arange() and reshape(), shown below,
define and invoke using chain function calls, the myArange() and myReshape() methods in
one line of code as shown in the figure below:
Note: Do not call the built-in methods arange() and reshape()'''
import numpy as np

class myClass:
    def myArrange(self,start,stop):
        l = [x for x in range(start,stop)]
        self.l= l
        print(self.l)
        return self

    def myReshape(self,rows,cols):
        i=0
        self.arr=np.ones((rows,cols),dtype=int)
        for r in range(rows):
            for c in range(cols):
                self.arr[r][c] = self.l[i]
                i=i+1
        return self


obj=myClass()
rows=int(input("Enter number of rows: "))
cols=int(input("Enter number of columns: "))
arrMain=obj.myArrange(1,16).myReshape(rows,cols)
print(f'\n{arrMain.arr}')