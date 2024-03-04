'''Given the code below, create a Constructor function as well as a print() function that prints
the name, ID, GPA of students'''

class Student:
    def __init__(self,name,id,gpa):
        self.name=name
        self.id=id
        self.gpa=gpa
    def print(self):
        print(self.name,self.id,self.gpa)            


s1=Student('Johnny',10,3.39)
s2=Student('Catherine',20,3.82)

s1.print()
s2.print()