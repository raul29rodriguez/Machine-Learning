'''Exercise 2
From the textbook Elementary Linear Algebra by Larson, Edwards, Falvo (6th Ed.), read the
section on Cryptography, pages 102-105 and, using Python, replicate the example shown, i.e.,
encode a message and then decode it. Your algorithm should be able to work for any message
of any length (assume only alphabetic characters and use a whitespace instead of an underscore)
Note 1: Use the same numbers that correspond to characters with the only exception being
the replacement of the underscore with the whitespace
Note 2: You can use the np.matmul() function to perform matrix multiplication
Note 3: You can use the np.linalg.inv() function to calculate the inverse of a matrix'''
import numpy as np
message=input("Enter messsage: ")
message=message.upper()
print(message)
chars=[ord(i)-32 if i==" " else ord(i)-64 for i in message]
while len(chars)%3!=0:
    chars.append(0)
#print(chars)
nums=[]
for i in range(len(chars)):
    if i%3==0:
        nums.append(chars[i:i+3])
nums=np.array(nums)
#print(nums)
A=np.array([[1,-2,2],
    [-1,1,3],
    [1,-1,-4]])
encoded=np.matmul(nums,A)
print(f'encoded message {encoded.flatten()}')
Ainv=np.linalg.inv(A)
Ainv=Ainv.astype(int)
#print(Ainv)
decoded=np.matmul(encoded,Ainv)
#print(decoded)
decoded=decoded.flatten()
message2=[chr(i+32) if i==0 else chr(i+64) for i in decoded]
print(f'decoded message {"".join(message2)}')