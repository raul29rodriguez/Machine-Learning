'''Exercise 1
You have recently been hired by a major enterprise to filter out spam emails. Your man-
ager has given given you access to the following three Normal (i.e., not Spam) training emails:
train_N_I, train_N_II, train_N_III and to the following three Spam training emails: train_S_I,
train_S_II, train_S_III. Classify the following two emails: testEmail_I.txt, testEmail_II.txt as to
whether they are Normal or Spam (use Naïve Bayes classifier). Using two plots side by side,
plot the frequency of words for Normal and Spam emails
Note 1: The Prior Probability for Normal emails is: 0.73 while for Spam is: 0.27
Note 2: To open a file and place its words into a list:
with open("train_N_I.txt", "r") as f:
train_N_I = f.read().split()
Note 3: You may wish to use:
from collections import Counter
countsN = Counter(train_N_I)
to count the number of words in a file
Note 4: You may wish to use
key_listN = list(countsN.keys())
val_listN = list(countsN.values())
to get the keys/values from a dictionary and convert them to a list
Note 5: You may wish to merge the Normal emails into one file and the Spam emails into
another one'''
from collections import Counter
from matplotlib import pyplot as plt

with open("train_N_I.txt", "r") as f:
    train_N_I = f.read().split()
with open("train_N_II.txt", "r") as f:
    train_N_II = f.read().split()
with open("train_N_III.txt", "r") as f:
    train_N_III = f.read().split()

with open("train_S_I.txt", "r") as f:
    train_S_I = f.read().split()
with open("train_S_II.txt", "r") as f:
    train_S_II = f.read().split()
with open("train_S_III.txt", "r") as f:
    train_S_III = f.read().split()

with open("testEmail_I.txt", "r") as f:
    testEmail_I = f.read().split()
with open("testEmail_II.txt", "r") as f:
    testEmail_II = f.read().split()

normalTrain=train_N_I+train_N_II+train_N_III
spamTrain=train_S_I+train_S_II+train_S_III
countsN = Counter(normalTrain)
countsS = Counter(spamTrain)
key_listN = list(countsN.keys())
val_listN = list(countsN.values())
wordCountN=0
for i in range(len(val_listN)):
    wordCountN+=val_listN[i]

key_listS = list(countsS.keys())
val_listS = list(countsS.values())
wordCountS=0
for i in range(len(val_listS)):
    wordCountS+=val_listS[i]

countsT1 = Counter(testEmail_I)
countsT2 = Counter(testEmail_II)
print(countsT1)
print(countsT2)
key_listT1 = list(countsT1.keys())
key_listT2 = list(countsT2.keys())
pST1=.73
pNT1=.27
pST2=.73
pNT2=.27
for i in range(len(key_listT1)):
    n1=countsN.get(key_listT1[i])/wordCountN
    s1=countsS.get(key_listT1[i])/wordCountS
    pNT1*=n1
    pST1*=s1
for i in range(len(key_listT2)):
    n2=countsN.get(key_listT2[i])/wordCountN
    s2=countsS.get(key_listT2[i])/wordCountS
    pNT2*=n2
    pST2*=s2

if(pNT1>pST1):
    print(f'Email 1 is normal')
else:
    print(f'Email 1 is spam')
if(pNT2>pST2):
    print(f'Email 2 is normal')
else:
    print(f'Email 2 is spam')

fig, ax=plt.subplots(nrows=1,ncols=2)
ax[0].bar(key_listN,val_listN)
ax[1].bar(key_listS,val_listS)
ax[0].set_title('Normal')
ax[1].set_title('Spam')
plt.show()