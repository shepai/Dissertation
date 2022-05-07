
import numpy as np
import matplotlib.pyplot as plt

path="D:/Documents/Computer Science/Year 3/Dissertation/SavedModelsAndResults/"
a=None
with open(path+'stored.npy', 'rb') as f:
    a = np.load(f)

def makeYerr(malt,interval=1):
    stds=[]
    for i in range(len(malt[0])):
        a=[]
        if i%interval==0:
            for j in range(len(malt)):
                a.append(malt[j][i])
            a=np.array(a)
            stds.append(np.std(a))
    return np.array(stds)

archs=[[10,10],[20,20],[30,30],[40,40],[50,50],[60,60]]

averages=[]
stds=[]
print(a.shape)
for i in range(len(a)):
    z1=a[i][0]
    z2=a[i][1]
    z3=a[i][2]
    averages.append((z1+z2+z3)/3)
    stds.append(makeYerr((z1,z2,z3)))

plt.title("Results of population fitness with different architectures")
for i in range(len(archs)):
    plt.errorbar([i for i in range(len(averages[i]))], averages[i], yerr=stds[i],label=str(archs[i][0])+" "+str(archs[i][1])+" Network",errorevery=40+(i*5))

plt.ylabel("Fitness Units")
plt.xlabel("Generation")
plt.legend(loc="lower right")
plt.show()
"""
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from tqdm import notebook
import time
from torchviz import make_dot

from agent import *
vectors=[(1,1),(1,0),(0,1),(-1,-1),(-1,0),(0,-1),(-1,1),(1,-1)] #possible moves
whegBot=Agent_Conv2D((5*5)+2,[40,40],len(vectors)) #define the agent
bestGene=np.load("D:/Documents/Computer Science/Year 3/Dissertation/best.npy") #load file
whegBot.set_genes(bestGene)
x = torch.zeros(5, 5, dtype=torch.float, requires_grad=False)
print(x)
out = whegBot.forward(x,np.array([1,1]))
disp=make_dot(out)#.render("attached.png", format="png"
disp.render(directory='doctest-output')
"""