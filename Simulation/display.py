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
