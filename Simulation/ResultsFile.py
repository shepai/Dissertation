import numpy as np
import matplotlib.pyplot as plt

a=None
b=None
c=None
d=None
e=None
path="D:/Documents/Computer Science/Year 3/Dissertation/"
with open(path+'microbial.npy', 'rb') as f:
    a = np.load(f)
with open(path+'group.npy', 'rb') as f:
    b = np.load(f)
with open(path+'elite.npy', 'rb') as f:
    c = np.load(f)
    c=c[0:len(a)]
with open(path+'microbialConv.npy', 'rb') as f:
    d = np.load(f)
with open(path+'microbialConv2D.npy', 'rb') as f:
    e = np.load(f)
print(a.shape,b.shape,c.shape)

plt.plot([i for i in range(len(a))],a,c="b",label="Microbial") #show fintesses over generations
plt.plot([i for i in range(len(b))],b,c="r",label="Group") #show fintesses over generations
plt.plot([i for i in range(len(c))],c,c="g",label="Elitist") #show fintesses over generations
plt.plot([i for i in range(len(d))],d,c="y",label="Convolutional 1D") #show fintesses over generations
plt.plot([i for i in range(len(e))],e,c="k",label="Convolutional 2D") #show fintesses over generations

plt.title("Results of population fitness over "+str(len(a))+" generations")
plt.ylabel("Fitness Units")
plt.xlabel("Generation")
plt.legend(loc="lower right")
plt.show()

"""
Show results of alternative mutation function

"""

with open(path+'microbialPart.npy', 'rb') as f:
    aa = np.load(f)
with open(path+'groupPart.npy', 'rb') as f:
    bb = np.load(f)

plt.plot([i for i in range(len(aa))],aa,c="b",label="Microbial") #show fintesses over generations
plt.plot([i for i in range(len(bb))],bb,c="r",label="Group") #show fintesses over generations
plt.plot([i for i in range(len(a))],a,"--",c="b",label="Microbial old") #show fintesses over generations
plt.plot([i for i in range(len(b))],b,"--",c="r",label="Group old") #show fintesses over generations

plt.title("Results of population fitness over "+str(len(a))+" generations with alternative mutation")
plt.ylabel("Fitness Units")
plt.xlabel("Generation")
plt.legend(loc="upper left")
plt.show()