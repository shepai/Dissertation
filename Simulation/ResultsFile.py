import numpy as np
import matplotlib.pyplot as plt

a=None
b=None
c=None
path="D:/Documents/Computer Science/Year 3/Dissertation/"
with open(path+'microbial.npy', 'rb') as f:
    a = np.load(f)
with open(path+'group.npy', 'rb') as f:
    b = np.load(f)
    
with open(path+'elite.npy', 'rb') as f:
    c = np.load(f)
    c=c[0:500]
    
print(a.shape,b.shape,c.shape)

plt.plot([i for i in range(len(a))],a,c="b",label="Microbial") #show fintesses over generations
plt.plot([i for i in range(len(b))],b,c="r",label="Group") #show fintesses over generations
plt.plot([i for i in range(len(c))],c,c="g",label="Elitist") #show fintesses over generations

plt.title("Results of population fitness over "+str(len(a))+" generations")
plt.ylabel("Fitness Units")
plt.xlabel("Generation")
plt.legend(loc="upper left")
plt.show()