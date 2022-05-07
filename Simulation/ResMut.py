import numpy as np
import matplotlib.pyplot as plt

path="D:/Documents/Computer Science/Year 3/Dissertation/SavedModelsAndResults/"

a1=None
a2=None
a3=None

b1=None
b2=None

#microbial
with open(path+'microbial.npy', 'rb') as f:
    a1 = np.load(f)
with open(path+'microbialPart1.npy', 'rb') as f:
    a2 = np.load(f)
with open(path+'microbialPart2.npy', 'rb') as f:
    a3 = np.load(f)
a=(a1+a2+a3)/3
with open(path+'microbialPartOld.npy', 'rb') as f:
    b1 = np.load(f)
with open(path+'microbialPartOld1.npy', 'rb') as f:
    b2 = np.load(f)
with open(path+'microbialPartOld1.npy', 'rb') as f:
    b3 = np.load(f)

b=(b1+b2+b3)/3
#conv
with open(path+'microbialConv2DOld1.npy', 'rb') as f:
    c1 = np.load(f)
with open(path+'microbialConv2DOld2.npy', 'rb') as f:
    c2 = np.load(f)
with open(path+'microbialConv2DOld3.npy', 'rb') as f:
    c3 = np.load(f)

c=(c1+c2+c3)/3

with open(path+'microbialConv2D.npy', 'rb') as f:
    d1 = np.load(f)
with open(path+'microbialConv2D1.npy', 'rb') as f:
    d2 = np.load(f)
with open(path+'microbialConv2D2.npy', 'rb') as f:
    d3 = np.load(f)
d=(d1+d2+d3)/3

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

#plt.plot([i for i in range(len(a))],a,c="b",label="Microbial") #show fintesses over generations
#microbial
mult=np.array([a1,a2,a3])
yerr = makeYerr(mult)/2

plt.errorbar([i for i in range(len(a1))], a, yerr=yerr,c="b",label="Microbial",errorevery=50)
#microbial
mult=np.array([b1,b2,b3])
yerr = makeYerr(mult)/2

plt.errorbar([i for i in range(len(b1))], b, linestyle="--", yerr=yerr,c="b",label="Microbial Old",errorevery=55)
#2d
mult=np.array([c1,c2,c3])
yerr = makeYerr(mult)/2

plt.errorbar([i for i in range(len(c1))], c,linestyle="--",  yerr=yerr,c="r",label="Conv2D old",errorevery=60)

#2d
mult=np.array([d1,d2,d3])
yerr = makeYerr(mult)/2

plt.errorbar([i for i in range(len(d1))], d, yerr=yerr,c="r",label="Conv2D",errorevery=65)

plt.title("Results of genotypes with different selection and mutation functions")
plt.ylabel("Fitness Units")
plt.xlabel("Generation")
plt.legend(loc="lower right")
plt.show()
