import numpy as np
import matplotlib.pyplot as plt

a=None
b=None
c=None
d=None
e=None
g=None
path="D:/Documents/Computer Science/Year 3/Dissertation/SavedModelsAndResults/"

#microbial
with open(path+'microbial.npy', 'rb') as f:
    a1 = np.load(f)
with open(path+'microbialPart1.npy', 'rb') as f:
    a2 = np.load(f)
with open(path+'microbialPart2.npy', 'rb') as f:
    a3 = np.load(f)

#group
with open(path+'group.npy', 'rb') as f:
    b1 = np.load(f)
with open(path+'group1.npy', 'rb') as f:
    b2 = np.load(f)
with open(path+'group2.npy', 'rb') as f:
    b3 = np.load(f)
#elite
with open(path+'elite.npy', 'rb') as f:
    c = np.load(f)
    c1=c[len(a1):len(a1)*2]
with open(path+'elite1.npy', 'rb') as f:
    c = np.load(f)
    c2=c[len(a1):len(a1)*2]
with open(path+'elite2.npy', 'rb') as f:
    c = np.load(f)
    c3=c[len(a1):len(a1)*2]
#
with open(path+'microbialConv.npy', 'rb') as f:
    d = np.load(f)
#2D
with open(path+'microbialConv2D.npy', 'rb') as f:
    e1 = np.load(f)
with open(path+'microbialConv2D1.npy', 'rb') as f:
    e2 = np.load(f)
with open(path+'microbialConv2D2.npy', 'rb') as f:
    e3 = np.load(f)
#2dG
with open(path+'group2D.npy', 'rb') as f:
    g1 = np.load(f)
with open(path+'group2D1.npy', 'rb') as f:
    g2 = np.load(f)
with open(path+'group2D2.npy', 'rb') as f:
    g3 = np.load(f)
print(a1.shape,b1.shape,c1.shape)

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
""""""
mult=np.array([a1,a2,a3])
yerr = makeYerr(mult)/2

plt.errorbar([i for i in range(len(a1))], a1,  yerr=yerr,c="b",label="Microbial",errorevery=50)
#group
mult=np.array([b1,b2,b3])
yerr = makeYerr(mult)/2

plt.errorbar([i for i in range(len(b1))], b1,  yerr=yerr,c="r",label="Group",errorevery=55)
#elitist
mult=np.array([c1,c2,c3])
yerr = makeYerr(mult)/2

plt.errorbar([i for i in range(len(c1))], c1,  yerr=yerr,c="g",label="Elitist",errorevery=45)
#2d
mult=np.array([e1,e2,e3])
yerr = makeYerr(mult)/2

plt.errorbar([i for i in range(len(e1))], e1,  yerr=yerr,c="k",label="Convolutional 2D",errorevery=40)
#2dG
mult=np.array([g1,g2,g3])
yerr = makeYerr(mult)/2

plt.errorbar([i for i in range(len(g1))], g1,  yerr=yerr,c="m",label="Convolutional 2D Group",errorevery=50)
#plt.plot([i for i in range(len(b))],b,c="r",label="Group") #show fintesses over generations
#plt.plot([i for i in range(len(c))],c,c="g",label="Elitist") #show fintesses over generations
#plt.plot([i for i in range(len(d))],d,c="y",label="Convolutional 1D") #show fintesses over generations
#plt.plot([i for i in range(len(e))],e,c="k",label="Convolutional 2D") #show fintesses over generations
#plt.plot([i for i in range(len(g))],e,c="m",label="Convolutional 2D Group") #show fintesses over generations

plt.title("Results of population fitness over "+str(len(a1))+" generations")

plt.ylabel("Fitness Units")
plt.xlabel("Generation")
plt.legend(loc="lower right")
plt.show()
exit()

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