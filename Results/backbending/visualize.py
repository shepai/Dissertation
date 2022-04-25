import matplotlib.pyplot as plt
fileN="D:/Documents/Computer Science/Year 3/Dissertation/Results/backbending/history1.txt"
fileN2="D:/Documents/Computer Science/Year 3/Dissertation/Results/backbending/history2.txt"
def getData(filename):
    file=open(filename,"r")
    r=file.read()
    file.close()

    r=r.replace("[","")
    r=r.replace("]","")
    r=r.split(",")
    a=[0]
    for val in r:
        a.append((float(val)/0.4)*100) if (float(val)/0.4)*100>=max(a) else a.append(max(a))
    return a
a=getData(fileN)
a1=getData(fileN2)

plt.title("Results to show the changes in accuracy over time with different trials")
plt.plot([x+1 for x in range(len(a))],a)
plt.plot([x+1 for x in range(len(a1))],a1)
plt.xlabel("Generation")
plt.ylabel("Accuracy %")
plt.show()