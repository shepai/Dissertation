import matplotlib.pyplot as plt
import numpy as np


array = np.random.normal(0, 0.5, (100))

while True:
    plt.cla()
    plt.plot(array)
    array+=np.random.normal(0, 0.1, (100))
    array[array<-4]=-4
    array[array>4]=4
    plt.ylim(bottom=-4,top=4)
    plt.title("Gaussian distribution mutation over time")
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.pause(0.1)
    plt.show(block=False)