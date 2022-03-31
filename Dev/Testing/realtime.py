import matplotlib.pyplot as plt
import numpy as np


array = np.random.normal(0, 0.5, (100))

while True:
    plt.cla()
    plt.plot(array)
    array+=np.random.normal(0, 0.5, (100))
    array[array<-4]=-4
    array[array>4]=4
    plt.pause(0.1)
    plt.ylim(ymax = -4, ymin = 4)
    plt.show(block=False)