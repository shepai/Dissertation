import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

capA = cv.VideoCapture(0)
capB = cv.VideoCapture(1)

not_broken=True
while not_broken: #loop through
    retA, imgOriginalScene = capA.read()
    retB, imgOriginalScene = capB.read()
    #stereo creation
    stereo = cv.StereoBM_create(numDisparities=16, blockSize=15)
    disparity = stereo.compute(retA,retB)
    plt.imshow(disparity,'gray')
    plt.show()