from imu import MPU6050
import time
from machine import Pin, I2C

class gyro:
    def __init__(self):  
        self.i2c = I2C(0, sda=Pin(0), scl=Pin(1), freq=400000)
        self.imu = MPU6050(self.i2c)
        self.a=0
        self.getZero()
    def getZero(self): #create a zerod value to detect change
        self.a=0
        for i in range(200):
            self.a+=int(self.imu.gyro.x*10)
        self.a=int(self.a/200)
    def movementDetected(self):
        val=abs(int(self.imu.gyro.x*10)-self.a) #get thresholded value
        data=""
        if val>4: #chec significance
            data = "movement"
        self.getZero() #reevaluate movement threshold
        print(str(val)+data)
        return data

g=gyro()
while True:
    #print(imu.accel.xyz,imu.gyro.xyz,imu.temperature,end='\r')
    time.sleep(0.5)
    g.movementDetected()
    