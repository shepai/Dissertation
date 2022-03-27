from adafruit_servokit import ServoKit
import board
import adafruit_mpu6050
import time

kit=ServoKit(channels=16)
i2c = board.I2C()  # uses board.SCL and board.SDA
mpu = adafruit_mpu6050.MPU6050(i2c)
class Gyro:
    def __init__(self,mpu=mpu):
        self.mpu=mpu
        self.a=0
        self.getZero()
    def getAcc(self):
        return self.mpu.acceleration
    def getGyro(self):
        return self.mpu.gyro
    def getTemp(self):
        return self.mpu.temperature
        
    def getZero(self): #create a zerod value to detect change
        self.a=0
        for i in range(200):
            self.a+=int(self.getGyro()[2]*10)
        self.a=int(self.a/200)
    def movementDetected(self,sensitivity=150):
        val=abs(int(self.getGyro()[2]*10)-self.a) #get thresholded value
        data=""
        if val>sensitivity: #chec significance
            data = "movement"
        self.getZero() #reevaluate movement threshold
        print(str(val)+data)
        return data



class wheg:
    def __init__(self,kit=kit):
        self.movingForward=False
        self.movingBackward=False
        self.gyro = Gyro()
        g=self.gyro.getGyro() #get gyroscopic data
        self.xg0=0#g[0]*-1
        self.yg0=0#g[1]*-1
        self.zg0=0#g[2]*-1
        acc=self.gyro.getAcc() #get accelerometer data
        self.xa0=acc[0]*-1
        self.ya0=acc[1]*-1
        self.za0=acc[2]*-1
        self.getData()
    def leftTurn(self):
        if kit.servo[5].angle+10<=180:
            kit.servo[5].angle+=10  
    def rightTurn(self):
        if kit.servo[5].angle-10<=180:
            kit.servo[5].angle-=10
    def forward(self):
        self.movingForward=True
        self.movingBackward=False
        kit.continuous_servo[0].throttle = 1
        kit.continuous_servo[1].throttle = -1
        kit.continuous_servo[2].throttle = 1
        kit.continuous_servo[3].throttle = -1
    def backward(self):
        self.movingForward=False
        self.movingBackward=True
        kit.continuous_servo[0].throttle = -1
        kit.continuous_servo[1].throttle = 1
        kit.continuous_servo[2].throttle = -1
        kit.continuous_servo[3].throttle = 1
    def stop(self):
        self.movingForward=False
        self.movingBackward=False
        print("Stop")
        kit.continuous_servo[0].throttle = 0
        kit.continuous_servo[1].throttle = 0
        kit.continuous_servo[2].throttle = 0
        kit.continuous_servo[3].throttle = 0
    def rotateUp(self):
        if kit.servo[4].angle+10<=180:
            kit.servo[4].angle+=10
    def rotateDown(self):
        if kit.servo[4].angle-10>=0:
            kit.servo[4].angle-=10
    def isStuck(self,gyro):
        if (self.movingForward or self.movingBackward) and gyro.movementDetected()!="":
            return True
        return False
    def getData(self):
        print("data:")
        dat=self.gyro.getAcc()
        print("acc",dat[0]+self.xa0,dat[1]+self.ya0,dat[2]+self.za0)
        dat=self.gyro.getGyro()
        print("gyro",dat[0]+self.xg0,dat[1]+self.yg0,dat[2]+self.zg0)
        print("temp",self.gyro.getTemp())


g=Gyro()
bot=wheg()
bot.stop()
time.sleep(2)
bot.getData()
for i in range(10):
    bot.forward()
    print(g.movementDetected())
bot.getData()
time.sleep(2)
bot.getData()
bot.backward()
bot.getData()
time.sleep(2)
bot.stop()
print(g.movementDetected())
bot.getData()


while True:
    #print(imu.accel.xyz,imu.gyro.xyz,imu.temperature,end='\r')
    time.sleep(0.5)
    g.movementDetected()
    