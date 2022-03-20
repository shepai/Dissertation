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
    def getAcc(self):
        return self.mpu.acceleration
    def getGyro(self):
        return self.mpu.gyro
    def getTemp(self):
        return self.mpu.temperature

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
    def isStuck(self):
        if (self.movingForward or self.movingBackward) and True:
            pass
    def getData(self):
        print("data:")
        dat=self.gyro.getAcc()
        print("acc",dat[0]+self.xa0,dat[1]+self.ya0,dat[2]+self.za0)
        dat=self.gyro.getGyro()
        print("gyro",dat[0]+self.xg0,dat[1]+self.yg0,dat[2]+self.zg0)
        print("temp",self.gyro.getTemp())

bot=wheg()
time.sleep(1)
bot.getData()
bot.forward()
bot.getData()
time.sleep(2)
bot.getData()
bot.backward()
bot.getData()
time.sleep(2)
bot.stop()
bot.getData()