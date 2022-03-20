# For windows
from pynput import keyboard
from adafruit_servokit import ServoKit
import cv2
import numpy as np

kit=ServoKit(channels=16)

movingForward=False
movingBackward=False

def leftTurn():
    #kit.continuous_servo[0].throttle = 1
    #kit.continuous_servo[1].throttle = 1
    #kit.continuous_servo[2].throttle = 1
    #kit.continuous_servo[3].throttle = 1
    if kit.servo[5].angle+10<=180:
        kit.servo[5].angle+=10
    
def rightTurn():
    #kit.continuous_servo[0].throttle = -1
    #kit.continuous_servo[1].throttle = -1
    #kit.continuous_servo[2].throttle = -1
    #kit.continuous_servo[3].throttle = -1
    if kit.servo[5].angle-10<=180:
        kit.servo[5].angle-=10

def forward():
    kit.continuous_servo[0].throttle = 1
    kit.continuous_servo[1].throttle = -1
    kit.continuous_servo[2].throttle = 1
    kit.continuous_servo[3].throttle = -1

def backward():
    kit.continuous_servo[0].throttle = -1
    kit.continuous_servo[1].throttle = 1
    kit.continuous_servo[2].throttle = -1
    kit.continuous_servo[3].throttle = 1

def stop():
    print("Stop")
    kit.continuous_servo[0].throttle = 0
    kit.continuous_servo[1].throttle = 0
    kit.continuous_servo[2].throttle = 0
    kit.continuous_servo[3].throttle = 0

def rotateUp():
    if kit.servo[4].angle+10<=180:
        kit.servo[4].angle+=10
def rotateDown():
    if kit.servo[4].angle-10>=0:
        kit.servo[4].angle-=10


def on_press(key):
    try:
        k = key.char  # single-char keys
    except:
        k = key.name  # other keys
    print('Key pressed: ' + k)
    if str(k)=="up":
        forward()
    if str(k)=="down":
        backward()
    if str(k)=="right":
        rightTurn()
    if str(k)=="left":
        leftTurn()
        
listener = keyboard.Listener(on_press=on_press)
listener.start()  # start to listen on a separate thread
listener.join()  # remove if main thread is polling self.keys

