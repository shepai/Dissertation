from adafruit_servokit import ServoKit
#import cv2
import numpy as np


kit=ServoKit(channels=16,address=0x44)
class whegbot:
    def __init__(self):
        self.movingForward=False
        self.movingBackward=False

    def rightTurn(self):
        #kit.continuous_servo[0].throttle = 1
        #kit.continuous_servo[1].throttle = 1
        #kit.continuous_servo[2].throttle = 1
        #kit.continuous_servo[3].throttle = 1
        try:
            if kit.servo[5].angle+10<=180:
                kit.servo[5].angle+=10
        except:
            pass
        
    def leftTurn(self):
        #kit.continuous_servo[0].throttle = -1
        #kit.continuous_servo[1].throttle = -1
        #kit.continuous_servo[2].throttle = -1
        #kit.continuous_servo[3].throttle = -1
        print(kit.servo[5].angle)
        try:
            if kit.servo[5].angle-10>=0:
                kit.servo[5].angle-=10
        except:
            pass

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
        print("Stop")
        self.movingForward=False
        self.movingBackward=False
        kit.continuous_servo[0].throttle = 0
        kit.continuous_servo[1].throttle = 0
        kit.continuous_servo[2].throttle = 0
        kit.continuous_servo[3].throttle = 0

    def rotateUp(self):
        try:
            if kit.servo[4].angle+10<=180:
                kit.servo[4].angle+=10
        except:
            pass
    def rotateDown(self):
        try:
            if kit.servo[4].angle-10>=0:
                kit.servo[4].angle-=10
        except:
            pass
"""
import time
moving=False

cap=cv2.VideoCapture(-1, cv2.CAP_V4L)

filename = 'video.avi'
frames_per_second = 24.0
res = '720p'
def change_res(cap, width, height):
    cap.set(3, width)
    cap.set(4, height)

# Standard Video Dimensions Sizes
STD_DIMENSIONS =  {
    "480p": (640, 480),
    "720p": (1280, 720),
    "1080p": (1920, 1080),
    "4k": (3840, 2160),
}


# grab resolution dimensions and set video capture to it.
def get_dims(cap, res='1080p'):
    width, height = STD_DIMENSIONS["480p"]
    if res in STD_DIMENSIONS:
        width,height = STD_DIMENSIONS[res]
    ## change the current caputre device
    ## to the resulting resolution
    change_res(cap, width, height)
    return width, height

# Video Encoding, might require additional installs
# Types of Codes: http://www.fourcc.org/codecs.php
VIDEO_TYPE = {
    'avi': cv2.VideoWriter_fourcc(*'XVID'),
    #'mp4': cv2.VideoWriter_fourcc(*'H264'),
    'mp4': cv2.VideoWriter_fourcc(*'XVID'),
}

def get_video_type(filename):
    filename, ext = os.path.splitext(filename)
    if ext in VIDEO_TYPE:
      return  VIDEO_TYPE[ext]
    return VIDEO_TYPE['avi']
out = cv2.VideoWriter(filename, get_video_type(filename), 25, get_dims(cap, res))


kit.servo[4].angle=90
"""
"""
while 1:
     try:
        events = get_gamepad()
     except:
        stop()
        raise OSError("...")
     _,frame=cap.read()
     out.write(frame)
     for event in events:
         #left stick
         #print(event.code)
         _,frame=cap.read()
         out.write(frame)
         if event.code=="BTN_TRIGGER_HAPPY1":
             if moving:
                 moving=False
                 stop()
             else:
                moving=True
                rightTurn()
         elif event.code=="BTN_TRIGGER_HAPPY2":
             if moving:
                 moving=False
                 stop()
             else:
                moving=True
                leftTurn() 
         elif event.code=="ABS_RZ":
             if event.state==0:
                 stop() 
                 movingForward=False
             else:
                 forward()
                 movingForward=True
         elif event.code=="ABS_Z":
             if event.state==0:
                 stop()
                 movingBackward=False
             else:
                 backward()
                 movingBackward=True
         elif event.code=="BTN_TRIGGER_HAPPY3":
             rotateDown()
         elif event.code=="BTN_TRIGGER_HAPPY4":
             rotateUp()
         #right stick
         elif event.code == "ABS_RX":
             print("right stick forward/backward", event.state)
         elif event.code == "ABS_RY":
             print("right stick turn", event.state)
         elif event.code!="ABS_X" and event.code!="ABS_RZ" and event.code!="ABS_Z":
             pass
     if movingForward:
        forward()
     elif movingBackward:
        backward()
        
stop()
"""
