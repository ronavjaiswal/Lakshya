#
# Project LAKSHYA
# Author: Ronav Jaiswal
# Copyright 2023. All Rights Reserved.
#


import argparse
import sys
import time

import cv2
import numpy as np
import utils
import pygame

from datetime import date
from time import sleep

def left():
    pygame.mixer.music.load("/home/pi/Downloads/left.mp3")
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy() == True:
        continue

def left2():
    if pygame.mixer.music.get_busy() == False:
        pygame.mixer.music.load("/home/pi/Downloads/left.mp3")
        pygame.mixer.music.play()

def right():
    pygame.mixer.music.load("/home/pi/Downloads/right.mp3")
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy() == True:
        continue
    
def right2():
    if pygame.mixer.music.get_busy() == False:
        pygame.mixer.music.load("/home/pi/Downloads/right.mp3")
        pygame.mixer.music.play()

def center():
    pygame.mixer.music.load("/home/pi/Downloads/center.mp3")
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy() == True:
        continue

def center2():
    if pygame.mixer.music.get_busy() == False:
        pygame.mixer.music.load("/home/pi/Downloads/center.mp3")
        pygame.mixer.music.play()


def run(model: str, camera_id: int, width: int, height: int, num_threads: int,
        enable_edgetpu: bool, object_name: str) -> None:
  """Continuously run inference on images acquired from the camera.

  Args:
    model: Name of the TFLite object detection model.
    camera_id: The camera id to be passed to OpenCV.
    width: The width of the frame captured from the camera.
    height: The height of the frame captured from the camera.
    num_threads: The number of CPU threads to run the model.
    enable_edgetpu: True/False whether the model is a EdgeTPU model.
  """

  print(object_name)

  # Variables to calculate FPS
  counter, fps = 0, 0
  start_time = time.time()

  # Start capturing video input from the camera
  cap = cv2.VideoCapture(camera_id)
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

  # Visualization parameters
  row_size = 20  # pixels
  left_margin = 24  # pixels
  text_color = (0, 0, 255)  # red
  font_size = 1
  font_thickness = 1
  fps_avg_frame_count = 10
  
  pygame.mixer.init()

  # Continuously capture images from the camera and run inference
  while cap.isOpened():
    #cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    #cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

    success, image = cap.read()
    if not success:
      sys.exit(
          'ERROR: Unable to read from webcam. Please verify your webcam settings.'
      )

    counter += 1

    # Calculate the FPS
    if counter % fps_avg_frame_count == 0:
      end_time = time.time()
      fps = fps_avg_frame_count / (end_time - start_time)
      start_time = time.time()

    # Show the FPS
    fps_text = 'FPS = {:.1f}'.format(fps)
    print(fps_text)
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # red color boundaries [B, G, R]
    #lower = [1, 0, 20]
    #upper = [60, 40, 200]
    lower = np.array([160,20,10])
    upper = np.array([190,255,255])

    lower = np.array(lower, dtype="uint8")
    upper = np.array(upper, dtype="uint8")

    mask = cv2.inRange(image, lower, upper)
    output = cv2.bitwise_and(image, image, mask=mask)
    ret,thresh = cv2.threshold(mask, 220, 255, cv2.THRESH_BINARY)
    contours,hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(len(contours))
    #cv2.drawContours(output, contours, -1, (0, 255, 255), 3)
    if len(contours) > 0:
        red_area = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(red_area)
        #print(x, y, w, h)
        cv2.rectangle(output,(x, y),(x+w, y+h),(0, 0, 255), 2)
        if (x + w/2) < (320 - w/2):
            left2()
            print("LEFT")
        elif (x + w/2) > (320 + w/2):
            right2()
            print("RIGHT")
        else:
            center2()
            print("CENTER")
            
    cv2.imshow("Result", np.hstack([image, output]))
        
        

    # Stop the program if the ESC key is pressed.
    if cv2.waitKey(1) == 27:
      break
    #cv2.imshow('object_detector', image)

  cap.release()
  cv2.destroyAllWindows()


def main():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '--model',
      help='Path of the object detection model.',
      required=False,
      default='efficientdet_lite0.tflite')
  parser.add_argument(
      '--cameraId', help='Id of camera.', required=False, type=int, default=0)
  parser.add_argument(
      '--frameWidth',
      help='Width of frame to capture from camera.',
      required=False,
      type=int,
      default=640)
  parser.add_argument(
      '--frameHeight',
      help='Height of frame to capture from camera.',
      required=False,
      type=int,
      default=480)
  parser.add_argument(
      '--numThreads',
      help='Number of CPU threads to run the model.',
      required=False,
      type=int,
      default=4)
  parser.add_argument(
      '--enableEdgeTPU',
      help='Whether to run the model on EdgeTPU.',
      action='store_true',
      required=False,
      default=False)
  args = parser.parse_args()

  #NAME = getObjName()
  NAME = "apple"
  run(args.model, int(args.cameraId), args.frameWidth, args.frameHeight,
      int(args.numThreads), bool(args.enableEdgeTPU), NAME)


if __name__ == '__main__':
  main()
