#
# Project LAKSHYA
# Author: Ronav Jaiswal
# Copyright 2023. All Rights Reserved.
#
import argparse
import sys
import time
import pygame

import cv2
from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision
import utils

import speech_recognition as sr
from datetime import date
from time import sleep

from tensorflow_lite_support.cc.task.processor.proto import bounding_box_pb2

def left():
    pygame.mixer.music.load("/home/pi/Downloads/left.mp3")
    pygame.mixer.music.play()

def right():
    pygame.mixer.music.load("/home/pi/Downloads/right.mp3")
    pygame.mixer.music.play()

def center():
    pygame.mixer.music.load("/home/pi/Downloads/center.mp3")
    pygame.mixer.music.play()

def left2():
    if pygame.mixer.music.get_busy() == False:
        pygame.mixer.music.load("/home/pi/Downloads/left.mp3")
        pygame.mixer.music.play()

def right2():
    if pygame.mixer.music.get_busy() == False:
        pygame.mixer.music.load("/home/pi/Downloads/right.mp3")
        pygame.mixer.music.play()

def center2():
    if pygame.mixer.music.get_busy() == False:
        pygame.mixer.music.load("/home/pi/Downloads/center.mp3")
        pygame.mixer.music.play()

def up2():
    if pygame.mixer.music.get_busy() == False:
        pygame.mixer.music.load("/home/pi/Downloads/LAKSHYA Audio Files/audio files/audipio/up.mp3")
        pygame.mixer.music.play()

def down2():
    if pygame.mixer.music.get_busy() == False:
        pygame.mixer.music.load("/home/pi/Downloads/LAKSHYA Audio Files/audio files/audipio/down.mp3")
        pygame.mixer.music.play()

def getObjName() -> str:
    r = sr.Recognizer()

    with sr.Microphone() as source2:
        print("a")
        r.adjust_for_ambient_noise(source2, duration=0.2)
        print("Speak")
        audio2 = r.listen(source2)
        print("Audio Received")
        mytext = r.recognize_google(audio2, show_all=True)
        #print(mytext)
        if len(mytext) > 0 and mytext['alternative'] and len(mytext['alternative']) > 0 and mytext['alternative'][0]['transcript']:
        #    print(mytext['alternative'][0]['transcript'])
            name = mytext['alternative'][0]['transcript']
            name = name.lower()
            
    return name

def run(model: str, camera_id: int, width: int, height: int, num_threads: int,
        enable_edgetpu: bool, object_name: str) -> None:
  """Continuously run inference on images acquired from the camera.
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

  # Initialize the object detection model
  base_options = core.BaseOptions(
      file_name=model, use_coral=enable_edgetpu, num_threads=num_threads)
  detection_options = processor.DetectionOptions(
      max_results=5, score_threshold=0.4)
  options = vision.ObjectDetectorOptions(
      base_options=base_options, detection_options=detection_options)
  detector = vision.ObjectDetector.create_from_options(options)

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
    image = cv2.flip(image, 1)

    # Convert the image from BGR to RGB as required by the TFLite model.
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Create a TensorImage object from the RGB image.
    input_tensor = vision.TensorImage.create_from_array(rgb_image)

    # Run object detection estimation using the model.
    detection_result = detector.detect(input_tensor)

    # Draw keypoints and edges on input image
    image = utils.visualize(image, detection_result)

    # Calculate the FPS
    if counter % fps_avg_frame_count == 0:
      end_time = time.time()
      fps = fps_avg_frame_count / (end_time - start_time)
      start_time = time.time()

    # Show the FPS
    fps_text = 'FPS = {:.1f}'.format(fps)

    print(fps_text)
    #print(detection_result.detections[0].bounding_box)
    #print(detection_result.detections[0].categories[0])

    object_found = False
    mybox = bounding_box_pb2.BoundingBox()
    for label in detection_result.detections:
        #print(label.categories[0])
        if label.categories[0].category_name == object_name:
            mybox = label.bounding_box
            object_found = True
    
    if object_found == True:
        print("OBJECT FOUND!")
        print(mybox)
        if (mybox.origin_x + mybox.width/2) < (300 - mybox.width/2):
            print("MOVE RIGHT")
            right2()
         
            
        elif (mybox.origin_x + mybox.width/2) > (300 + mybox.width/2):
            print("MOVE LEFT")
            left2()
           
        else:
            print("OBJECT IS IN CENTER")           
            if ((mybox.origin_y + mybox.height/2) < (240 - mybox.height/2)):
                print("MOVE DOWN")
                down2()
            elif (mybox.origin_y + mybox.height/2) > (240 + mybox.height/2):
                print("MOVE UP")
                up2()
            else:
                print("OBJECT FOUND")
                center2()

    text_location = (left_margin, row_size)
    cv2.putText(image, fps_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                font_size, text_color, font_thickness)

    # Stop the program if the ESC key is pressed.
    if cv2.waitKey(1) == 27:
      break
    cv2.imshow('object_detector', image)

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
      #default='/home/pi/Documents/Keras/TFlite/model.tflite')
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


  
  while True:
      #inp = input()
      #x = int(inp)
      x = 1
      if(x == 1):
          NAME = getObjName()
          # NAME = "banana"
          pygame.mixer.init()
          run(args.model, int(args.cameraId), args.frameWidth, args.frameHeight,
          int(args.numThreads), bool(args.enableEdgeTPU), NAME)
      if(x == 0):
          break 
     
if __name__ == '__main__':
  main()
