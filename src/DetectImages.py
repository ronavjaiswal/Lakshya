#
# Project LAKSHYA
# Author: Ronav Jaiswal
# Copyright 2023. All Rights Reserved.
#

import argparse
import time
import cv2
import pygame
from time import sleep

import numpy as np
from PIL import Image
#import tensorflow as tf
import tflite_runtime.interpreter as tflite


def load_labels(filename):
  with open(filename, 'r') as f:
    return [line.strip() for line in f.readlines()]

def speak(name):
    new_name = name.split()[1].lower()

    pygame.mixer.music.load("/home/pi/Downloads/LAKSHYA Audio Files/audio files/aapke saamne hain.mp3")
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy()==True:
        continue

    pygame.mixer.music.load("/home/pi/Downloads/LAKSHYA Audio Files/audio files/"+new_name+".mp3")
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy()==True:
        continue
    
if __name__ == '__main__':
  pygame.mixer.init()
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '-i',
      '--image',
      default='/tmp/grace_hopper.bmp',
      help='image to be classified')
  parser.add_argument(
      '-m',
      '--model_file',
      default='model5.tflite',
      help='.tflite model to be executed')
  parser.add_argument(
      '-l',
      '--label_file',
      default='labels5.txt',
      help='name of file containing labels')
  parser.add_argument(
      '--input_mean',
      default=127.5, type=float,
      help='input_mean')
  parser.add_argument(
      '--input_std',
      default=127.5, type=float,
      help='input standard deviation')
  parser.add_argument(
      '--num_threads', default=None, type=int, help='number of threads')
  parser.add_argument(
      '-e', '--ext_delegate', help='external_delegate_library path')
  parser.add_argument(
      '-o',
      '--ext_delegate_options',
      help='external delegate options, \
            format: "option1: value1; option2: value2"')

  # Start capturing video input from the camera
  camera_id = 0
  counter = 0
  cap = cv2.VideoCapture(camera_id)
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, 224)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 224)

  args = parser.parse_args()

  ext_delegate = None
  ext_delegate_options = {}

  interpreter = tflite.Interpreter(
      model_path=args.model_file,
      experimental_delegates=ext_delegate,
      num_threads=args.num_threads)
  
  interpreter.allocate_tensors()

  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()

  # check the type of the input tensor
  floating_model = input_details[0]['dtype'] == np.float32

  # NxHxWxC, H:1, W:2
  height = input_details[0]['shape'][1]
  width = input_details[0]['shape'][2]
  
  labels = load_labels(args.label_file)  
  
  # Continuously capture images from the camera and run inference
  while cap.isOpened():
    #img = Image.open(args.image).resize((width, height))
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
    #input_tensor = vision.TensorImage.create_from_array(rgb_image)

    # add N dim
    input_data = np.expand_dims(image, axis=0)

    if floating_model:
      input_data = (np.float32(input_data) - args.input_mean) / args.input_std

    interpreter.set_tensor(input_details[0]['index'], input_data)

    start_time = time.time()
    interpreter.invoke()
    stop_time = time.time()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    results = np.squeeze(output_data)

    top_k = results.argsort()[-5:][::-1]

    image_label = labels[top_k[0]]
    image_score = results[top_k[0]] / 255.0
    test = False
    
    if (image_score > 0.9):
      print(image_label, image_score)
      speak(image_label)
      cap.release()
      cap = cv2.VideoCapture(camera_id)
      cap.set(cv2.CAP_PROP_FRAME_WIDTH, 224)
      cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 224)
      #sleep(5)
          
      
    print('time: {:.3f}ms'.format((stop_time - start_time) * 1000))

    # Stop the program if the ESC key is pressed.
    if cv2.waitKey(1) == 27:
      break
    #cv2.imshow('object_detector', image)

  cap.release()
  cv2.destroyAllWindows()
