#
# Project LAKSHYA
# Author: Ronav Jaiswal
# Copyright 2023. All Rights Reserved.
#

from machine import Pin
from time import sleep
import random


def SlowPattern(vibrate_pin):
    for i in range(0,4):
        Pin(vibrate_pin, Pin.OUT).value(1)
        sleep(0.1)
        Pin(vibrate_pin, Pin.OUT).value(0)
        sleep(0.4)

def MediumPattern(vibrate_pin):
    for i in range(0,4):
        Pin(vibrate_pin, Pin.OUT).value(1)
        sleep(0.25)
        Pin(vibrate_pin, Pin.OUT).value(0)
        sleep(0.25)

def FastPattern(vibrate_pin):
    for i in range(0,4):
        Pin(vibrate_pin, Pin.OUT).value(1)
        sleep(0.1)
        Pin(vibrate_pin, Pin.OUT).value(0)
        sleep(0.1)

def ContinuousPattern(vibrate_pin):
    Pin(vibrate_pin, Pin.OUT).value(1)
    sleep(2)
    Pin(vibrate_pin, Pin.OUT).value(0)


# simulation of haptic feedback
LEFT_PIN = 2
RIGHT_PIN = 4
UP_PIN = 5
DOWN_PIN = 18

pinChoices = [ LEFT_PIN, RIGHT_PIN, UP_PIN, DOWN_PIN ]

while True:
    pinChoice = random.choice(pinChoices)
    patternChoice = random.randint(0, 3)
    print(pinChoice, patternChoice)
    if (patternChoice == 0):
        SlowPattern(pinChoice)
    elif (patternChoice == 1):
        MediumPattern(pinChoice)
    elif (patternChoice == 2):
        FastPattern(pinChoice)
    else:
        ContinuousPattern(pinChoice)
            
    sleep(1)
