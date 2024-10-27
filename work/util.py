import cv2
import numpy as np
import torch


def box2d_preProcess(image):
    image = image[:84, 6:90]  # CarRacing-v2-specific cropping
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) / 255.0
    return image

def atari_preProcess(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # To grayscale
    image = image[20:210:2, 0:160:2]
    #image = cv2.resize(image, (64, 80))  # Resize
    image = image.reshape(95, 80) / 255  # Normalize

    return image

def display_action_distribution(actions, action_num):
    display = '['
    for i in range(action_num):
        display = (display + str(i) + ':' + str(actions.count(i))
                   + '/' + str(len(actions)) + ' | ')
    display = display + ']'
    return display