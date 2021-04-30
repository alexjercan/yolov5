"""Perform test request"""
import os
import io
import cv2
import pickle
import numpy as np

import pprint

import requests


def png2bgr(path):
    if not os.path.isfile(path):
            return None

    img = cv2.imread(path)
    img = np.array(img).astype(np.float32).reshape(img.shape[0], img.shape[1], -1)
    img = img / 255
    
    return img


def exr2depth(path):
    if not os.path.isfile(path):
            return None
        
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH)

    maxvalue = np.max(img[img < np.max(img)])
    img[img > maxvalue] = maxvalue
    img = img / maxvalue

    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = np.array(img).astype(np.float32).reshape(img.shape[0], img.shape[1], -1)

    return img


def exr2normal(path):
    if not os.path.isfile(path):
            return None

    img = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    img = np.array(img).astype(np.float32).reshape(img.shape[0], img.shape[1], -1)

    return img

def save_layers(path, data: np.ndarray):
    cv2.imwrite(path, data, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

DETECTION_URL = "http://localhost:5000/v1/object-detection/yolov5s"
TEST_RGB = "000000rgb.png"
TEST_DEPTH = "000000depth.exr"
TEST_NORMAL = "000000normal.exr"

image_data = png2bgr(TEST_RGB)
depth_data = exr2depth(TEST_DEPTH)
normal_data = exr2normal(TEST_NORMAL)

layers = np.concatenate([image_data, depth_data, normal_data], axis=2)

bytes_io = io.BytesIO()
pickle.dump(layers, bytes_io)
bytes_io.seek(0)
image_data = bytes_io.read()

response = requests.post(DETECTION_URL, files={"image": image_data}).json()

pprint.pprint(response)
