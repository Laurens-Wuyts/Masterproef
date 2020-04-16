import tensorflow as tf

from tensorflow.keras.models import load_model
from skimage import io
from imutils import paths
import numpy as np
import argparse
import random
import cv2
import os

from ImageDataset import Load_Dataset

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path",    required=True, help="path to root of project")
ap.add_argument("-m", "--model",   required=True, help="path to pre-trained network")
ap.add_argument("-d", "--dataset", required=True, help="path to preprocessed dataset")
args = vars(ap.parse_args())

DATA_PATH = args["path"] + "Data/"

print("[INFO] loading images...")
test_ds  = Load_Dataset(DATA_PATH + args["dataset"] + "/preprocessed/Test/color", 5).take(1)
for i, d in test_ds:
	color_inp = i.numpy()
	depth_inp = d.numpy()

print("[INFO] loading model...")
model = load_model(args["model"])

print("[INFO] predicting...")
preds = model.predict(color_inp)

print("[INFO] visualising...")
image = None

for idx in range(len(color_inp)):
	tmp_pred = tf.image.convert_image_dtype(preds[idx], tf.uint8)
	tmp_d    = tf.image.convert_image_dtype(depth_inp[idx], tf.uint8)
	tmp_i    = tf.image.convert_image_dtype(color_inp[idx], tf.uint8)
	tmp_pred = np.stack((np.squeeze(tmp_pred),)*3, axis=-1)
	tmp_d    = np.stack((np.squeeze(tmp_d),)*3, axis=-1)

	temp = np.hstack((tmp_i, tmp_d, tmp_pred))
	if image is None:
		image = temp.copy()
	else:
		image = np.vstack((image, temp))

cv2.imwrite(args["path"] + "Data/predictions.jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))