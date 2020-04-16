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
# images = []
# depths = []
# 
# random.seed()
# 
# for _ in range(5):
# 	idx = random.randint(0, f["testX"].shape[0])
# 	i = f["testX"][idx]
# 	images.append(i)
# 	d = f["testY"][idx]
# 	depths.append(d)
# 
# images = np.asarray(images)
# depths = np.asarray(depths)
# print(images.shape)

test_ds  = Load_Dataset(DATA_PATH + args["dataset"] + "/preprocessed/Test/color", 5).take(1)
for i, d in test_ds:
	color_inp = i.as_numpy()
	depth_inp = d.as_numpy()
	print(color_inp.shape, depth_inp.shape)

print("[INFO] loading model...")
model = load_model(args["model"])

print("[INFO] predicting...")
preds = model.predict(color_inp)

#preds = 255 * preds
#pred_imgs = preds.astype(np.uint8)
#print(pred_imgs.shape)

#im_preds_color = np.stack((np.squeeze(pred_imgs),)*3, axis=-1)
#print(im_preds_color.shape, images.shape)

print("[INFO] visualising...")
image = None

for idx in range(len(color_inp)):
	tmp_pred = tf.image.convert_image_dtype(preds[idx], tf.uint8)
	tmp_d    = tf.image.convert_image_dtype(depth_inp[idx], tf.uint8)
	tmp_i    = tf.image.convert_image_dtype(color_inp[idx], tf.uint8)
	tmp_pred = np.stack((np.squeeze(tmp_pred),)*3, axis=-1)
	tmp_d    = np.stack((np.squeeze(tmp_d),)*3, axis=-1)
	print(idx, tmp_i.shape, tmp_d.shape, tmp_pred.shape)
	temp = np.hstack((tmp_i, tmp_d, tmp_pred))
	if image is None:
		image = temp.copy()
	else:
		image = np.vstack((image, temp))

cv2.imwrite(args["path"] + "Data/predictions.jpg", image)