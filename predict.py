import tensorflow as tf

from tensorflow.keras.models import load_model
import numpy as np
import argparse
import cv2

from ImageDataset import Load_Dataset
from utils import infoPrint

from datetime import datetime

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model",   	required=True,	help="Path to pre-trained network")
ap.add_argument("-d", "--dataset", 	required=True,	help="Path to preprocessed dataset")
ap.add_argument("-b", "--batch",	required=False, help="Number of images to predict", default=5, type=int)
args, _ = ap.parse_known_args()

infoPrint.startTime = datetime.now()

infoPrint("loading images...")
_, test_ds  = Load_Dataset(args.dataset + "/preprocessed/Test/color", args.batch).take(1)
color_inp, depth_inp = [i.numpy(), d.numpy() for i, d in test_ds]

#for i, d in test_ds:
	#color_inp = i.numpy()
	#depth_inp = d.numpy()

infoPrint("loading model...")
model = load_model(args.model)

infoPrint("predicting...")
preds = model.predict(color_inp)

infoPrint("visualising...")
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

cv2.imwrite(args.dataset + "predictions.jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))