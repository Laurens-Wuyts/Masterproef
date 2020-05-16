import tensorflow as tf

from tensorflow.keras.models import load_model
import numpy as np
import argparse

from ImageDataset import Load_Dataset
from utils import infoPrint

from datetime import datetime

import math

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model",   	required=True,	help="Path to trained network")
ap.add_argument("-d", "--dataset", 	required=True,	help="Path to preprocessed dataset")
ap.add_argument("-b", "--batch",	required=False, help="Number of images to evaluate", default=5, type=int)
args, _ = ap.parse_known_args()

infoPrint.startTime = datetime.now()

infoPrint("loading images...")
_, test_ds  = Load_Dataset(args.dataset + "/preprocessed/Test/color", args.batch)

for i, d in test_ds.take(1):
	color_inp = i.numpy()
	depth_inp = d.numpy()

infoPrint("loading model...")
model = load_model(args.model)

infoPrint("evaluating...")
preds = model.predict(color_inp)

max_delta_1 = 0
max_delta_2 = 0
max_delta_3 = 0
min_sq      = 0


thr_1 = 1.25
thr_2 = 1.5625
thr_3 = 1.953125

for idx in range(len(color_inp)):
	p = np.squeeze(tf.image.convert_image_dtype(preds[idx], tf.uint8))
	d = np.squeeze(tf.image.convert_image_dtype(depth_inp[idx], tf.uint8))

	w = d.shape[0]
	h = d.shape[1]

	cnt_1 = 0
	cnt_2 = 0
	cnt_3 = 0

	n = w * h
	sum_sq = 0
	for y in range(h):
		for x in range(w):
			tmp = 0
			if p[y][x] == 0 or d[y][x] == 0:	
				tmp = thr_3 * 2
			else:
				tmp = max(p[y][x] / d[y][x], d[y][x] / p[y][x])
			if tmp < thr_1:	cnt_1 += 1
			if tmp < thr_2:	cnt_2 += 1
			if tmp < thr_3:	cnt_3 += 1

			sum_sq +=( ((d[y][x] - p[y][x]) ** 2) / n)
	delta_1 = cnt_1 / n
	delta_2 = cnt_2 / n
	delta_3 = cnt_3 / n
	sum_rt  = math.sqrt(sum_sq)

	if delta_1 > max_delta_1: max_delta_1 = delta_1
	if delta_2 > max_delta_2: max_delta_2 = delta_2
	if delta_3 > max_delta_3: max_delta_3 = delta_3
	if sum_rt  < min_sq     : min_sq      = sum_rt

print("Delta 1: {}".format(max_delta_1))
print("Delta 2: {}".format(max_delta_2))
print("Delta 3: {}".format(max_delta_3))
print("RMSE   : {}".format(min_sq))


				
		
