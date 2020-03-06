import numpy as np
import cv2
import os
import sys
import random


directory = sys.argv[1] + 'Data/' + sys.argv[2] + "/"
preprocessedDir = directory + "preprocessed/"

if not os.path.exists(preprocessedDir + "Train/depth/"):
	os.makedirs(preprocessedDir + "Train/depth/")
if not os.path.exists(preprocessedDir + "Test/depth/"):
	os.makedirs(preprocessedDir + "Test/depth/")
if not os.path.exists(preprocessedDir + "Train/color/"):
	os.makedirs(preprocessedDir + "Train/color/")
if not os.path.exists(preprocessedDir + "Test/color/"):
	os.makedirs(preprocessedDir + "Test/color/")

dep = [f for f in os.listdir(directory + "depth/") if os.path.isfile(directory + "depth/" + f)]
random.shuffle(dep)
idx = 0

for f in dep:
	idx += 1
	if idx % 1000 == 0:
		print("Preprocessed {} images.".format(idx))

	depth_image = cv2.imread(directory + "depth/" + f)
	color_image = cv2.imread(directory + "color/" + f)
	
	depth_crop = depth_image[104:616, 384:896]
	depth_crop = cv2.resize(depth_crop,  (224, 224))
	color_crop = cv2.resize(color_image, (398, 224))
	color_crop = color_crop[0:224, 87:311]

	if idx < 250:
		cv2.imwrite(preprocessedDir + "Test/depth/" + f, depth_crop)
		cv2.imwrite(preprocessedDir + "Test/color/" + f, color_crop)
	else:
		cv2.imwrite(preprocessedDir + "Train/depth/" + f, depth_crop)
		cv2.imwrite(preprocessedDir + "Train/color/" + f, color_crop)
