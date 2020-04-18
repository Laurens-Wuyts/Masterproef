import numpy as np
import cv2
import os
import sys
import random

import multiprocessing
from multiprocessing import Pool
from multiprocessing import cpu_count

import matplotlib.pyplot as plt

directory = sys.argv[1]
preprocessedDir = directory + "preprocessed/"

def removeBlack(img):
	img = img.copy()
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			if img[i][j] < 20:
				temp = np.zeros(4, dtype=int)
				if i > 0:
					temp[0] = img[i-1][j]
				if i < img.shape[0] - 1:
					temp[1] = img[i+1][j]
				if j > 0:
					temp[2] = img[i][j-1]
				if j < img.shape[1] - 1:
					temp[3] = img[i][j+1]
				
				img[i][j] = max(temp)
	return img


def preprocessImage(dire, file):
	depth_image = cv2.imread(directory + "depth/" + file, cv2.IMREAD_GRAYSCALE)
	color_image = cv2.imread(directory + "color/" + file)
	
	depth_crop = depth_image[104:616, 384:896]
	depth_crop = cv2.resize(depth_crop,  (224, 224))
	color_crop = cv2.resize(color_image, (398, 224))
	color_crop = color_crop[0:224, 87:311]

	depth_crop = removeBlack(depth_crop)

	cv2.imwrite(preprocessedDir + dire + "depth/" + file, depth_crop)
	cv2.imwrite(preprocessedDir + dire + "color/" + file, color_crop)

def preprocessImageTrain(file):
	preprocessImage("Train/", file)
def preprocessImageTest(file):
	preprocessImage("Test/", file)


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

lenTrain = int(len(dep) * 0.8)
trainSplit = dep[:lenTrain]
testSplit  = dep[lenTrain:]

p = Pool(cpu_count())
p.map(preprocessImageTrain, trainSplit)
p.map(preprocessImageTest, testSplit)
p.close()