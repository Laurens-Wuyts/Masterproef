import numpy as np
import cv2
import os
import sys
import random
import h5py

import multiprocessing
from multiprocessing import Pool
from multiprocessing import cpu_count

directory = sys.argv[1] + 'Data/' + sys.argv[2] + "/"
preprocessedDir = directory + "preprocessed/"

def preprocessImage(dire, file):
	depth_image = cv2.imread(directory + "depth/" + file, cv2.IMREAD_GRAYSCALE)
	color_image = cv2.imread(directory + "color/" + file)
	
	depth_crop = depth_image[104:616, 384:896]
	depth_crop = cv2.resize(depth_crop,  (224, 224))
	color_crop = cv2.resize(color_image, (398, 224))
	color_crop = color_crop[0:224, 87:311]

	# cv2.imwrite(preprocessedDir + dire + "depth/" + file, depth_crop)
	# cv2.imwrite(preprocessedDir + dire + "color/" + file, color_crop)
	return (color_crop, depth_crop)

def preprocessImageTrain(file):
	return preprocessImage("Train/", file)
def preprocessImageTest(file):
	return preprocessImage("Test/", file)


# if not os.path.exists(preprocessedDir + "Train/depth/"):
# 	os.makedirs(preprocessedDir + "Train/depth/")
# if not os.path.exists(preprocessedDir + "Test/depth/"):
# 	os.makedirs(preprocessedDir + "Test/depth/")
# if not os.path.exists(preprocessedDir + "Train/color/"):
# 	os.makedirs(preprocessedDir + "Train/color/")
# if not os.path.exists(preprocessedDir + "Test/color/"):
# 	os.makedirs(preprocessedDir + "Test/color/")

dep = [f for f in os.listdir(directory + "depth/") if os.path.isfile(directory + "depth/" + f)]
random.shuffle(dep)

lenTrain = int(len(dep) * 0.8)
trainSplit = dep[:lenTrain]
testSplit  = dep[lenTrain:]

p = Pool(cpu_count())
trainSet = p.map(preprocessImageTrain, trainSplit)
testSet  = p.map(preprocessImageTest, testSplit)
p.close()

trainSetX, trainSetY = zip(*trainSet)
testSetX, testSetY = zip(*testSet)

with h5py.File(directory + sys.argv[2] + "_preprocessed_comp.h5", "w") as f:
    f.create_dataset("trainX", data=trainSetX, compression="gzip")
    f.create_dataset("trainY", data=trainSetY, compression="gzip")
    f.create_dataset("testX",  data=testSetX,  compression="gzip")
    f.create_dataset("testY",  data=testSetY,  compression="gzip")