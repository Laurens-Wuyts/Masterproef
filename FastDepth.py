import matplotlib
matplotlib.use("Agg")

from FastDepthNet import FastDepthNet

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard

from sklearn.metrics import classification_report
from skimage import io

import matplotlib.pyplot as plt
import numpy as np
import random
import os
import sys
from datetime import datetime

def infoPrint(str):
	diff = datetime.now() - infoPrint.startTime
	print("\033[92m[INFO] \033[94m{0:>5} \033[0m{1}".format(diff.seconds, str))

def loadSplit(path):
	dep = [f for f in os.listdir(path + "depth/") if os.path.isfile(path + "depth/" + f)]
	random.shuffle(dep)
	inp = []
	outp= []
	print(dep)

	for f in dep:
		inp.append(io.imread(path + "color/" + f)) 
		outp.append(io.imread(path + "depth/" + f, as_gray=True)) 

	data = np.array(inp)
	output = np.array(outp)

	return (data, output)

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

NUM_EPOCHS = 30
INIT_LR = 1e-3
BS = 8

DATA_PATH = sys.argv[1] + "Data/"

infoPrint.startTime = datetime.now()


infoPrint("Loading training and testing data...")
trainPath = DATA_PATH + sys.argv[2] + "/preprocessed/Train/"
testPath  = DATA_PATH + sys.argv[2] + "/preprocessed/Test/"

(trainX, trainY) = loadSplit(trainPath)
(testX, testY)   = loadSplit(testPath)

trainX = trainX.astype("float32") / 255.0
trainY = trainY.astype("float32") / 255.0
testX  = testX.astype("float32") / 255.0
testY  = testY.astype("float32") / 255.0

print(trainX.shape)
print(trainY.shape)


infoPrint("Building model...")
model = FastDepthNet.build()


infoPrint("Compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / (NUM_EPOCHS * 0.5))
model.compile(loss="mse", optimizer=opt, metrics=["accuracy"])
# model.summary()


infoPrint("Training network...")
logdir="logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=logdir)

H = model.fit(
	x=trainX,
	y=trainY, 
	batch_size=BS,
	validation_data=(testX, testY),
	steps_per_epoch=trainX.shape[0] // BS,
	epochs=NUM_EPOCHS,
	verbose=1,
	callbacks=[tensorboard_callback])


infoPrint("Evaluating network...")
predictions = model.predict(testX, batch_size=BS)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=labelNames))


infoPrint("Serializing network to '{}'...".format("output/FastDepth.model"))
model.save("output/FastDepth.model")
# 
# N = np.arange(0, NUM_EPOCHS)
# plt.style.use("ggplot")
# plt.figure()
# plt.plot(N, H.history["loss"], label="train_loss")
# plt.plot(N, H.history["val_loss"], label="val_loss")
# plt.plot(N, H.history["accuracy"], label="train_acc")
# plt.plot(N, H.history["val_accuracy"], label="val_acc")
# plt.title("Training Loss and Accuracy on Dataset")
# plt.xlabel("Epoch #")
# plt.ylabel("Loss/Accuracy")
# plt.legend(loc="lower left")
# plt.savefig("output/graph.png")
