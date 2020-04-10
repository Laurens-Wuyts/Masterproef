import matplotlib
matplotlib.use("Agg")

from FastDepthNet import FastDepthNet

import tensorflow as tf
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping

from ImageDataset import Load_Dataset
from ImageDataset import Load_Dummy_Dataset

from skimage import io

import numpy as np
import random
import os
import sys
import h5py
from datetime import datetime

def infoPrint(str):
	diff = datetime.now() - infoPrint.startTime
	print("\033[92m[INFO] \033[94m{0:>5} \033[0m{1}".format(diff.seconds, str))

def loadSplit(path):
	dep = [f for f in os.listdir(path + "depth/") if os.path.isfile(path + "depth/" + f)]
	random.shuffle(dep)
	inp = []
	outp= []

	idx = 0
	for f in dep:
		idx+=1
		if idx % 100 == 0:
			infoPrint("Loaded {} images...".format(idx))
		inp.append(io.imread(path + "color/" + f)) 
		outp.append(io.imread(path + "depth/" + f, as_gray=True)) 

	data = np.array(inp)
	output = np.array(outp)

	return (data, output)

def loadData(fn):
	with h5py.File(fn, 'r') as f:
		i1 = np.array(f["trainX"])# .transpose(0, 3, 2, 1)
		d1 = np.array(f["trainY"])# .transpose(0, 2, 1)
		i2 = np.array(f["testX"])# .transpose(0, 3, 2, 1)
		d2 = np.array(f["testY"])# .transpose(0, 2, 1)

		return (i1, d1, i2, d2)


# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

NUM_EPOCHS = 200
BS = 64

DATA_PATH = sys.argv[1] + "Data/"


checkpoint = False
if len(sys.argv) > 3:
	checkpoint = True

infoPrint.startTime = datetime.now()


infoPrint("Loading training and testing data...")
#trainPath = DATA_PATH + sys.argv[2] + "/preprocessed/Train/"
#testPath  = DATA_PATH + sys.argv[2] + "/preprocessed/Test/"

#(trainX, trainY) = loadSplit(trainPath)
#(testX, testY)   = loadSplit(testPath)

#trainX, trainY, testX, testY = loadData(DATA_PATH + sys.argv[2])

#trainX = trainX.astype("float32") / 255.0
#trainY = trainY.astype("float32") / 255.0
#testX  = testX.astype("float32") / 255.0
#testY  = testY.astype("float32") / 255.0

#train_ds = Load_Dataset(DATA_PATH + sys.argv[2] + "/preprocessed/Train/color", BS)
#test_ds  = Load_Dataset(DATA_PATH + sys.argv[2] + "/preprocessed/Test/color", BS)

train_ds = Load_Dummy_Dataset(DATA_PATH + sys.argv[2] + "/preprocessed/Train/color", BS)
test_ds  = Load_Dummy_Dataset(DATA_PATH + sys.argv[2] + "/preprocessed/Test/color", BS)

infoPrint("Building model...")
model = FastDepthNet.build()

infoPrint("Compiling model...")
sgd = SGD(lr=0.0001, decay=1e-4, momentum=0.9, nesterov=True)
model.compile(loss="mae", optimizer=sgd, metrics=["accuracy"]) # mae = Mean absolute Error = L1 Loss  mean(abs(T - P))

infoPrint("Training network...")
logdir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=logdir )
earlystop_callback   = EarlyStopping( monitor="val_loss", patience=15 )
checkpoint_callback  = ModelCheckpoint(
	"checkpoints/{epoch}-fast-depth-cp.h5", 
	monitor= "val_loss", 
	save_weights_only = True,
	mode   = "min",
	period = 10 )

if checkpoint:
	latest = tf.train.latest_checkpoint("checkpoints/")
	print(latest)
	model.load_weights(latest)
	first_epoch = int(latest.split("-")[0])
	print(first_epoch)

# H = model.fit(
# 	x=trainX,
# 	y=trainY, 
# 	batch_size=BS,
# 	validation_data=(testX, testY),
# 	# steps_per_epoch=trainX.shape[0] // BS,
# 	epochs=NUM_EPOCHS,
# 	verbose=1,
# 	callbacks=[tensorboard_callback, checkpoint_callback, earlystop_callback])

H = model.fit(
	x 				= train_ds,
	validation_data	= test_ds,
	validation_steps= 20,
	epochs 			= NUM_EPOCHS,
	steps_per_epoch = 128,
	verbose 		= 1,
	callbacks 		= [tensorboard_callback, checkpoint_callback, earlystop_callback])



# infoPrint("Evaluating network...")
# predictions = model.predict(testX, batch_size=BS)

infoPrint("Serializing network to '{}'...".format("output/FastDepth.model"))
model.save("output/FastDepth.model")
