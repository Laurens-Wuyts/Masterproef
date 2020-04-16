import matplotlib
matplotlib.use("Agg")

from FastDepthNet import FastDepthNet

from ImageDataset import Load_Dataset
from ImageDataset import Load_Dummy_Dataset

from utils import infoPrint


import tensorflow as tf
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks  import TensorBoard
from tensorflow.keras.callbacks  import ModelCheckpoint
from tensorflow.keras.callbacks  import EarlyStopping

import sys
from datetime import datetime

NUM_EPOCHS = 200
BS = 64

DATA_PATH = sys.argv[1] + "Data/"


checkpoint = False
if len(sys.argv) > 3:
	checkpoint = True

infoPrint.startTime = datetime.now()


infoPrint("Loading training and testing data...")
train_ds = Load_Dataset(DATA_PATH + sys.argv[2] + "/preprocessed/Train/color", BS)
test_ds  = Load_Dataset(DATA_PATH + sys.argv[2] + "/preprocessed/Test/color", BS)

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

H = model.fit(
	x 				= train_ds,
	validation_data	= test_ds,
	validation_steps= 20,
	epochs 			= NUM_EPOCHS,
	steps_per_epoch = 318,
	verbose 		= 1,
	callbacks 		= [tensorboard_callback, checkpoint_callback, earlystop_callback])

infoPrint("Serializing network to '{}'...".format("output/FastDepth.model"))
model.save("output/FastDepth.model")
