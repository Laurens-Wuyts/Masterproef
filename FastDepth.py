import matplotlib
matplotlib.use("Agg")

# Import the network model from FastDepthNet.py
from FastDepthNet import FastDepthNet

# Import the dataset loader from ImageDataset.py
from ImageDataset import Load_Dataset

# Import a logger from utils.py
from utils import infoPrint


import tensorflow as tf
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks  import TensorBoard
from tensorflow.keras.callbacks  import ModelCheckpoint
from tensorflow.keras.callbacks  import EarlyStopping

import argparse
from datetime import datetime

ap = argparse.ArgumentParser()
ap.add_argument("path",	 nargs=1,	  	help="Path to the dataset")
ap.add_argument("-o", "--output", 	 	required=False, help="Path to save the model",	 	default="FastDepth.model")
ap.add_argument("-b", "--batch_size",   required=False, help="Size of a batch", 		 	default=64,	type=int)
ap.add_argument("-e", "--epochs",  		required=False, help="Maximum number of epochs", 	default=200,	type=int)
ap.add_argument("-l", "--learning_rate",required=False, help="Learning rate of the network",default=0.001,	type=float)
ap.add_argument("-c", "--checkpoint",  	required=False, help="Start from checkpoint",	 	default="0",	const="1",	nargs="?")
args, _ = ap.parse_known_args()

# Initialise the start time of the debug messages
infoPrint.startTime = datetime.now()

# Load the training and testing data in two separate TensorFlow datasets
infoPrint("Loading training and testing data...")
train_ds = Load_Dataset(args.path[0] + "/preprocessed/Train/color", args.batch_size)
test_ds  = Load_Dataset(args.path[0] + "/preprocessed/Test/color",  args.batch_size)

# Build the layers of the network
infoPrint("Building model...")
model = FastDepthNet.build()

# Compile the model with a specific optimizer and loss function
infoPrint("Compiling model...")
sgd = SGD(lr=args.learning_rate, decay=1e-4, momentum=0.9, nesterov=True)
model.compile(loss="mae", optimizer=sgd, metrics=["accuracy"]) # mae = Mean absolute Error = L1 Loss  mean(abs(T - P))

# Training of the actual network
infoPrint("Training network...")
logdir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")		# Directory where to save the TensorBoard logs to.
tensorboard_callback = TensorBoard(log_dir=logdir )					# Define the TensorBoard callback
earlystop_callback   = EarlyStopping( monitor="val_loss", patience=15 )	# Define EarlyStopping callback to make sure the network stops when it stagnates.
checkpoint_callback  = ModelCheckpoint(								# Define a checkpoint callback, mostly for running in Colab and being disconnected
	"checkpoints/{epoch}-fast-depth-cp.h5", 
	monitor= "val_loss", 		# Monitor the validation loss
	save_weights_only = True,	# Only save the weights of the network, not the whole model
	mode   = "min",				# The loss should be lower to be better
	period = 10 )				# Save every 10 epochs

# Fit the model to the training data
H = model.fit(
	x 				= train_ds,		# The dataset with the training images
	validation_data	= test_ds,		# The dataset with the validation images
	validation_steps= 20,			# Validate 20 batches per epoch
	epochs 			= args.epochs,	# Run for a maximum of args.epochs
	steps_per_epoch = 318,			# Run 318 batches of data every epoch
	verbose 		= 1,			# Print a lot of debug info
	callbacks 		= [tensorboard_callback, checkpoint_callback, earlystop_callback])	# Load al the different callbacks

# Save the network for later use
infoPrint("Serializing network to '{}'...".format("output/FastDepth.model"))
model.save("output/" + args.output)
