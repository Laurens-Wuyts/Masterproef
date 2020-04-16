import matplotlib
matplotlib.use("Agg")

# Import the network model from FastDepthNet.py
from FastDepthNet import FastDepthNet

# Import the dataset loader from ImageDataset.ŷ
from ImageDataset import Load_Dataset
from ImageDataset import Load_Dummy_Dataset

# Import a logger from utils.py
from utils import infoPrint


import tensorflow as tf
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks  import TensorBoard
from tensorflow.keras.callbacks  import ModelCheckpoint
from tensorflow.keras.callbacks  import EarlyStopping

import sys
from datetime import datetime

NUM_EPOCHS = 200	# Maximum number of times to run the network
BS = 64				# Batch size of the dataset

DATA_PATH = sys.argv[1] + "Data/"	# Path to the datasets


# Check if it needs to start from a checkpoint
checkpoint = False
if len(sys.argv) > 3:
	checkpoint = True

# Initialise the start time of the debug messages
infoPrint.startTime = datetime.now()

# Load the training and testing data in two separate TensorFlow datasets
infoPrint("Loading training and testing data...")
train_ds = Load_Dataset(DATA_PATH + sys.argv[2] + "/preprocessed/Train/color", BS)
test_ds  = Load_Dataset(DATA_PATH + sys.argv[2] + "/preprocessed/Test/color", BS)

# Build the layers of the network
infoPrint("Building model...")
model = FastDepthNet.build()

# Compile the model with a specific optimizer and loss function
infoPrint("Compiling model...")
sgd = SGD(lr=0.0001, decay=1e-4, momentum=0.9, nesterov=True)
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
	epochs 			= NUM_EPOCHS,	# Run for a maximum of NUM_EPOCHS
	steps_per_epoch = 318,			# Run 318 batches of data every epoch
	verbose 		= 1,			# Print a lot of debug info
	callbacks 		= [tensorboard_callback, checkpoint_callback, earlystop_callback])	# Load al the different callbacks

# Save the network for later use
infoPrint("Serializing network to '{}'...".format("output/FastDepth.model"))
model.save("output/FastDepth.model")
