import tensorflow as tf
import numpy 	  as np
import h5py
import pathlib
import matplotlib.pyplot as plt

import time
import random

AUTOTUNE = tf.data.experimental.AUTOTUNE

# Extract the path to the depth image using the path to the color image
def get_depth_path(path):
	return tf.strings.regex_replace(path, "color", "depth")

# Decode the JPEG image to an array of bytes and convert it to floats
def decode_image(img, ch):
	img = tf.image.decode_jpeg(img, channels=ch)
	return tf.image.convert_image_dtype(img, tf.float32)

# Load the color and depth image from the path
def process_path(path):
	depth_path = get_depth_path(path)
	img = tf.io.read_file(path)
	img = decode_image(img, 3)
	dep = tf.io.read_file(depth_path)
	dep = decode_image(dep, 1)

	if bool(random.getrandbits(1)):
		img = tf.image.flip_left_right(img)
		dep = tf.image.flip_left_right(dep)

	return img, dep

# Prepare the dataset for using it for training
def prepare_for_training(ds, shuffle_buffer_size=1000, batch_size=64):
	ds = ds.shuffle(buffer_size=shuffle_buffer_size)		# Shuffle all the file paths
	ds = ds.map(process_path, num_parallel_calls=AUTOTUNE)	# Apply the function to load all images separately and load the images parallel on different threads
	ds = ds.repeat()										# Repeat the dataset infinately
	ds = ds.batch(batch_size)								# Divide the dataset in batches for training
	ds = ds.prefetch(buffer_size=AUTOTUNE)					# Setup prefetching so the data can be loaded in the previous step
	return ds

def Load_Dataset(path, batch_size):
	folder = pathlib.Path(path)								# Define the path to the dataset
	list_ds = tf.data.Dataset.list_files(str(folder/'*'))	# Load all filenames in the dataset
	return len(list(folder.glob("*.jpg"))), prepare_for_training(list_ds, batch_size=batch_size) # Return the length of the dataset and the dataset itself




def process_path_dummy(path, file):
	path = tf.strings.regex_replace(path, "\d*\.jpg", file)
	depth_path = get_depth_path(path)
	img = tf.io.read_file(path)
	img = decode_image(img, 3)
	dep = tf.io.read_file(depth_path)
	dep = decode_image(dep, 1)

	return img, dep

def prepare_dummy_for_training(ds, shuffle_buffer_size=1000, batch_size=64, file=None):
	ds = ds.map(lambda path: process_path_dummy(path, file), num_parallel_calls=AUTOTUNE)
	ds = ds.repeat()
	ds = ds.batch(batch_size)
	ds = ds.prefetch(buffer_size=AUTOTUNE)

	return ds

def time_it(ds, steps=1000):
	start = time.time()
	it = iter(ds)
	for i in range(steps):
		batch = next(it)
		if i%10 == 0:
			print(".", end="", flush=True)
	print()
	end = time.time()

	duration = end - start
	print("{} batches: {} s".format(steps, duration))
	print("{:0.5f} Images/s".format(BATCH*steps / duration))

def Load_Dummy_Dataset(path, file, batch_size):
	folder = pathlib.Path(path)
	list_ds = tf.data.Dataset.list_files(str(folder/'*'))
	return prepare_dummy_for_training(list_ds, batch_size=batch_size, file=file)