import tensorflow as tf
import numpy 	  as np
import h5py

def _bytes_feature(val):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[val]))


tfrec_fn = "dataset_test.tfrecord"
hdf5_fn = "/home/laurens/masterproef/Data/Realsense_Dataset/Realsense_Dataset_preprocessed_comp.h5"

writer = tf.io.TFRecordWriter(tfrec_fn)

with h5py.File(hdf5_fn, 'r') as f:
	i = np.array(f["trainX"])
	d = np.array(f["trainY"])

	for idx in range(i.shape[0]):

		feature = { 
			'image': _bytes_feature(i[idx].tostring()),
			'depth': _bytes_feature(d[idx].tostring())
		}

		example = tf.train.Example(features=tf.train.Features(feature=feature)) 
		writer.write(example.SerializeToString())

writer.close()