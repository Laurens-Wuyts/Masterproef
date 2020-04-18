from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.models import Model

from tensorflow.keras.applications import MobileNet

class FastDepthNet:
	@staticmethod
	def build():
		# Encoder: MobileNet (feature extractor)
		mobNet = MobileNet(
			input_shape=(224, 224, 3),	# Use 224 by 224 images with 3 channels (RGB)
			alpha=1.0,
			depth_multiplier=1,
			dropout=1e-3,
			include_top=False,			# Remove the last classifier
			weights='imagenet',			# Pretrained on ImageNet
			input_tensor=None,
			pooling=None)


		decIn = mobNet.layers[-1].output
		
		# Decoder
		# Upsample 1
		conv1Out = Conv2D(512, (5,5), padding="same")(decIn)
		up1Out   = UpSampling2D(size=(2,2), interpolation="nearest")(conv1Out)
		# Upsample 2
		conv2Out = Conv2D(256, (5,5), padding="same")(up1Out)
		up2Out   = UpSampling2D(size=(2,2), interpolation="nearest")(conv2Out)
		# Skip connection 1
		skip1    = mobNet.get_layer("conv_pw_5_relu").output
		skip1Out = Add()([up2Out, skip1])
		# Upsample 3		
		conv3Out = Conv2D(128, (5,5), padding="same")(skip1Out)
		up3Out   = UpSampling2D(size=(2,2), interpolation="nearest")(conv3Out)
		# Skip connection 2
		skip2    = mobNet.get_layer("conv_pw_3_relu").output
		skip2Out = Add()([up3Out, skip2])
		# Upsample 4
		conv4Out = Conv2D(64, (5,5), padding="same")(skip2Out)
		up4Out   = UpSampling2D(size=(2,2), interpolation="nearest")(conv4Out)
		# Skip connection 3
		skip3    = mobNet.get_layer("conv_pw_1_relu").output
		skip3Out = Add()([up4Out, skip3])
		# Upsample 5
		conv5Out = Conv2D(32, (5,5), padding="same")(skip3Out)
		up5Out   = UpSampling2D(size=(2,2), interpolation="nearest")(conv5Out)
		# Pointwise conv
		decOut   = Conv2D(1, (1,1), padding="same")(up5Out)

		# Combine full model
		model = Model(inputs=mobNet.input, outputs=decOut)
		return model