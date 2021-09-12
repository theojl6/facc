from helper_functions import load_images, concat_data
import tensorflow as tf
from tensorflow import keras
from keras.callbacks import EarlyStopping
import os

def build_model():
	base_model = keras.applications.Xception(
		weights='imagenet',
		input_shape=(256, 256, 3),
		include_top=False)
	inputs = keras.Input(shape=(256, 256, 3))
	x = base_model(inputs, training=False)
	x = keras.layers.GlobalAveragePooling2D()(x)
	outputs = keras.layers.Dense(1)(x)
	model = keras.Model(inputs, outputs)
	return model

def compile(model):
	optimizer = keras.optimizers.Adam(1e-5)
	loss = keras.losses.BinaryCrossentropy(from_logits=True)
	binary_accuracy = keras.metrics.BinaryAccuracy()
	model.compile(
		optimizer=optimizer,
		loss=loss,
		metrics=[binary_accuracy])
	print(model.summary())
	return model

def fine_tune(model, X, y):
	early_stopping_cb = EarlyStopping(patience=5)
	model.fit(X, y, epochs=10, callbacks=[early_stopping_cb], validation_split=0.2)
	return model

def save(model):
	if not os.path.isdir('models/'):
		os.mkdir('models/')
	model.save('models/fine_tuned_xception_facc')
	print(f"Model successfully saved to models folder")
	return

def export_model():
	fac = load_images('data/fishandchipimgs')
	not_fac = load_images('data/notfishandchipimgs')
	X, y = concat_data(fac, not_fac)
	model = build_model()
	model = compile(model)
	model = fine_tune(model, X, y)
	save(model)
	return

if __name__ == "__main__":
	export_model()
	
