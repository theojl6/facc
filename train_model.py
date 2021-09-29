import tensorflow as tf
from tensorflow import keras
from keras.callbacks import EarlyStopping
import os

AUTOTUNE = tf.data.AUTOTUNE
IMG_SIZE = 256
num_classes = 2

def load_data(data_dir):
	train_ds = keras.preprocessing.image_dataset_from_directory(
		data_dir,
		label_mode='binary',
		validation_split=0.2,
		subset="training",
		seed=123,
		)
	val_ds = keras.preprocessing.image_dataset_from_directory(
		data_dir,
		label_mode='binary',
		validation_split=0.2,
		subset="validation",
		seed=123,
		)
	return train_ds, val_ds

def prepare_data(ds, shuffle=False, augment=False):
	resize_and_rescale = keras.Sequential([
	  keras.layers.experimental.preprocessing.Resizing(IMG_SIZE, IMG_SIZE),
	  keras.layers.Rescaling(1./255)
	])
	data_augmentation = keras.Sequential([
	  keras.layers.experimental.preprocessing.RandomFlip("vertical"),
	  keras.layers.experimental.preprocessing.RandomRotation(0.2),
	])

	ds = ds.map(lambda x, y: (resize_and_rescale(x), y),
		num_parallel_calls=AUTOTUNE)

	if shuffle:
		ds = ds.shuffle(1000)

	if augment:
		ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y),
			num_parallel_calls=AUTOTUNE)

	return ds.prefetch(buffer_size=AUTOTUNE)


def build_model():
	base_model = keras.applications.ResNet50(
		weights='imagenet',
		input_shape=(IMG_SIZE, IMG_SIZE, 3),
		include_top=False)

	for layer in base_model.layers:
		layer.trainable = False

	global_avg_pooling = keras.layers.GlobalAveragePooling2D()(base_model.output)
	output = keras.layers.Dense(1, activation="sigmoid")(global_avg_pooling)

	model = keras.models.Model(inputs=base_model.input,
	                                      outputs=output)
	return model

def compile(model):
	optimizer = keras.optimizers.Adam(0.01)
	loss = keras.losses.BinaryCrossentropy()
	model.compile(
		optimizer=optimizer,
		loss=loss,
		metrics=["accuracy"])
	print(model.summary())
	return model

def fine_tune(model, train_ds, val_ds):
	early_stopping_cb = EarlyStopping(patience=5)
	model.fit(train_ds, validation_data=val_ds, epochs=10, callbacks=[early_stopping_cb])
	return model

def save(model):
	if not os.path.isdir('models/'):
		os.mkdir('models/')
	model.save('models/fine_tuned_model')
	print(f"Model successfully saved to models folder")

def export_model():
	train_ds, val_ds = load_data("data")
	model = build_model()
	model = compile(model)
	model = fine_tune(model, train_ds, val_ds)
	save(model)

if __name__ == "__main__":
	export_model()
	
