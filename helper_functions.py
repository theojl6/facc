import cv2
import numpy as np
import os
import pickle

"""TODO: Implement random cropping instead of cv2.resize"""

def load_images(folder):
	images = []
	for filename in os.listdir(folder):
		img = cv2.imread(os.path.join(folder, filename))
		if img is not None:
			img = cv2.resize(img, (256, 256))
			img = img / 255.
			images.append(img)
	print(f"Loaded {len(images)} images from {folder}")
	return np.array(images)

def concat_data(fac, not_fac):
	X = np.concatenate((fac, not_fac), axis=0)
	y = np.zeros((X.shape[0]))
	y[:fac.shape[0]] = 1
	print(f"X has shape {X.shape}")
	print(f"y has shape {y.shape}")
	return X, y

def save_as_pickle(clf):
	if not os.path.isdir('models/'):
		os.mkdir('models/')
	pickle.dump(clf, open('models/svc_facc.pickle', 'wb'))
	print(f"Classifier successfully saved to models folder")
	return
	