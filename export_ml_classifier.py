from helper_functions import load_images, concat_data, save_as_pickle
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

# TODO: use tempfile to load and store images from zip

def flatten(X):
	X_flattened = X.reshape(X.shape[0], 256 * 256 * 3)
	print(f"X flattened has shape {X_flattened.shape}")
	return X_flattened

def fit_classifier(X_train, y_train):
	clf = LinearSVC()
	print("Training linear support vector classifier...")
	clf.fit(X_train, y_train)
	print("Training complete!")
	return clf

def print_score(clf, X_test, y_test):
	y_pred = clf.predict(X_test)
	print(classification_report(y_test, y_pred))
	return

def export_classifier():
	fac = load_images('data/fishandchipimgs')
	not_fac = load_images('data/notfishandchipimgs')
	X, y = concat_data(fac, not_fac)
	X_flattened = flatten(X)
	X_train, X_test, y_train, y_test = train_test_split(X_flattened, y, test_size=0.2)
	clf = fit_classifier(X_train, y_train)
	print_score(clf, X_test, y_test)
	save_as_pickle(clf)
	return

if __name__ == "__main__":
	export_classifier()
	