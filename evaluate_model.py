import numpy as np 
from keras.models import load_model
import metrics
from sklearn.preprocessing import StandardScaler
from typing import List, Dict
import os 
from keras.utils import np_utils

class Evaluate():
	def __init__(self, model_path, params):
		self.model = load_model(model_path)
		self.context_length = params['context_length']
		self.n_layers = params['n_layers']
		self.hidden_nodes = params['hidden_nodes']
		self.as_mat = params['as_mat']
		self.feature_name = 'mspec' if params['use_mspec'] else 'lmfcc'
		self.phones = self.import_phonemes()
		self.feature_import = self.feature_name
		if not params['use_mspec'] :
			self.feature_import = ''
		self.train, self.test = self.__load_data(self.feature_import)
		self.use_dynamic_features = params['use_dynamic_features']
		self.feature_func = self.__dynamic_features if self.use_dynamic_features else self.__regular_features
		self.x_train, self.y_train = self.feature_func(self.train)       
		self.x_test, self.y_test = self.feature_func(self.test)

		self.scaler = StandardScaler()
		self.x_train = self.scaler.fit_transform(self.x_train)
		self.x_test = self.scaler.transform(self.x_test)
		
		if self.as_mat:
			self.x_train = self.x_train.reshape((self.x_train.shape[0], self.context_length, 13))
			self.x_test = self.x_test.reshape((self.x_test.shape[0], self.context_length, 13))

		self.x_test = np.expand_dims(self.x_test, 3)
		self.y_val_hat = self.model.predict(self.x_test)
		self.pred_labels = np.argmax(self.y_val_hat, axis=1)
		self.test_labels = np.argmax(self.y_test, axis=1)

	@staticmethod
	def __load_data(feature) -> List[Dict]:
		return [np.load(os.path.join('dataset','traindata_'+ str(feature) +'.npz'))['data'],
		        np.load(os.path.join('dataset', 'testdata_'+ str(feature) + '.npz'))['data']]
    
	@staticmethod
	def import_phonemes():
	    return sorted([x.strip() for x in open('phonemeList.txt').readlines()])

	def __regular_features(self, data):
		x = np.concatenate([x[self.feature_name] for x in data])
		y = np_utils.to_categorical(np.concatenate([d['targets'] for d in data]))
		return x, y

	def __dynamic_features(self, data):
		X = []
		Y = []
		half = self.context_length // 2
		for d in data:
			m = d[self.feature_name]
			N = len(m)
			for i, _ in enumerate(m):
				if i < half:
					res = np.array([m[abs(k)] for k in range(-half, half + 1)])
				elif i >= N - half:
					res = np.array([m[k] if k < N else m[N - (k - N) - 2] for k in range(i-half, i+half+1)])
				else:
					res = np.array(m[i - half:i + half + 1])
				X.append(np.concatenate(res))
				Y.append(self.phones.index(d['target'][i]))

		return np.array(X).astype('float32'), np_utils.to_categorical(Y)



	def get_accuracy(self):
		return metrics.get_accuracy(self.test_labels, self.pred_labels)

	def eval_edit_dist(self):
		return metrics.eval_edit_dist(self.test_labels, self.pred_labels, self.test, self.feature_name)


	def get_classification_report(self):
		return metrics.get_classification_report(self.test_labels, self.pred_labels, self.phones)

	def get_f1_score(self):
		return metrics.get_f1_score(self.test_labels, self.pred_labels)

	def get_confusion_matrix(self):
		return metrics.get_confusion_matrix(self.test_labels, self.pred_labels)




