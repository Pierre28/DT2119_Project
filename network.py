import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, f1_score
from keras.utils import np_utils
from keras import Sequential
from keras.models import load_model
from typing import List, Dict
import matplotlib.pyplot as plt


class Network:
    def __init__(self, params):
        self.model = None
        self.train_loss_history = []
        self.train_acc_history = []
        self.context_length = params['context_length']
        self.n_layers = params['n_layers']
        self.hidden_nodes = params['hidden_nodes']
        self.epochs = params['epochs']
        self.feature_name = 'mspec' if params['use_mspec'] else 'lmfcc'
        self.as_mat = params['as_mat']
        self.phones = self.import_phonemes()
        self.phones.remove('2')
        self.phones.remove('1')
        self.phones.remove('q')
        self.phones_reduced = self.reduce_phones(self.phones)
        self.train, self.test, self.val = self.__load_data()
        if params['speaker_norm']:
            self.train = self.speaker_normalisation(self.train)
            self.test = self.speaker_normalisation(self.test)
            self.val = self.speaker_normalisation(self.val)
        self.use_dynamic_features = params['use_dynamic_features']
        self.feature_func = self.__dynamic_features if self.use_dynamic_features else self.__regular_features
        self.x_train, self.y_train = self.feature_func(self.train)
        print(self.y_train.shape)
        self.x_test, self.y_test = self.feature_func(self.test)
        self.x_val, self.y_val = self.feature_func(self.val)
        self.scaler = StandardScaler()
        self.x_train = self.scaler.fit_transform(self.x_train)
        self.x_test = self.scaler.transform(self.x_test)
        self.x_val = self.scaler.transform(self.x_val)
        if self.as_mat:
            """ 13 is number of MFCC features. Don't love the magic number."""
            n_feats = 40 if self.feature_name == 'mspec' else 13
            self.x_train = self.x_train.reshape((self.x_train.shape[0], self.context_length, n_feats))
            self.x_test = self.x_test.reshape((self.x_test.shape[0], self.context_length, n_feats))
            self.x_val = self.x_val.reshape((self.x_val.shape[0], self.context_length, n_feats))

    @staticmethod
    def import_phonemes():
        return [x.strip() for x in open('phonemeList.txt').readlines()]

    @staticmethod
    def reduce_phones(phones):
        mapping_dict = {'aa': ['aa', 'ao'], 'ah': ['ah', 'ax', 'ax-h'], 'er': ['er', 'axr'], 'hh': ['hh', 'hv'],
                        'ih': ['ih', 'ix'], 'l': ['l', 'el'], 'm': ['m', 'em'], 'n': ['n', 'en', 'nx'],
                        'ng': ['ng', 'eng'], 'sh': ['sh', 'zh'], 'uw': ['uw', 'ux'],
                        'sil': ['pcl', 'tcl', 'kcl', 'tck', 'bcl', 'dcl', 'gcl', 'h#', 'pau', 'epi']}
        tmp_phones = list(phones)
        tmp_phones.append('sil')
        for phone_index in range(len(phones)):
            for key, val in mapping_dict.items():
                if phones[phone_index] in val:
                    if phones[phone_index] != key:
                        tmp_phones.remove(phones[phone_index])
        return tmp_phones

    @staticmethod
    def phone_to_reduced_phone(target):
        mapping_dict = {'aa': ['aa', 'ao'], 'ah': ['ah', 'ax', 'ax-h'], 'er': ['er', 'axr'], 'hh': ['hh', 'hv'],
                        'ih': ['ih', 'ix'], 'l': ['l', 'el'], 'm': ['m', 'em'], 'n': ['n', 'en', 'nx'],
                        'ng': ['ng', 'eng'], 'sh': ['sh', 'zh'], 'uw': ['uw', 'ux'],
                        'sil': ['pcl', 'tcl', 'kcl', 'tck', 'bcl', 'dcl', 'gcl', 'h#', 'pau', 'epi']}
        output_target = target
        for key, val in mapping_dict.items():
            if target in val:
                output_target = key
        return output_target

    def __load_data(self) -> List[Dict]:
        if self.feature_name == 'mspec':
            return [np.load('dataset/traindata_mspec.npz')['data'],
                    np.load('dataset/testdata_mspec.npz')['data'],
                    np.load('dataset/valdata_mspec.npz')['data']
                    ]
        return [np.load('dataset/traindata_.npz')['data'],
                np.load('dataset/testdata_.npz')['data'],
                np.load('dataset/valdata_.npz')['data']]

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
                    res = np.array([m[k] if k < N else m[N - (k - N) - 2] for k in range(i - half, i + half + 1)])
                else:
                    res = np.array(m[i - half:i + half + 1])
                X.append(np.concatenate(res))
                Y.append(self.phones_reduced.index(self.phone_to_reduced_phone(d['target'][i])))
        return np.array(X).astype('float32'), np_utils.to_categorical(Y)

    def speaker_normalisation(self, data):
        self.speaker_norm = {}
        current_speaker = ''
        # Compute means
        for utterance in data:
            speaker_id = utterance['filename'].split('/')[-2]
            if speaker_id != current_speaker:
                self.speaker_norm[speaker_id] = [np.zeros(13), np.zeros(13), 0]  # mean, std, count
                self.speaker_norm[speaker_id][0] += np.mean(utterance['lmfcc'], axis=0)
                self.speaker_norm[speaker_id][2] += 1
                current_speaker = speaker_id
            if speaker_id == current_speaker:
                self.speaker_norm[speaker_id][0] += np.mean(utterance['lmfcc'], axis=0)
                self.speaker_norm[speaker_id][2] += 1

        # divide mean by number of utterance
        for speaker, val in self.speaker_norm.items():
            self.speaker_norm[speaker][0] = val[0] / val[2]

        # Compute std
        for utterance in data:
            speaker_id = utterance['filename'].split('/')[-2]
            self.speaker_norm[speaker_id][1] += np.square(utterance['lmfcc'] - self.speaker_norm[speaker_id][0]).sum(
                axis=0)

        # divide std by number of utterance
        for speaker, val in self.speaker_norm.items():
            self.speaker_norm[speaker][1] = np.sqrt(val[1]) / val[2]

        # Update for training
        for utterance in data:
            speaker_id = utterance['filename'].split('/')[-2]
            utterance['lmfcc'] = (utterance['lmfcc'] - self.speaker_norm[speaker_id][0]) / \
                                 self.speaker_norm[speaker_id][1]

        return data

    def params_to_folder(self) -> str:
        folder_name = f"{str(self.n_layers)}_{'-'.join([str(i) for i in self.hidden_nodes])}" \
                      f"_{self.feature_name}_context_length_{self.context_length}_"
        if not os.path.exists(os.path.join(os.getcwd(), folder_name)):
            os.mkdir(folder_name)
        return folder_name

    def params_to_file(self, model: Sequential):
        with open(self.params_to_folder() + os.sep + 'report.txt', 'w') as fh:
            model.summary(print_fn=lambda x: fh.write(x + '\n'))

    def plot_confusion_matrix(self, cm, classes, path, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
        np.set_printoptions(suppress=True)  # removes scientific notation when saving to files
        plt.rcParams['mathtext.fontset'] = 'stix'
        plt.rcParams['font.family'] = 'STIXGeneral'
        plt.rcParams['savefig.dpi'] = 500
        plt.style.use('ggplot')
        plt.rcParams["errorbar.capsize"] = 3

        title = 'Normalized Confusion Matrix' if normalize else 'confusion'
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        plt.figure(figsize=(9,9))
        plt.clf()
        # np.savetxt(self.params_to_folder() + os.sep + title + '.csv', cm, delimiter=",", fmt='%f')
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
       # plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig(path)

    def store_test_acc(self, acc, f1):
        with open(self.params_to_folder() + os.sep + 'test_acc.txt', 'w') as f:
            f.write(f"Accuracy: {str(acc)}'\n'")
            f.write(f"F1 score: {str(f1)}")

    def store_loss_history(self, train, val):
        plt.figure()
        plt.title(f"Training and validation loss")
        plt.plot(train, 'g', label="Train")
        plt.plot(val, 'r', label="Validation")
        plt.legend()
        plt.ylabel("Loss")
        plt.xlabel("Number of epochs")
        plt.savefig(f"{self.params_to_folder() + os.sep}train_val_loss.png")

    def store_acc_history(self, train, val):
        plt.figure()
        plt.title(f"Training and validation acc")
        plt.plot(train, 'g', label="Train")
        plt.plot(val, 'r', label="Validation")
        plt.legend()
        plt.ylabel("Acc")
        plt.xlabel("Number of epochs")
        plt.savefig(f"{self.params_to_folder() + os.sep}train_val_acc.png")

    def set_model(self, model):
        self.model = model

    def set_model_by_path(self, path):
        self.model = load_model(path)

    def save_model(self, path):
        self.model.save(path)

    def predict_on_test(self, return_posterios=False):
        y_val_hat = self.model.predict(self.x_test)
        pred_labels = np.argmax(y_val_hat, axis=1)
        true_labels = np.argmax(self.y_test, axis=1)
        if return_posterios:
            return true_labels, pred_labels, y_val_hat
        return true_labels, pred_labels
