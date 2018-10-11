from ffnn import FFNN
from rnn import RNN
from cnn import CNN
from cldnn import CLDNN
from metrics import get_accuracy, f1_score, get_classification_report, get_confusion_matrix, eval_edit_dist
from keras.models import load_model
from features_to_tsne import plot_features
from metrics import get_f1_score, classification_report, get_accuracy
from evaluate_model import Evaluate
import os 


def fixed_phones(net):
    tmp = list(net.phones_reduced)
#    tmp.remove('2')
#    tmp.remove('1')
#    tmp.remove('q')
    return tmp


def store_results(y_true, yp, net, model_path, model):
    phones = fixed_phones(net)
    with open(os.path.join(os.getcwd(), model_path + os.sep + 'results.txt'), 'w') as f:
        f.write('acc: {}\n'.format(str(get_accuracy(y_true, yp))))
        f.write('edit: {}\n'.format(str(eval_edit_dist(y_true, yp, net.test, feature_name=net.feature_name))))
        f.write('f1-score: {}\n'.format(str(get_f1_score(y_true, yp))))
        report = get_classification_report(y_true, yp, phones)
        f.write(str(report))
    cm = get_confusion_matrix(y_true, yp)
    net.plot_confusion_matrix(cm, phones, os.path.join(os.getcwd(), model_path + os.sep + 'confusion_matrix.png'), normalize=True)
    model.save(os.path.join(os.getcwd(), model_path + os.sep + 'model.h5'))


def test_ffnn():
    params = {'n_layers': 4, 'hidden_nodes': [512, 512, 512, 512],
              'epochs': 10, 'use_dynamic_features': True,
              'use_mspec': False, 'as_mat': False,
              'speaker_norm': False,
              'context_length': 17}
    net = FFNN(params)
    model = net.train_model()
    net.set_model(model)
    y_true, yp = net.predict_on_test()
    print("FFNN RESULTS")
    print(get_f1_score(y_true, yp))
    print(get_accuracy(y_true, yp))
    print(classification_report(y_true, yp))


def test_rnn():
    """Notice as_mat is true here!"""
    params = {'n_layers': 2, 'hidden_nodes': [32, 32],
              'epochs': 100, 'use_dynamic_features': True,
              'use_mspec': True, 'as_mat': True,
              'speaker_norm': False,
              'context_length': 35}
    net = RNN(params)
    model = net.train_model(params['unroll'])
    net.set_model(model)
    y_true, yp = net.predict_on_test()
    print("RNN RESULTS")
    print(get_f1_score(y_true, yp))
    print(get_accuracy(y_true, yp))
    print(classification_report(y_true, yp))
    model.save('rnn-64-64-context-35.h5')


def test_cnn(params):
    net = CNN(params)
    model, model_path = net.train_model(kernel_sizes=params['kernel_sizes'], strides=params['strides'])
    net.set_model(model)
    y_true, yp = net.predict_on_test()
    store_results(y_true, yp, net, model_path, model)


def test_cldnn(params):
    net = CLDNN(params)
    model, model_path = net.train_model(params['conv_hidden'], params['lstm_hidden'],
                            params['conv_kernels'], params['dense_hidden'], params['bidirectional'])
    net.set_model(model)
    y_true, yp = net.predict_on_test()
    store_results(y_true, yp, net, model_path, model)



if __name__ == "__main__":
    #    CNN setup
    # ap = []
    # ap.append({'n_layers': 2, 'hidden_nodes': [32, 32], 'epochs': 1000, 'use_dynamic_features': True, 'use_mspec': True,
    #           'as_mat': True, 'speaker_norm': False, 'context_length': 13, 'kernel_sizes': [(3, 3), (3, 3)],
    #            'strides': [(1, 1), (1, 1)]})
    #
    # ap.append({'n_layers': 2, 'hidden_nodes': [64, 64], 'epochs': 1000, 'use_dynamic_features': True, 'use_mspec': True,
    #           'as_mat': True, 'speaker_norm': False, 'context_length': 13, 'kernel_sizes': [(3, 3), (3, 3)],
    #            'strides': [(1, 1), (1, 1)]})
    #
    # ap.append({'n_layers': 2, 'hidden_nodes': [32, 32], 'epochs': 1000, 'use_dynamic_features': True, 'use_mspec': True,
    #           'as_mat': True, 'speaker_norm': False, 'context_length': 31, 'kernel_sizes': [(3, 3), (3, 3)],
    #            'strides': [(1, 1), (1, 1)]})
    #
    # ap.append({'n_layers': 2, 'hidden_nodes': [32, 32], 'epochs': 1000, 'use_dynamic_features': True, 'use_mspec': True,
    #           'as_mat': True, 'speaker_norm': False, 'context_length': 31, 'kernel_sizes': [(3, 3), (3, 3)],
    #            'strides': [(2, 2), (2, 2)]})
    #
    # for i, params in enumerate(ap):
    #     print("training model: {}".format(i))
    #     test_cnn(params)

    # LSTM settings
    # ap = []
    # ap.append({'n_layers': 2, 'hidden_nodes': [32, 32], 'epochs': 1000, 'use_dynamic_features': True, 'use_mspec': True,
    #            'as_mat': True, 'speaker_norm': False, 'context_length': 13, 'unroll': False})
    #
    # ap.append({'n_layers': 2, 'hidden_nodes': [32, 32], 'epochs': 1000, 'use_dynamic_features': True, 'use_mspec': True,
    #            'as_mat': True, 'speaker_norm': False, 'context_length': 31, 'unroll': False})
    #
    # ap.append({'n_layers': 2, 'hidden_nodes': [32, 32], 'epochs': 1000, 'use_dynamic_features': True, 'use_mspec': True,
    #            'as_mat': True, 'speaker_norm': False, 'context_length': 13, 'unroll': True})
    #
    # ap.append({'n_layers': 2, 'hidden_nodes': [32, 32], 'epochs': 1000, 'use_dynamic_features': True, 'use_mspec': True,
    #            'as_mat': True, 'speaker_norm': False, 'context_length': 31, 'unroll': True})
    #
    # for i, params in enumerate(ap):
    #     print("training model: {}".format(i))
    #     test_rnn(params)

    #  CLDN setup
    ap = []
    ap.append({'lstm_hidden': [32, 32], 'dense_hidden': [32, 32], 'conv_hidden': [32, 32], 'conv_kernels': [3, 3],
               'epochs': 200, 'use_dynamic_features': True, 'use_mspec': True, 'bidirectional': True,
              'as_mat': True, 'speaker_norm': False, 'context_length': 31, 'n_layers': None, 'hidden_nodes': None})

    ap.append({'lstm_hidden': [32, 32], 'dense_hidden': [32, 32], 'conv_hidden': [32, 32], 'conv_kernels': [3, 3],
               'epochs': 150, 'use_dynamic_features': True, 'use_mspec': True, 'bidirectional': False,
              'as_mat': True, 'speaker_norm': False, 'context_length': 31, 'n_layers': None, 'hidden_nodes': None})

    # ap.append({'lstm_hidden': [32, 32], 'dense_hidden': [32, 32], 'conv_hidden': [32, 32], 'conv_kernels': [9, 4],
    #            'epochs': 150, 'use_dynamic_features': True, 'use_mspec': True,
    #           'as_mat': True, 'speaker_norm': False, 'context_length': 13, 'n_layers': None, 'hidden_nodes': None})
    #
    # ap.append({'lstm_hidden': [32, 32], 'dense_hidden': [32, 32], 'conv_hidden': [32, 32], 'conv_kernels': [9, 9],
    #            'epochs': 150, 'use_dynamic_features': True, 'use_mspec': True,
    #           'as_mat': True, 'speaker_norm': False, 'context_length': 13, 'n_layers': None, 'hidden_nodes': None})

    for params in ap:
        test_cldnn(params)



