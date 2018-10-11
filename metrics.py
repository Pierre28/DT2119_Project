import numpy as np
from sklearn.metrics import f1_score, classification_report, confusion_matrix


def get_accuracy(y_true, yp):
    return np.count_nonzero(yp == y_true) / yp.shape[0]


def levenshtein(source, target):
    if len(source) < len(target):
        return levenshtein(target, source)

    # So now we have len(source) >= len(target).
    if len(target) == 0:
        return len(source)

    # We call tuple() to force strings to be used as sequences
    # ('c', 'a', 't', 's') - numpy uses them as values by default.
    source = np.array(tuple(source))
    target = np.array(tuple(target))

    # We use a dynamic programming algorithm, but with the
    # added optimization that we only need the last two rows
    # of the matrix.
    previous_row = np.arange(target.size + 1)
    for s in source:
        # Insertion (target grows longer than source):
        current_row = previous_row + 1

        # Substitution or matching:
        # Target and source items are aligned, and either
        # are different (cost of 1), or are the same (cost of 0).
        current_row[1:] = np.minimum(
            current_row[1:],
            np.add(previous_row[:-1], target != s))

        # Deletion (target grows shorter than source):
        current_row[1:] = np.minimum(
            current_row[1:],
            current_row[0:-1] + 1)

        previous_row = current_row

    return previous_row[-1]


def eval_edit_dist(y_true, yp, test_data, feature_name = 'lmfcc'):

    def merge_identical(array):
        tmp_array = np.array([array[0]])
        for i in range(1, len(array)):
            if array[i] != tmp_array[-1]:
                tmp_array = np.append(tmp_array, array[i])
        return tmp_array

    curr_index = 0
    errors = 0
    for utterance in test_data:
        len_utter = utterance[feature_name].shape[0]
        merge_state = merge_identical(yp[curr_index:curr_index + len_utter])
        test_merge_state = merge_identical(y_true[curr_index:curr_index + len_utter])
        curr_index += len_utter
        errors += levenshtein(merge_state, test_merge_state) / test_merge_state.shape[0]
    return errors / len(test_data)


def get_classification_report(y_true, yp, labels):
    return classification_report(y_true, yp, target_names=labels)


def get_f1_score(y_true, yp):
    return f1_score(y_true, yp, average='weighted')


def get_confusion_matrix(y_true, yp):
    return confusion_matrix(y_true, yp)
