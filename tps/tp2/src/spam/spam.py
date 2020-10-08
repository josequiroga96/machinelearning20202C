import numpy as np
import tps.tp2.src.spam.util as util
import tps.tp2.src.spam.svm as svm


def get_words(message):
    """Get the normalized list of words from a message string.

    This function should split a message into words, normalize them, and return
    the resulting list. For simplicity, you should split on whitespace, not
    punctuation or any other character. For normalization, you should convert
    everything to lowercase.  Please do not consider the empty string (" ") to be a word.

    Args:
        message: A string containing an SMS message

    Returns:
       The list of normalized words from the message.
    """

    words = message.lower().split(" ")

    while words.count(""):
        words.remove("")

    return words

def create_dictionary(messages):
    """Create a dictionary mapping words to integer indices.

    This function should create a dictionary of word to indices using the provided
    training messages. Use get_words to process each message.

    Rare words are often not useful for modeling. Please only add words to the dictionary
    if they occur in at least five messages.

    Args:
        messages: A list of strings containing SMS messages

    Returns:
        A python dict mapping words to integers.
    """
    dictionary = {}
    failing_words = []
    word_lists = []
    index = 0

    for message in messages:
        word_lists.append(get_words(message))

    if len(word_lists) >= 5:
        for i in range(len(word_lists) - 4):
            for word in word_lists[i]:
                if not (word in dictionary) and not word in failing_words:
                    count = 1
                    for j in range(i + 1, len(word_lists)):
                        if word in word_lists[j]:
                            count += 1
                    if count >= 5:
                        dictionary[word] = index
                        index += 1
                    else:
                        failing_words.append(word)

    return dictionary


def transform_text(messages, word_dictionary):
    """Transform a list of text messages into a numpy array for further processing.

    This function should create a numpy array that contains the number of times each word
    of the vocabulary appears in each message.
    Each row in the resulting array should correspond to each message
    and each column should correspond to a word of the vocabulary.

    Use the provided word dictionary to map words to column indices. Ignore words that
    are not present in the dictionary. Use get_words to get the words for a message.

    Args:
        messages: A list of strings where each string is an SMS message.
        word_dictionary: A python dict mapping words to integers.

    Returns:
        A numpy array marking the words present in each message.
        Where the component (i,j) is the number of occurrences of the
        j-th vocabulary word in the i-th message.
    """

    arr = np.zeros((len(messages), len(word_dictionary)))

    for i in range(len(messages)):
        for word in get_words(messages[i]):
            if word in word_dictionary:
                arr[i, word_dictionary[word]] += 1

    return arr


def fit_naive_bayes_model(matrix, labels):
    """Fit a naive bayes model.

    This function should fit a Naive Bayes model given a training matrix and labels.

    The function should return the state of that model.

    Feel free to use whatever datatype you wish for the state of the model.

    Args:
        matrix: A numpy array containing word counts for the training data
        labels: The binary (0 or 1) labels for that training data

    Returns: The trained model
    """

    model = {}
    average = sum(labels) / len(labels)
    value_0 = (matrix[labels == 0]).sum(axis=0) + 1
    value_1 = (matrix[labels == 1]).sum(axis=0) + 1

    model['log_phi_0'] = np.log(1 - average)
    model['log_phi_1'] = np.log(average)
    model['log_theta_0'] = np.log(value_0/value_0.sum())
    model['log_theta_1'] = np.log(value_1/value_1.sum())

    return model


def predict_from_naive_bayes_model(model, matrix):
    """Use a Naive Bayes model to compute predictions for a target matrix.

    This function should be able to predict on the models that fit_naive_bayes_model
    outputs.

    Args:
        model: A trained model from fit_naive_bayes_model
        matrix: A numpy array containing word counts

    Returns: A numpy array containg the predictions from the model
    """

    log_phi_0 = model['log_phi_0']
    log_phi_1 = model['log_phi_1']
    log_theta_0 = model['log_theta_0']
    log_theta_1 = model['log_theta_1']
    log_probs_0 = (matrix * log_theta_0).sum(axis=1) + log_phi_0
    log_probs_1 = (matrix * log_theta_1).sum(axis=1) + log_phi_1

    matrix_greater = log_probs_1 > log_probs_0
    output = matrix_greater.astype(int)

    return output


def get_top_five_naive_bayes_words(model, dictionary):
    """Compute the top five words that are most indicative of the spam (i.e positive) class.

    Ues the metric given in part-c as a measure of how indicative a word is.
    Return the words in sorted form, with the most indicative word first.

    Args:
        model: The Naive Bayes model returned from fit_naive_bayes_model
        dictionary: A mapping of word to integer ids

    Returns: A list of the top five most indicative words in sorted order with the most indicative first
    """
    ids = np.argsort(model['log_theta_0'] - model['log_theta_1'])[:5]
    reverse_dictionary = {i: word for word, i in dictionary.items()}
    return [reverse_dictionary[i] for i in ids]


def compute_best_svm_radius(train_matrix, train_labels, val_matrix, val_labels, radius_to_consider):
    """Compute the optimal SVM radius using the provided training and evaluation datasets.

    You should only consider radius values within the radius_to_consider list.
    You should use accuracy as a metric for comparing the different radius values.

    Args:
        train_matrix: The word counts for the training data
        train_labels: The spma or not spam labels for the training data
        val_matrix: The word counts for the validation data
        val_labels: The spam or not spam labels for the validation data
        radius_to_consider: The radius values to consider

    Returns:
        The best radius which maximizes SVM accuracy.
    """
    best_radius = None
    for radius in radius_to_consider:
        svm_predictions = svm.train_and_predict_svm(train_matrix, train_labels, val_matrix, radius)
        svm_accuracy = np.mean(svm_predictions == val_labels)
        if best_radius is None:
            best_radius = (svm_accuracy, radius)
        else:
            best_radius = max(best_radius, (svm_accuracy, radius))

    return best_radius[1]


def main():
    train_messages, train_labels = util.load_spam_dataset('spam_train.tsv')
    val_messages, val_labels = util.load_spam_dataset('spam_val.tsv')
    test_messages, test_labels = util.load_spam_dataset('spam_test.tsv')

    dictionary = create_dictionary(train_messages)

    print('Size of dictionary: ', len(dictionary))

    util.write_json('spam_dictionary', dictionary)

    train_matrix = transform_text(train_messages, dictionary)

    np.savetxt('spam_sample_train_matrix', train_matrix[:100,:])

    val_matrix = transform_text(val_messages, dictionary)
    test_matrix = transform_text(test_messages, dictionary)

    naive_bayes_model = fit_naive_bayes_model(train_matrix, train_labels)

    naive_bayes_predictions = predict_from_naive_bayes_model(naive_bayes_model, test_matrix)

    np.savetxt('spam_naive_bayes_predictions', naive_bayes_predictions)

    naive_bayes_accuracy = np.mean(naive_bayes_predictions == test_labels)

    print('Naive Bayes had an accuracy of {} on the testing set'.format(naive_bayes_accuracy))

    top_5_words = get_top_five_naive_bayes_words(naive_bayes_model, dictionary)

    print('The top 5 indicative words for Naive Bayes are: ', top_5_words)

    util.write_json('spam_top_indicative_words', top_5_words)

    optimal_radius = compute_best_svm_radius(train_matrix, train_labels, val_matrix, val_labels, [0.01, 0.1, 1, 10])

    util.write_json('spam_optimal_radius', optimal_radius)

    print('The optimal SVM radius was {}'.format(optimal_radius))

    svm_predictions = svm.train_and_predict_svm(train_matrix, train_labels, test_matrix, optimal_radius)

    svm_accuracy = np.mean(svm_predictions == test_labels)

    print('The SVM model had an accuracy of {} on the testing set'.format(svm_accuracy, optimal_radius))


if __name__ == "__main__":
    main()
