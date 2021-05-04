import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
nltk.download('punkt')
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

import string

NEG = 0
POS = 1
ALL = 2


def load_file():
    data = pd.read_csv("spam.csv")
    data = data[['v1', 'v2']]
    data = data.rename(columns={'v1': 'label', 'v2': 'text'})
    return data


def remove_punct(text):
    #     text_nopunct = "".join([char for char in text if char not in string.punctuation])
    message_not_punc = []
    for char in text:
        if char not in string.punctuation:
            message_not_punc.append(char)

    text_nopunct = "".join(message_not_punc)

    return text_nopunct


def build_vocabulary(data, limit):
    vocabulary = {}
    for index, row in data.iterrows():
        wordarray = row[2].lower().split()
        wordarray = set(wordarray)
        for word in wordarray:
            if word.lower() not in vocabulary:
                vocabulary[word.lower()] = [0, 0, 0]

            vocabulary[word.lower()][ALL] += 1

            if row[0] == POS:
                vocabulary[word.lower()][POS] += 1
            else:
                vocabulary[word.lower()][NEG] += 1

    for word in list(vocabulary.keys()):
        if vocabulary[word][ALL] < limit:
            del vocabulary[word]

    return (vocabulary)


def build_vocabulary_text_opt(data, limit):
    vocabulary = {}
    ps = PorterStemmer()

    for index, row in data.iterrows():
        wordarray = row[2].lower().split()
        wordarray = set(wordarray)

        removed_stopwords = [word for word in wordarray if word not in stopwords.words('english')]
        wordarray = [ps.stem(token) for token in removed_stopwords]

        for word in wordarray:
            if word.lower() not in vocabulary:
                vocabulary[word.lower()] = [0, 0, 0]

            vocabulary[word.lower()][ALL] += 1

            if row[0] == POS:
                vocabulary[word.lower()][POS] += 1
            else:
                vocabulary[word.lower()][NEG] += 1

    for word in list(vocabulary.keys()):
        if vocabulary[word][ALL] < limit:
            del vocabulary[word]

    return (vocabulary)


def probability_of_the_occurrence(word, vocabulary, doc_count):
    return vocabulary[word.lower()][ALL] / doc_count


def conditional_probability(word, sentiment, vocab, train_data):
    if word.lower() not in vocab:
        return 0

    numerator = vocab[word.lower()][sentiment]

    pos_total, neg_total = num_all_docs_per_sentiment(train_data)

    if sentiment == POS:
        return numerator / pos_total
    else:
        return numerator / neg_total


def conditional_probability_with_smoothing(word, sentiment, vocab, train_data):
    numerator = 0

    if word.lower() not in vocab:
        numerator = 1
    else:
        numerator = vocab[word.lower()][sentiment] + 1

    pos_total, neg_total = num_all_docs_per_sentiment(train_data)

    if sentiment == POS:
        return numerator / (pos_total + neg_total)
    else:
        return numerator / (pos_total + neg_total)


def num_all_docs_per_sentiment(train_data):
    pos_docs = 0
    neg_docs = 0

    for index, row in train_data.iterrows():
        if row[0] == 1:
            pos_docs += 1
        else:
            neg_docs += 1

    return pos_docs, neg_docs


def probability_per_sentiment(train_data):
    pos_total, neg_total = num_all_docs_per_sentiment(train_data)
    total = len(train_data)

    return (pos_total / total), (neg_total / total)


def predict(words, train_data, train_vocab, smoothing):
    pos_prob, neg_prob = probability_per_sentiment(train_data)

    if smoothing:
        for word in words:
            pos_prob *= conditional_probability_with_smoothing(word, POS, train_vocab, train_data)
            neg_prob *= conditional_probability_with_smoothing(word, NEG, train_vocab, train_data)
    else:
        for word in words:
            pos_prob *= conditional_probability(word, POS, train_vocab, train_data)
            neg_prob *= conditional_probability(word, NEG, train_vocab, train_data)

    if pos_prob == 0:
        v_pos = 0
    else:
        v_pos = (pos_prob / (pos_prob + neg_prob))

    if neg_prob == 0:
        v_neg = 0
    else:
        v_neg = (neg_prob / (pos_prob + neg_prob))

    if v_pos > v_neg:
        return POS
    else:
        return NEG


def calculate_accuracy(test_data, train_data, train_vocab, smoothing):
    correct = 0

    for index, row in test_data.iterrows():
        wordarray = row[1].lower().split()
        words = set(wordarray)

        prediction = predict(words, train_data, train_vocab, smoothing)
        actual = row[0]

        if prediction == actual:
            correct += 1

    return correct / len(test_data)


def remove_stopwords(vocabulary):
    stop_words = set(stopwords.words('english'))
    for word in list(vocabulary.keys()):
        if word in stop_words:
            del vocabulary[word]

    return (vocabulary)


def calculate_accuracy_with_text_optimization(test_data, train_data, train_vocab, smoothing):
    correct = 0
    ps = PorterStemmer()

    for index, row in test_data.iterrows():
        wordarray = row[2].lower().split()
        words = set(wordarray)

        removed_stopwords = [word for word in words if word not in stopwords.words('english')]
        words = [ps.stem(token) for token in removed_stopwords]

        prediction = predict(words, train_data, train_vocab, smoothing)
        actual = row[0]

        if prediction == actual:
            correct += 1

    return correct / len(test_data)


if __name__ == '__main__':

    data = load_file()

    d = {'spam': 0, 'ham': 1}
    data['label'] = data['label'].apply(lambda x: d[x])

    # remove punctuations
    data['text_clean'] = data['text'].apply(lambda x: remove_punct(x.lower()))

    df1 = data[data['label'] == 0]
    df2 = data[data['label'] == 1]

    split1 = int(1 * len(df1))
    split2 = int(2 * len(df1))
    split3 = int(3 * len(df1))
    split4 = int(4 * len(df1))
    split5 = int(5 * len(df1))

    data1 = df2[:split1]
    data2 = df2[split1:split2]
    data3 = df2[split2:split3]
    data4 = df2[split3:split4]
    data5 = df2[split4:split5]

    data_break = [data1, data2, data3, data4, data5]
    accuracy_opt = []
    accuracy = []

    for data in data_break:
        bigdata = data.append(df1, ignore_index=True)
        bigdata = bigdata.sample(frac=1, random_state=0)

        split1 = int(0.9 * len(bigdata))
        train_data = bigdata[:split1]
        dev_data = bigdata[split1:]

        train_vocabulary = build_vocabulary(train_data, 0)
        train_vocabulary_opt = build_vocabulary_text_opt(train_data, 0)

        acc = calculate_accuracy(dev_data, train_data, train_vocabulary, smoothing=False)
        acc_opt = calculate_accuracy_with_text_optimization(dev_data, train_data, train_vocabulary_opt, smoothing=True)

        accuracy.append(acc)
        accuracy_opt.append(acc_opt)

    print("\n Accuracy without Optimization: ", accuracy)
    print("\n Average Accuracy without Optimization: {:.4f}".format(sum(accuracy) / len(accuracy)))
    print("\n Accuracy with Optimization: ", accuracy_opt)
    print("\n Average Accuracy with Optimization: {:.4f}".format(sum(accuracy_opt) / len(accuracy_opt)))

    accuracy.append(sum(accuracy) / len(accuracy))
    accuracy_opt.append(sum(accuracy_opt) / len(accuracy_opt))

    # Plot data in a bar chart
    plot_data = [accuracy, accuracy_opt]
    X = np.arange(6)

    plt.figure()

    plt.bar(X + 0.00, plot_data[0], color='gold', width=0.25)
    plt.bar(X + 0.25, plot_data[1], color='green', width=0.25)

    plt.ylabel('Accuracy')
    plt.xlabel('Data Sets & Average')
    plt.xticks(X, ['dataset_1', 'dataset_2', 'dataset_3', 'dataset_4', 'dataset_5', 'Average'])
    plt.title('Accuracy: Not optimized Vs. Optimized')
    plt.legend(labels=['Not_optimized', 'Optimized'], loc='upper left')
    plt.show()
