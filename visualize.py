import tkinter as tk
import pandas as pd
import nltk

from main import conditional_probability

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


def remove_stopwords(vocabulary):
    stop_words = set(stopwords.words('english'))
    for word in list(vocabulary.keys()):
        if word in stop_words:
            del vocabulary[word]

    return (vocabulary)


data = load_file()

d = {'spam': 0, 'ham': 1}
data['label'] = data['label'].apply(lambda x: d[x])

# remove punctuations
data['text_clean'] = data['text'].apply(lambda x: remove_punct(x.lower()))

df1 = data[data['label'] == 0]
df2 = data[data['label'] == 1]

split1 = int(1 * len(df1))

data1 = df2[:split1]

bigdata = data1.append(df1, ignore_index=True)
bigdata = bigdata.sample(frac=1, random_state=0)

split1 = int(0.9 * len(bigdata))
train_data = bigdata[:split1]
dev_data = bigdata[split1:]

train_vocabulary_opt = build_vocabulary_text_opt(train_data, 0)


H = 500
W = 600

def get_prediction():
    ps = PorterStemmer()
    wordarray= entry.get().lower().split()
    words = set(wordarray)

    removed_stopwords = [word for word in words if word not in stopwords.words('english')]
    words = [ps.stem(token) for token in removed_stopwords]

    prediction = predict(words, train_data, train_vocabulary_opt, smoothing=True)

    if prediction == 0:
        label['text'] = "SPAM MESSAGE Alert!!!! "
    else:
        label['text'] = "This message is not a spam"

root = tk.Tk()
canvas = tk.Canvas(root, height=H, width=W, bg='whitesmoke')
canvas.pack()

fra = tk.Frame(root)
fra.place(relx=0.5, rely=0.1, relwidth=0.75, relheight=0.1, anchor='n')

label_1 = tk.Label(fra, text="Spam Detector", font=('Courier', 18), bg='blueviolet', fg='white')
label_1.place(relwidth=1, relheight=1)

frame = tk.Frame(root, bg='silver', bd=5)
frame.place(relx=0.5, rely=0.25, relwidth=0.75, relheight=0.2, anchor='n')

entry=tk.Entry(frame, font=('Courier', 18))
entry.place(relwidth=1, relheight=1)

frame_2 = tk.Frame(root, bd=5)
frame_2.place(relx=0.5, rely=0.5, relwidth=0.75, relheight=0.1, anchor='n')

button = tk.Button(frame_2, text="Check Spam", font=40, command=get_prediction)
button.place(relheight=1, relwidth=0.3)

lower_frame = tk.Frame(root, bg='silver', bd=10)
lower_frame.place(relx=0.5, rely=0.65, relwidth=0.75, relheight=0.2, anchor='n')

label = tk.Label(lower_frame, font=('Courier', 18))
label.place(relwidth=1, relheight=1)

root.mainloop()
