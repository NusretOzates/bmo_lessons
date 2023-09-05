from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List
from collections import Counter
import string
import numpy as np
from tqdm import tqdm

sentences = [
    "This is a foo bar sentence.",
    "This sentence is similar to a foo bar sentence.",
    "This is another string, but it is not quite similar to the previous ones.",
]

tfidf = TfidfVectorizer()
X = tfidf.fit_transform(sentences)
print(X.toarray()[1,:])


class CustomTFIDF:
    def __init__(self):
        self.sentences = []
        self.doc_count = 0
        self.df = Counter()
        self.idf = {}
        self.vocabulary = {}

    def fit(self, sentences: List[str]):
        self.doc_count = len(sentences)
        self.sentences = [self.__preprocess_sentence(sentence) for sentence in tqdm(sentences)]
        for sentence in tqdm(self.sentences):
            self.df.update(set(sentence))

        df_values = np.array(list(self.df.values()), dtype=np.float64) + 1.0
        idf_values = np.log((self.doc_count + 1) / df_values) + 1

        self.idf = {
            key: idf_values[index]
            for index, key in tqdm(enumerate(self.df.keys()))
        }
        self.vocabulary = {key: i for i, key in tqdm(enumerate(self.idf.keys()))}

    def transform_train(self):
        transformed = np.zeros((self.doc_count, len(self.vocabulary)))

        # Calculate TF for every sentence
        for i, sentence in enumerate(self.sentences):
            word_counts = self.calculate_tf(sentence)
            for word, value in word_counts.items():
                if word not in self.vocabulary:
                    continue
                transformed[i][self.vocabulary[word]] = value * self.idf[word]

        # L2 normalize
        for i, row in enumerate(transformed):
            transformed[i] = transformed[i] / np.sqrt(
                np.sum(transformed[i] ** 2)
            )

        return transformed

    def transform(self, sentence: str):
        transformed = np.zeros((len(self.vocabulary)))
        words = self.__preprocess_sentence(sentence)
        word_counts = self.calculate_tf(words)
        for word, value in word_counts.items():
            if word not in self.vocabulary:
                continue
            transformed[self.vocabulary[word]] = value * self.idf[word]

        return transformed

    def fit_transform(self, sentences: List[str]):
        self.fit(sentences)
        return self.transform_train()

    def calculate_tf(self, words: List[str]):
        word_counts = Counter(words)
        # word_counts = {word: value/len(sentence) for word, value in word_counts.items()} Uses true Term-Frequency function
        return word_counts

    def __remove_punctuations(self, sentence: str):
        translate_table = str.maketrans("", "", string.punctuation)
        sentence = sentence.translate(translate_table)
        return sentence

    def __preprocess_sentence(self, sentence: str):
        sentence = sentence.lower()
        sentence = self.__remove_punctuations(sentence)
        sentence = sentence.split()
        sentence = filter(lambda x: len(x) >= 2, sentence)

        return list(sentence)


x = CustomTFIDF()
print(x.fit_transform(sentences)[1])

print(np.sum(X.toarray()[1]))
print(np.sum(x.fit_transform(sentences)[1]))

from sklearn.datasets import fetch_20newsgroups

data, target = fetch_20newsgroups(return_X_y=True,subset='train')
test_data, test_target = fetch_20newsgroups(return_X_y=True,subset='test')

x = CustomTFIDF()
data = x.fit_transform(data)

from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier(max_depth=50)
classifier.fit(data,target)

predictions = classifier.predict([x.transform(data)for data in tqdm(test_data)])

from sklearn.metrics import classification_report

print(classification_report(test_target,predictions))