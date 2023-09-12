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
print(X.toarray()[1])


class CustomTFIDF:
    def __init__(self):
        self.documents = []
        self.doc_count = 0
        self.df = Counter()
        self.idf = {}
        self.vocabulary = {}

    def fit(self, documents: List[str]):
        """Fits the TF-IDF Algorithm
        It will use the implementation from scikit-learn. Steps:

        - Lowercase the sentences, remove punctuations and split into words
        - For every word, count how many documents they are in. This is DF.
        - Calculate IDF by dividing doc_count by DF of the word and get the log of it
        - Create a vocabulary using the idf keys

        Args:
            documents: List of documents.

        Returns:

        """
        self.doc_count = len(documents)
        self.documents = [
            self.__preprocess_sentence(sentence) for sentence in tqdm(documents)
        ]
        for document in tqdm(self.documents):
            self.df.update(set(document))

        df_values = np.array(list(self.df.values()), dtype=np.float64) + 1.0
        idf_values = np.log((self.doc_count + 1) / df_values) + 1

        self.idf = {
            key: idf_values[index] for index, key in tqdm(enumerate(self.df.keys()))
        }
        self.vocabulary = {key: i for i, key in tqdm(enumerate(self.idf.keys()))}

    def transform_train(self) -> np.ndarray:
        """Transform the train documents

        Returns:
            Transformed document vectors

        """
        transformed = np.zeros((self.doc_count, len(self.vocabulary)))

        # Calculate TF for every sentence
        for i, sentence in enumerate(self.documents):
            word_counts = self.calculate_tf(sentence)
            for word, value in word_counts.items():
                if word not in self.vocabulary:
                    continue
                transformed[i][self.vocabulary[word]] = value * self.idf[word]

        # L2 normalize
        for i, row in enumerate(transformed):
            transformed[i] = transformed[i] / np.sqrt(np.sum(transformed[i] ** 2))

        return transformed

    def transform(self, document: str) -> np.ndarray:
        """Transform a given document

        Args:
            document: A str document

        Returns:
            A numpy array of shape (1,vocab_length)
        """
        transformed = np.zeros((len(self.vocabulary)))
        words = self.__preprocess_sentence(document)
        word_counts = self.calculate_tf(words)
        for word, value in word_counts.items():
            if word not in self.vocabulary:
                continue
            transformed[self.vocabulary[word]] = value * self.idf[word]

        return transformed

    def fit_transform(self, documents: List[str]) -> np.ndarray:
        self.fit(documents)
        return self.transform_train()

    def calculate_tf(self, words: List[str]) -> Counter:
        word_counts = Counter(words)
        # word_counts = {word: value/len(sentence) for word, value in word_counts.items()} Uses true Term-Frequency function
        return word_counts

    def __remove_punctuations(self, sentence: str) -> str:
        translate_table = str.maketrans("", "", string.punctuation)
        sentence = sentence.translate(translate_table)
        return sentence

    def __preprocess_sentence(self, document: str) -> List[str]:
        """Cleans the documents

        - Lowercase
        - Remove punctuations
        - Split document into words
        - Keep words that has 2 or more letters

        Args:
            document: A document to process

        Returns:

        """
        document = document.lower()
        document = self.__remove_punctuations(document)
        document = document.split()
        document = filter(lambda word: len(word) >= 2, document)

        return list(document)


x = CustomTFIDF()
print(x.fit_transform(sentences)[1])

print(np.sum(X.toarray()[1]))
print(np.sum(x.fit_transform(sentences)[1]))

from sklearn.datasets import fetch_20newsgroups
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

data, target = fetch_20newsgroups(return_X_y=True, subset="train")
test_data, test_target = fetch_20newsgroups(return_X_y=True, subset="test")

x = CustomTFIDF()
data = x.fit_transform(data)

classifier = DecisionTreeClassifier(max_depth=50)
classifier.fit(data, target)

predictions = classifier.predict([x.transform(data) for data in tqdm(test_data)])

print(classification_report(test_target, predictions))
