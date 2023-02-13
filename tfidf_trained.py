import re
import csv
import random
from numpy import ndarray
from numpy import zeros
from numpy import subtract
from numpy.random import rand 
from numpy import absolute
from numpy import std
from numpy import mean
from os import environ
from pathlib import Path
import pandas as pd
from pandas import DataFrame
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import nltk
from numpy import dot 
from numpy.linalg import norm

from prepare_data_set import prepare_data_set

nltk.download('stopwords')


class MisogynyModel:

    def __init__(self, data_set: DataFrame):
        self.data_set = data_set
        self.tfidf: ndarray = self.get_tfidf_embeddings(self.data_set)
        self.dimension = self.tfidf.shape[1]

    def get_tfidf_embeddings(self, data_set: DataFrame):
        vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'), use_idf=True)
        return vectorizer.fit_transform(data_set['Definition']).toarray()
        
    def select_training_set(self):
        total_elements = len(self.data_set)
        self.training_set_idx = random.sample(range(0, total_elements), int(total_elements * 0.9))
        self.test_set_idx = [i for i in range(total_elements) if i not in self.training_set_idx]
        self.training_set =  self.data_set.iloc[self.training_set_idx]
        self.validation_set = self.data_set.iloc[self.test_set_idx]
    
    def cosine(self, v1: ndarray, v2: ndarray) -> float:
        return dot(v1, v2) / (norm(v1) * norm(v2))
    
    def get_tfidf_embedding(self, idx: int) -> ndarray:
        return self.tfidf[idx].reshape((self.dimension, 1))

    def train(self):
        self.misogyny_embedding = zeros(shape=(self.dimension, 1))
        self.non_misogyny_embedding = zeros(shape=(self.dimension, 1))
        self.bias = 0
        for idx, item in self.training_set.iterrows():
            if item['is_misogyny'] == 1:
                embedding = self.get_tfidf_embedding(idx - 1)
                self.misogyny_embedding = mean([self.misogyny_embedding, embedding], axis=0)
            else:
                embedding = self.get_tfidf_embedding(idx - 1)
                self.non_misogyny_embedding = mean([self.non_misogyny_embedding, embedding], axis=0)
        
        similarity = self.misogyny_embedding - self.non_misogyny_embedding
        self.misogyny_embedding = subtract(self.misogyny_embedding, absolute(similarity))
        self.non_misogyny_embedding = subtract(self.non_misogyny_embedding, absolute(similarity))

    def validate(self):
        true_positive = 0
        false_positive = 0
        false_negative = 0
        true_negative = 0
        for idx, item in self.validation_set.iterrows():
            mis_sim = self.cosine(self.misogyny_embedding.T, self.get_tfidf_embedding(idx - 1))
            non_mis_sim = self.cosine(self.non_misogyny_embedding.T, self.get_tfidf_embedding(idx - 1))
            prediction = int(mis_sim > non_mis_sim)
            if prediction == 1:
                if prediction == item['is_misogyny']:
                    true_positive += 1
                elif prediction != item['is_misogyny']:
                    false_positive += 1
            else:
                if prediction != item['is_misogyny']:
                    false_negative += 1
                elif prediction == item['is_misogyny']:
                    true_negative += 1

        accuracy = true_positive / len(self.validation_set)
        print(f"accuracy: {accuracy}")
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1
        

if __name__ == '__main__':

    data_file_path = Path['DATA_FILE_NAME'](__file__) / 'data' / 'ManualTag_Misogyny.csv'
    df = pd.read_csv('data/ManualTag_Misogyny.csv', encoding='latin-1')
    data_set = prepare_data_set(df)
    mm = MisogynyModel(data_set)
    k = 10
    f1 = []
    for i in range(k):
        mm.select_training_set()
        mm.train()
        f1.append(mm.validate())
    print(f"{k}-fold repeated random subsampling validation: f1 score = {mean(f1)}")
    print(f"{k}-fold repeated random subsampling validation: f1 score standard deviation = {std(f1)}")

    