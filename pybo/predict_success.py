import pickle
import rhinoMorph
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

import sys
sys.stdin.reconfigure(encoding='utf-8')

import warnings
warnings.filterwarnings(action='ignore')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import logging
logging.getLogger().setLevel(logging.ERROR)

class PredictionSuccessService:
    def __init__ (self):
        self.rhino = rhinoMorph.startRhino()


    # 평점 예측을 시도한다.
    def predictRating(self, title, description, genre, format):
        user_dictionary = {
            "title" : title,
            "description" : description,
            "genre" : genre,
            "format" : format
        }

        user_dataFrame = pd.DataFrame(user_dictionary, index=[0])
        user_dataFrame['title_description'] = user_dataFrame['title'] + ' ' + user_dataFrame['description']
        user_dataFrame['title_description'] = user_dataFrame['title_description'].apply(self.text_preprocess)

        title_description = self.get_title_description_transformed_vectorizer(user_dataFrame)
        genre_format = self.get_category_value_one_hot_encoded(user_dataFrame)

        X_user = pd.concat([pd.DataFrame(title_description), pd.DataFrame(genre_format)], axis=1)

        logistic_regressor = self.get_model_logistic_regression()

        rating_prediction = logistic_regressor.predict(X_user)
        return rating_prediction[0]


    # rhino 형태소 분석
    def text_preprocess(self, text):
        pos = ['NNG', 'NNP', 'VV', 'VA', 'XR', 'IC', 'MM', 'MAG', 'MAJ']
        text = rhinoMorph.onlyMorph_list(self.rhino, text, pos, eomi=True)
        return ' '.join(text)


    # count Vectorizer
    def get_title_description_transformed_vectorizer(self, user_dataFrame):
        count_vectorizer = ''
        with open('./pybo/logistic_regression/count_vectorizer.pickle', 'rb') as f:
            count_vectorizer = pickle.load(f)

        title_description = count_vectorizer.transform(user_dataFrame['title_description'])
        return title_description.toarray()


    # 원-핫 인코딩
    def get_category_value_one_hot_encoded(self, user_dataFrame):
        one_hot_encoder = ''

        with open('./pybo/logistic_regression/one_hot_encoder.pickle', 'rb') as f:
            one_hot_encoder = pickle.load(f)

        one_hot_genre_format = one_hot_encoder.transform(user_dataFrame[['genre', 'format']])
        return one_hot_genre_format.toarray()


    # 로지스틱 회귀 모델
    def get_model_logistic_regression(self):
        model = ''
        with open('./pybo/logistic_regression/logistic_regression.pickle', 'rb') as f:
            model = pickle.load(f)
        return model
