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

class PredictionRatingService:
    def __init__ (self):
        self.rhino = rhinoMorph.startRhino()

    # 평점 예측을 시도한다.
    def predictRating(self, description, summary, genre):
        user_dictionary = {
            "description" : description,
            "summary" : summary,
            "genre" : genre
        }

        user_dataFrame = pd.DataFrame(user_dictionary, index=[0])

        # 토큰화 및 특수문자 제거
        user_dataFrame['description'] = user_dataFrame['description'].apply(self.text_preprocess)
        user_dataFrame['summary'] = user_dataFrame['summary'].astype(str)
        user_dataFrame['summary'] = user_dataFrame['summary'].apply(self.text_preprocess)

        # 소개글 벡터화
        tf_idf_description = self.get_tf_idf_vectorizer('tf_idf_description_linear_regression.pickle')
        description_transformed = tf_idf_description.transform(user_dataFrame['description'])

        # 개요 벡터화
        tf_idf_summary = self.get_tf_idf_vectorizer('tf_idf_summary_linear_regression.pickle')
        summary_transformed = tf_idf_summary.transform(user_dataFrame['summary'])

        # 원 핫 인코딩
        one_hot_genre = self.get_one_hot_encoder()
        genre_transformed = one_hot_genre.transform(user_dataFrame['genre'].values.reshape(-1, 1))

        X_user = self.toDataFrame(description_transformed, summary_transformed, genre_transformed)
        
        model = self.get_model_ridge()
        rating_prediction = model.predict(X_user)
        return round(rating_prediction[0], 2)

    # 모델에 적합된 데이터구조로 입력시키기 위해 데이터 구조 변경
    def toDataFrame(self, description, summary, genre):
        return pd.concat(
            [
                pd.DataFrame(description.toarray()), 
                pd.DataFrame(summary.toarray()),
                pd.DataFrame(genre.toarray())
            ], 
            axis = 1)

    # rhino 형태소 분석
    def text_preprocess(self, text):
        pos = ['NNG', 'NNP', 'VV', 'VA', 'XR', 'IC', 'MM', 'MAG', 'MAJ']
        text = rhinoMorph.onlyMorph_list(self.rhino, text, pos, eomi=True)
        return ' '.join(text)


    # tf-idf 객체 받아오기
    def get_tf_idf_vectorizerasdff(self, user_dataFrame):
        count_vectorizer = ''
        with open('./pybo/ridge_lasso.pickle', 'rb') as f:
            count_vectorizer = pickle.load(f)

        title_description = count_vectorizer.transform(user_dataFrame['title_description'])
        return title_description.toarray()


    # tf-idf 객체 받아오기
    def get_tf_idf_vectorizer(self, file_name):
        tf_idf_vectorizer = ''
        with open(f"./pybo/ridge_lasso/{file_name}", 'rb') as f:
            tf_idf_vectorizer = pickle.load(f)
        return tf_idf_vectorizer

    # 원-핫 인코딩 받아오기
    def get_one_hot_encoder(self):
        one_hot_encoder = ''
        with open('./pybo/ridge_lasso/one_hot_encoder_linear_regression.pickle', 'rb') as f:
            one_hot_encoder = pickle.load(f)
        return one_hot_encoder


    # Ridge 모델
    def get_model_ridge(self):
        model = ''
        with open('./pybo/ridge_lasso/ridge_cross_validate.pickle', 'rb') as f:
            model = pickle.load(f)
        return model
    
    # Lasso 모델
    def get_model_lasso(self):
        model = ''
        with open('./pybo/ridge_lasso/lasso_cross_validate.pickle', 'rb') as f:
            model = pickle.load(f)
        return model
