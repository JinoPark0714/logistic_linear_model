from flask import Flask, request
from pybo.predict_success import PredictionSuccessService
from pybo.predict_rating import PredictionRatingService
from operator import itemgetter

app = Flask(__name__)
predictionSuccessService = PredictionSuccessService()
predictionRatingService = PredictionRatingService()

@app.route('/')
def hello_world():
    return "Hello world!"

# 성공 유무 예측
@app.route('/predict/success', methods=['POST'])
def hello_pybo():
    params = request.json
    title, description, genre, format = itemgetter('title', 'description', 'genre', 'format')(params)
    isSuccess = predictionSuccessService.predictRating(title, description, genre, format)
    result = '잘못된 값'
    print(isSuccess)
    if isSuccess == 1:
        result = '성공'
    elif isSuccess == 0:
        result = '실패'
    return result

# 평점 예측
@app.route('/predict/rating', methods=["POST"])
def predict_rating():
    params = request.json
    description, summary, genre = itemgetter('description', 'summary', 'genre')(params)
    rating = predictionRatingService.predictRating(description, summary, genre)    
    return str(rating)