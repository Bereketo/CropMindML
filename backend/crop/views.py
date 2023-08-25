from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import warnings
import os
import sys

ml_path = os.path.dirname(os.path.dirname(__file__))
predict_path = os.path.join(ml_path, 'machine_learning')
sys.path.append(predict_path)

from src.predict import load_model, predict_samples
from src.model_training import preprocess_data, load_data

class FormDataView(APIView):
    def post(self, request):
        # prepare the model
        model = load_model(os.path.join(predict_path, 'models', 'crop_rf_model.pkl'))
        filename = load_data(os.path.join(predict_path,  'data', 'crop_data.csv'))
        X, y_onehot, label_encoder = preprocess_data(filename)
        
        # data form the url 
        data = request.data
        sample = []
        final_sample = []

        for k,v in data.items():
            sample.append(v)

        final_sample.append(sample)
        
        warnings.filterwarnings("ignore", category=UserWarning, message=".*RandomForestClassifier was fitted with feature names.*")
        predicted_value = predict_samples(model, final_sample, label_encoder)
        warnings.resetwarnings()

        print(predicted_value[0])
        response_data = {
                "message": "Data received successfully",
                "data": predicted_value[0]
                }
        return Response(predicted_value[0], status=status.HTTP_200_OK)
