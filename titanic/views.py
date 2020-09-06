from django.shortcuts import render
from . import fake_model
from . import ml_predict




def home(request):
    return render(request, 'index.html')

def result(request):
    user_input_age = request.GET["age"]
    return render(request, 'result.html', {'age':user_input_age})

def resultONHOLD(request):
    # pclass,sex,age,sibsp,parch,fare,embarked
    pclass = int(request.GET["pclass"])
    sex = int(request.GET["sex"])
    age = int(request.GET["age"])
    sibsp = int(request.GET["sibsp"])
    parch = int(request.GET["parch"])
    fare = int(request.GET["fare"])
    embarked = int(request.GET["embarked"])
    prediction = ml_predict.prediction_model(pclass,sex,age,sibsp,parch,fare,embarked)
    return render(request, 'result.html', {'prediction':prediction})
