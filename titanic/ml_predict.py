def prediction_model(pclass,sex,age,sibsp,parch,fare,embarked):  #,title):
    import pickle
    x = [[pclass,sex,age,sibsp,parch,fare,embarked]]#,title]]
    #randomforest = pickle.load(open(r"C:\Users\20190394\Documents\django\titanic\titanic_model.sav", 'rb'))
    randomforest = pickle.load(open("titanic_model.sav"))
    prediction = randomforest.predict(x)
    if prediction == 0:
        prediction = "Not survived"
    elif prediction == 1:
        prediction = "Survived"
    else:
        prediction = 'Error'
    return prediction

#prediction_model(1,1,11,1,1,19,1)
