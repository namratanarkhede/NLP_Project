import pandas as pd
from django.http import HttpResponse
from django.shortcuts import render
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier


df1 = pd.read_csv("./model/fake_or_real_news.csv")
x = df1['text']
y = df1['label']

x_train , x_test, y_train, y_test = train_test_split(x,y,test_size =0.2,random_state =42)

tfvect = TfidfVectorizer(stop_words='english', max_df=0.7)
tfid_x_train = tfvect.fit_transform(x_train)
tfid_x_test = tfvect.transform(x_test)


classifier = PassiveAggressiveClassifier(max_iter=50)
classifier.fit(tfid_x_train , y_train)

pac_pred = classifier.predict(tfid_x_test)



loaded_model = joblib.load('./model/passive_model.pkl')

def index(request):  # By defaut index takes an arguement otherwise it throws an error
    return render(request, 'index.html')

def analyze(request):

    if request.method =='POST':
        return render(request, 'analyze.html')

def predection(request):
    if request.method == 'POST':

        pred_text = request.POST.get('text')
        input_data = [pred_text]
        vectorized_input_data = tfvect.transform(input_data)
        prediction = classifier.predict(vectorized_input_data)
        print(pred_text)
        print(prediction)

    context = {'text': prediction}
    return render(request, 'analyze.html', context)


def about_us(request):
    return render(request, 'about_us.html')
