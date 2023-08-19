import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from flask import Flask,request ,render_template , jsonify

col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']

# load dataset
pima = pd.read_csv(r"C:\Users\User\PycharmProjects\testProj\diabetes.csv",header=None, names=col_names)
pima=pima[1:]
pima.head()


X = pima.drop(['label','pregnant'],axis=1)


y=pima['label'].values


# split X and y into training and testing sets
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1)

# Model Training using Decision Tree Classifier
model = DecisionTreeClassifier()

# training the model with training data
model.fit(X_train, y_train)
predictions = model.predict(X_train)

# accuracy score on training data
print("Accuracy : ",accuracy_score(y_train,predictions)*100)



app = Flask(__name__)


@app.route('/')
def home_page():
    return render_template('index.html')


@app.route('/prediction',methods=['POST'])

def prediction():
   
    if(request.method == 'POST'):
        glucose = float(request.form['glucose'])
        bp = float(request.form['bp'])
        skin = float(request.form['skin'])
        insulin = float(request.form['insulin'])
        bmi = float(request.form['bmi'])
        pedigree = float(request.form['pedigree'])
        age = float(request.form['age'])

        input_data = np.array([[glucose,bp,skin,insulin,bmi,pedigree,age]])
        r = model.predict(input_data)
        
        if r=='0':
            result = "User is not diabetic"

        if r=='1':
            result = "User is diabetic"

        return render_template('result.html' , result = result)



if __name__=="__main__":
    app.run(debug=True)
