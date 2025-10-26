import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

from flask import Flask, request, jsonify, render_template
import pickle
from csv import writer

app = Flask('')
@app.route('/')
def home():
	return render_template('predictorform.html')

@app.route('/result',methods=['POST'])
def prediction():
	e2 = float(request.form['entry2'])
	e3 = float(request.form['entry3'])
	e4 = float(request.form['entry4'])
	e5 = float(request.form['entry5'])
	e6 = float(request.form['entry6'])
	e7 = float(request.form['entry7'])
	e8 = float(request.form['entry8'])

	l = list()
    
	l1 = [e2,e3,e4,e5,e6,e7,e8]
	l.extend(l1)

	input_data = [l]

	diabetes_dataset = pd.read_csv('diabetes.csv')
	diabetes_dataset.head()
	diabetes_dataset.shape

	diabetes_dataset.describe()

	diabetes_dataset['Outcome'].value_counts()


	diabetes_dataset.groupby('Outcome').mean()
	X = diabetes_dataset.drop(columns = 'Outcome', axis=1)
	Y = diabetes_dataset['Outcome']

	print(X)

	print(Y)

	scaler = StandardScaler()
	scaler.fit(X)
	standardized_data = scaler.transform(X)
	print(standardized_data)
	X = standardized_data
	Y = diabetes_dataset['Outcome']
	print(X)
	print(Y)


	X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, stratify=Y, random_state=2)

	print(X.shape, X_train.shape, X_test.shape)



	classifier = svm.SVC(kernel='linear')

#training the support vector Machine Classifier
	classifier.fit(X_train, Y_train)

	X_train_prediction = classifier.predict(X_train)
	training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

	print('Accuracy score of the training data : ', training_data_accuracy)


	X_test_prediction = classifier.predict(X_test)
	test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

	print('Accuracy score of the test data : ', test_data_accuracy)
	#input_data = (166,72,19,175,25.8,0.587,51)
    
# changing the input_data to numpy array
	numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
	data_reshaped = numpy_array.reshape(1,-1)

# standardize the input data
	std_data = scaler.transform(data_reshaped)
	print(std_data)

	prediction = classifier.predict(std_data)
	print(prediction)
	statement = ""

	if (prediction[0] == 0):
  		return render_template('results.html',statement = "Not diabetic" )
	else:
  		return render_template('results.html',statement = "diabetic")

if __name__=="__main__":
    app.run("localhost", "9999", debug=True)
