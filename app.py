import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from flask import Flask, request, render_template

# Initialize Flask app
app = Flask(__name__)


# Load and preprocess dataset
diabetes_dataset = pd.read_csv('diabetes-dataset.csv')

# Separate features and target
X = diabetes_dataset.drop(columns='Outcome', axis=1)
Y = diabetes_dataset['Outcome']

# Standardize feature data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(
    X_scaled, Y, test_size=0.2, stratify=Y, random_state=2
)

# Train the SVM model
classifier = svm.SVC(kernel='rbf', C=100, gamma='scale', class_weight='balanced')
classifier.fit(X_train, Y_train)

# Evaluate model performance
train_acc = accuracy_score(Y_train, classifier.predict(X_train))
test_acc = accuracy_score(Y_test, classifier.predict(X_test))
print(f"Training Accuracy: {train_acc:.4f}")
print(f"Testing Accuracy: {test_acc:.4f}")


# Flask routes
@app.route('/')
def home():
    """Render input form."""
    return render_template('predictorform.html')


@app.route('/result', methods=['POST'])
def prediction():
    """Handle form submission and predict outcome."""
    try:
        # Extract form inputs
        inputs = [float(request.form[f'entry{i}']) for i in range(1, 9)]
        input_array = np.asarray(inputs).reshape(1, -1)

        # Standardize input data
        standardized_input = scaler.transform(input_array)

        # Predict diabetes outcome
        model_result = classifier.predict(standardized_input)[0]
        statement = "Not diabetic" if model_result == 0 else "Diabetic"

        return render_template('results.html', statement=statement)
    except Exception as e:
        return render_template('results.html', statement=f"Error: {str(e)}")


# run app
if __name__ == "__main__":
    app.run("localhost", 9999, debug=True)
