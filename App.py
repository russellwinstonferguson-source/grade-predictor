from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def web_page_gen():
    return render_template('index.txt')


@app.route('/predict', methods=['POST'])
def predict():
    # 1. Extract data from the form using the names defined in your HTML
    # Note: These names must match exactly what is in your HTML <input name="...">
    practice1 = int(request.form['Practice 1'])
    practice2 = int(request.form['Practice 2'])
    
    # 2. Prepare the data for the model
    input_features = np.array([[practice1, practice2]])
    
    # 3. Predict and calculate result
    prediction = model.predict(input_features)
    output = int(round(prediction[0]))
    
    # 4. Create the feedback string
    result_text = f'Scores of {practice1} and {practice2} in the practice test indicate a score of {output} in the final test'
    
    # 5. Return to the page with the answer populated
    return render_template('index.txt', answer=result_text)

if __name__ == "__main__":
    app.run(debug=True)