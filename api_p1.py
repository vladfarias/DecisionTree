from flask import Flask, request, render_template
import joblib

app = Flask(__name__)

model = joblib.load('models-decisionTree/logistic_model.pkl')
le_package_type = joblib.load('models-decisionTree/transformer_package_type.pkl')
le_product_type = joblib.load('models-decisionTree/transformer_product_type.pkl')

@app.route('/', methods=['GET'])
def index():
    return render_template('template.html')

@app.route('/predict', methods=['POST'])
def predict():
    weight = int(request.form['Weight'])
    package_type = le_package_type.transform([request.form['package_type']])[0]
    
    prediction = model.predict([[weight, package_type]])[0]
    
    # Revert the prediction to the original label
    product_type = le_product_type.inverse_transform([prediction])[0]
    
    return render_template('template.html', prediction = product_type)

port = 4040
if __name__ == '__main__':
    print(f'Server is running at port: {port}')
    app.run(debug=True, port=port)