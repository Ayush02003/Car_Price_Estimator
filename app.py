from flask import Flask, render_template, request, jsonify
import joblib

app = Flask(__name__)

model = joblib.load('model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    year = data['Year']
    mileage = data['Mileage']
    owners = data['Owners']
    purchase_price = data['PurchasePrice']
    predicted_price = model.predict([[year, mileage, owners, purchase_price ]])
    response = {'predicted_price': predicted_price[0]}
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
