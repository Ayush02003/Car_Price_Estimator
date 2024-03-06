import requests
import json

url = 'http://localhost:5000/predict' 

data = {
    'Year': 2015,
    'Mileage':25,
    'Owners': 2
}
json_data = json.dumps(data)

headers = {'Content-Type': 'application/json'}

response = requests.post(url, data=json_data, headers=headers)

print('Response:', response.json())
