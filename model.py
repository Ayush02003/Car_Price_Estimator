import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib

np.random.seed(0)
n_samples = 1000
year = np.random.randint(2000, 2023, size=n_samples)
mileage = np.random.randint(6, 50, size=n_samples)
owners = np.random.randint(1, 3, size=n_samples)
purchase_price = np.random.randint(200000, 5000000, size=n_samples)  # Adding purchase price field
price = (year - 2000) - 0.2 * mileage + 3000 * (3 - owners) + (purchase_price-(purchase_price*20/100))

data = pd.DataFrame({'Year': year, 'Mileage': mileage, 'Owners': owners, 'PurchasePrice': purchase_price, 'Price': price})

X = data[['Year', 'Mileage', 'Owners', 'PurchasePrice']]  # Include 'PurchasePrice'
y = data['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

joblib.dump(model, 'model.pkl')

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print('Mean Squared Error:', mse)

example_data = [[2015, 50000, 2, 20000]] 
predicted_price = model.predict(example_data)
print('Predicted Price:', predicted_price)
