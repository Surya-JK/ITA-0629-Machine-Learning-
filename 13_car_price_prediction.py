import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

data = {
    'Year': [2015, 2018, 2020, 2016, 2019, 2014],
    'Present_Price': [5.59, 9.54, 9.85, 4.15, 6.87, 4.60],
    'Kms_Driven': [27000, 43000, 6900, 5200, 42450, 72000],
    'Fuel_Type': ['Petrol', 'Diesel', 'Petrol', 'Petrol', 'Diesel', 'Petrol'],
    'Selling_Price': [3.35, 4.75, 7.25, 2.85, 5.90, 3.00]
}
df = pd.DataFrame(data)

X = df.drop('Selling_Price', axis=1)
y = df['Selling_Price']

ct = ColumnTransformer([('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = ct.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

print("R2 Score:", model.score(X_test, y_test))
print("Predicted Price:", model.predict(X_test[0].reshape(1, -1)))