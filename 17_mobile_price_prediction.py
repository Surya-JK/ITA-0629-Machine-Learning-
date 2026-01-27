import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = {
    'RAM': [4, 6, 8, 3, 12, 4],
    'Internal_Storage': [64, 128, 256, 32, 512, 64],
    'Battery': [4000, 5000, 4500, 3500, 5000, 4000],
    'Price': [15000, 25000, 40000, 10000, 70000, 15500]
}
df = pd.DataFrame(data)

X = df[['RAM', 'Internal_Storage', 'Battery']]
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

print("Predicted Price for 8GB RAM, 128GB Storage, 4000mAh:", model.predict([[8, 128, 4000]]))
print("Model Score:", model.score(X_test, y_test))