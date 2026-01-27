import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = {
    'TV_Ad_Budget': [230.1, 44.5, 17.2, 151.5, 180.8, 8.7],
    'Radio_Ad_Budget': [37.8, 39.3, 45.9, 41.3, 10.8, 48.9],
    'Sales': [22.1, 10.4, 9.3, 18.5, 12.9, 7.2]
}
df = pd.DataFrame(data)

X = df[['TV_Ad_Budget', 'Radio_Ad_Budget']]
y = df['Sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

print("Predicted Sales:", model.predict(X_test))
print("Model Score:", model.score(X_test, y_test))