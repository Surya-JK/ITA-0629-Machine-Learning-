import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

data = {
    'Income': [5000, 2000, 8000, 1500, 6000, 9000],
    'Credit_Score': [700, 500, 750, 450, 680, 800],
    'Loan_Approved': [1, 0, 1, 0, 1, 1] 
}
df = pd.DataFrame(data)

X = df[['Income', 'Credit_Score']]
y = df['Loan_Approved']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = GaussianNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Prediction for Income 4000, Score 600:", model.predict([[4000, 600]]))