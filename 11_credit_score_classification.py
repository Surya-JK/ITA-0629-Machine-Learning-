import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

np.random.seed(42)
n_samples = 100
income = np.random.randint(20000, 100000, n_samples)
loan = np.random.randint(10000, 200000, n_samples)

credit_score = (income > 0.4 * loan).astype(int)
mask = np.random.rand(n_samples) < 0.1
credit_score[mask] = 1 - credit_score[mask]

data = {
    "Income": income,
    "Loan": loan,
    "CreditScore": credit_score
}

df = pd.DataFrame(data)

X = df[["Income", "Loan"]]
y = df["CreditScore"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Credit Score Accuracy:", accuracy_score(y_test, y_pred))