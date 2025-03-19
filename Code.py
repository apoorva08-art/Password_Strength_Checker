import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Sample datasets
data = {
    'password': [
        '123456', 'password', 'qwerty', 'abc123', 'pass@123',
        'Apoorva@123', 'MyP@ssw0rd2023', 'helloWorld', 'LetMeIn!',
        '1qaz2wsx', 'S@f3_P@ss', 'iloveyou', 'admin', 'welcome123',
        'SecureP@ssword!', 'Data$2022', 'weakpass', 'superman123',
        'P@55word!', '12345678'
    ],
    'strength': [
        'Weak', 'Weak', 'Weak', 'Weak', 'Medium',
        'Strong', 'Strong', 'Medium', 'Medium',
        'Medium', 'Strong', 'Weak', 'Weak', 'Medium',
        'Strong', 'Strong', 'Weak', 'Medium',
        'Strong', 'Weak'
    ]
}

df = pd.DataFrame(data)

# Convert passwords to character-level features
vectorizer = CountVectorizer(analyzer='char', ngram_range=(2, 4))
X = vectorizer.fit_transform(df['password'])
y = df['strength']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Model Accuracy (Logistic Regression): {acc * 100:.2f}%")

# User input 
while True:
    user_pass = input("\nEnter a password to check its strength (or type 'exit'): ")
    if user_pass.lower() == 'exit':
        print("Exiting the Password Strength Checker.")
        break
    user_vec = vectorizer.transform([user_pass])
    result = model.predict(user_vec)[0]
    print(f"🔐 Predicted Strength: {result}")
