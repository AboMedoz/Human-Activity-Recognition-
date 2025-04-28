import pickle

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split


# https://www.kaggle.com/datasets/uciml/human-activity-recognition-with-smartphones
df = pd.read_csv('Dataset/train.csv')

# Drop 'subject' as it's useless ig?
df = df.drop('subject', axis=1)

# Factorize
df['Activity'], _ = pd.factorize(df['Activity'])

# Features
x = df.copy().drop('Activity', axis=1)
y = df['Activity']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
cv_score = cross_val_score(model, x, y, cv=5)
print(f"Cross-Validation Score: {cv_score}")
print(f"Mean Cross-Validation Score: {cv_score.mean():.4f}")
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

# Metrics
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}")
print(f"Classification Report: {classification_report(y_test, y_pred)}")
print(f"Confusion Matrix: {confusion_matrix(y_test, y_pred)}")

feature_importance = model.feature_importances_
important_features = pd.Series(feature_importance, index=x.columns).sort_values(ascending=False)
print(f"Feature Importance:\n{important_features}")

with open('human_activity_recognition.pkl', 'wb') as file:
    pickle.dump(model, file)
print('Model Saved.')