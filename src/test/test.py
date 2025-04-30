import os
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

BASE_DIR = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(BASE_DIR, '..'))
TEST_PATH = os.path.join(ROOT, 'data', 'Dataset', 'train.csv')
MODEL_PATH = os.path.join(ROOT, 'model', 'human_activity_recognition.pkl')
LABELS_PATH = os.path.join(ROOT, 'model', 'activity_labels.pkl')

with open(MODEL_PATH, 'rb') as model_file:
    model = pickle.load(model_file)
with open(LABELS_PATH, 'rb') as labels:
    activity_labels = pickle.load(labels)

df = pd.read_csv(TEST_PATH)
print(df.head(5))
df.info()

print(np.sum(df.isna(), axis=0))  # No NA's

df = df.drop('subject', axis=1)

df['Activity'] = pd.Categorical(df['Activity'], categories=activity_labels)
df['Activity'] = df['Activity'].cat.codes

x = df.copy().drop('Activity', axis=1)
y = df['Activity']

y_pred = model.predict(x)

df['Predicted_Activity'] = y_pred
df.to_csv('prediction.csv')

print(f"Accuracy: {accuracy_score(y, y_pred) * 100:.2f}")
print(f"Classification Report: {classification_report(y, y_pred)}")
print(f"Confusion Matrix: {confusion_matrix(y, y_pred)}")
