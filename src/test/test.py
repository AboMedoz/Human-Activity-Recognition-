import joblib
import os

import pandas as pd
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import load_model

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(ROOT, 'data', 'test')
MODELS_PATH = os.path.join(ROOT, 'models')

model = load_model(os.path.join(MODELS_PATH, 'ann_model.keras'))
scaler = joblib.load(os.path.join(MODELS_PATH, 'ann_scaler.pkl'))
le = joblib.load(os.path.join(MODELS_PATH, 'ann_label_encoder.pkl'))

df = pd.read_csv(os.path.join(DATA_PATH, 'test.csv'))

x = df.drop(columns=['subject', 'Activity'])
y = df['Activity']

x = scaler.transform(x)

y = le.transform(y)

pred_prob = model.predict(x)
predictions = pred_prob.argmax(axis=1)

accuracy = accuracy_score(y, predictions)
print(f"Accuracy: {accuracy:.2f}")