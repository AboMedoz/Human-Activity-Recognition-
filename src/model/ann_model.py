import joblib
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

# MACROS
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(ROOT, 'data', 'train')
MODELS_PATH = os.path.join(ROOT, 'models')

df = pd.read_csv(os.path.join(DATA_PATH, 'train.csv'))

x = df.drop(columns=['subject', 'Activity'])
y = df['Activity']
subjects = df['subject']

# Split Dataset by Subject
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(x, y, groups=subjects))
x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

# Scale
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Encode Labels
le = LabelEncoder()
y_train = to_categorical(le.fit_transform(y_train))
y_test = to_categorical(le.transform(y_test))

# Model
model = Sequential()
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(y_train.shape[1], activation='softmax'))
model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=['accuracy'])
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=20)

_, acc = model.evaluate(x_test, y_test)
print(f'Accuracy: {acc:.2f}')

# Save The Model & Utils
model.save(os.path.join(MODELS_PATH, 'ann_model.keras'))
joblib.dump(scaler, os.path.join(MODELS_PATH, "ann_scaler.pkl"))
joblib.dump(le, os.path.join(MODELS_PATH, "ann_label_encoder.pkl"))

