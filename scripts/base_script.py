import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import os

np.random.seed(1)

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
import joblib

from sklearn.metrics import confusion_matrix

DATASET_PATH = "data/dataset-sentiment-analysis-preprocessed.xlsx"

df_ = pd.read_excel(DATASET_PATH)
df = df_[['Comment', 'Sent. Anal. Category']].copy()

df['Sent. Anal. Category Clean'] = (
    df['Sent. Anal. Category']
      .astype(str)
      .str.strip()
)

# Mapimi i vlerave
mapping = {
    'Positive': 'Positive',
    'ositive': 'Positive',
    'POSITiVE': 'Positive',

    # Negative
    'Negative': 'Negative',
    'NEGATIVE': 'Negative',
    'Negativ': 'Negative',

    # Neutral
    'Neutral': 'Neutral',
    'NEUTRAL': 'Neutral',
    'Neutrale': 'Neutral',
    'Neutre': 'Neutral',
    'neutre': 'Neutral',
    'Neut': 'Neutral',
    'Mixed': 'Neutral',
    'nan': 'Neutral'
}

df['Sent. Anal. Category Clean'] = (
    df['Sent. Anal. Category Clean']
      .replace(mapping)
)

df = df.drop(columns=['Sent. Anal. Category'])

target_map = {'Positive': 1, 'Negative' : 0, 'Neutral': 2}
df['target'] = df['Sent. Anal. Category Clean'].map(target_map)

df_train, df_test = train_test_split(df)

vectorizer = TfidfVectorizer(max_features = 3000)
X_train = vectorizer.fit_transform(df_train['Comment'])

X_test = vectorizer.transform(df_test['Comment'])

Y_train = df_train['target']
Y_test = df_test['target']

model = LogisticRegression(max_iter = 500)
model.fit(X_train, Y_train)
print("Train acc:", model.score(X_train, Y_train))
print("Test acc:", model.score(X_test, Y_test))

Pr_train = model.predict_proba(X_train)
Pr_test = model.predict_proba(X_test)
print("Train AUC", roc_auc_score(Y_train, Pr_train, multi_class='ovo'))
print("Test AUC", roc_auc_score(Y_train, Pr_train, multi_class='ovo'))

P_train = model.predict(X_train)
P_test = model.predict(X_test)

def plot_cm(cm, title):
    classes = ['Negative', 'Positive', 'Neutral']

    df_cm = pd.DataFrame(cm, index=classes, columns=classes)

    plt.figure(figsize=(7, 5))
    ax = sn.heatmap(
        df_cm,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        cbar_kws={'label': 'Percentage'}
    )

    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(title)

    plt.tight_layout()
    plt.show()

cm_train = confusion_matrix(Y_train, P_train, normalize='true') * 100
plot_cm(cm_train, 'Train Confusion Matrix (%)')

cm_test = confusion_matrix(Y_test, P_test, normalize='true') * 100
plot_cm(cm_test, 'Test Confusion Matrix (%)')

# positive and negative binary target
binary_target_list = [target_map['Positive'], target_map['Negative']]
df_b_train = df_train[df_train['target'].isin(binary_target_list)]
df_b_test = df_test[df_test['target'].isin(binary_target_list)]

X_train = vectorizer.fit_transform(df_b_train['Comment'])
X_test = vectorizer.transform(df_b_test['Comment'])

Y_train = df_b_train['target']
Y_test = df_b_test['target']

model = LogisticRegression(max_iter=500)
model.fit(X_train, Y_train)
print("Train acc: ", model.score(X_train, Y_train))
print("Test acc: ", model.score(X_test, Y_test))

Pr_train = model.predict_proba(X_train)[:, 1]
Pr_test = model.predict_proba(X_test)[:, 1]
print("Train AUC: ", roc_auc_score(Y_train, Pr_train))
print("Test AUC: ", roc_auc_score(Y_test, Pr_test))

model.coef_
plt.hist(model.coef_[0], bins=30)

word_index_map = vectorizer.vocabulary_
word_index_map

threshold = 2

print("Most Positive Words :")
for word, index in word_index_map.items():
  weight = model.coef_[0][index]
  if weight > threshold:
    print(word, weight)

print("Most Negative Words :")
for word, index in word_index_map.items():
  weight = model.coef_[0][index]
  if weight < -threshold:
    print(word, weight)

P_train = model.predict(X_train)
P_test = model.predict(X_test)

# Predictions (0 ose 1)
cm = confusion_matrix(Y_test, P_test)

cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

plt.figure(figsize=(8,6))
sn.heatmap(
    cm_percent,
    annot=True,
    fmt='.1f',
    cmap='Blues',
    cbar_kws={'label': 'Percentage (%)'}
)

plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Normalized Confusion Matrix (%)')
plt.show()

# Predictions (0 ose 1)
cm = confusion_matrix(Y_train, P_train)

cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

plt.figure(figsize=(8,6))
sn.heatmap(
    cm_percent,
    annot=True,
    fmt='.1f',
    cmap='Blues',
    cbar_kws={'label': 'Percentage (%)'}
)

plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Normalized Confusion Matrix (%)')
plt.show()

MODEL_DIR = "models/base_model"
os.makedirs(MODEL_DIR, exist_ok=True)

joblib.dump(model, os.path.join(MODEL_DIR, "logistic_regression.pkl"))
joblib.dump(vectorizer, os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))

print("Model saved successfully!")