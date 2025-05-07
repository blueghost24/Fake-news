import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import os

# Load the data
true_df = pd.read_csv('data/True.csv')
false_df = pd.read_csv('data/Fake.csv')


# Add labels: 0 for real, 1 for fake
true_df['label'] = 0
false_df['label'] = 1

# Combine and shuffle
df = pd.concat([true_df, false_df])
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Combine title and text
df['content'] = df['title'].fillna('') + ' ' + df['text'].fillna('')

# Prepare features and labels
X = df['content']
y = df['label']

# Vectorize
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_vec = vectorizer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Save the model and vectorizer
os.makedirs('model', exist_ok=True)
joblib.dump((vectorizer, model), 'model/fake_news_model.pkl')

print("✅ Model trained and saved to model/fake_news_model.pkl")
