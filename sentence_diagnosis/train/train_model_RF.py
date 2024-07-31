from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import joblib
import os

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, 'data.csv')
model_path = os.path.join(script_dir, 'model/rf_model.pkl')

# Load the dataset
data = pd.read_csv(data_path)

# Split the dataset into training and testing sets
X = data['sentence']
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline that includes TF-IDF and Random Forest
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1, 2), stop_words='english')),
    ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Define hyperparameters for Grid Search
parameters = {
    'tfidf__max_df': (0.5, 0.75, 1.0),
    'tfidf__min_df': (1, 5, 10),
    'clf__n_estimators': (50, 100, 200),
    'clf__max_depth': (None, 10, 20, 30)
}

# Perform Grid Search with cross-validation
grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, cv=5, scoring='f1_weighted')
grid_search.fit(X_train, y_train)

# Print the best parameters found by Grid Search
print("Best parameters found: ", grid_search.best_params_)

# Train the model with the best parameters
best_model = grid_search.best_estimator_
best_model.fit(X_train, y_train)

# Save the trained model
joblib.dump(best_model, model_path)

# Evaluate the model
from sklearn.metrics import classification_report

y_pred = best_model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))
