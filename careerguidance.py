# ---------------------------------------
# IMPORTS
# ---------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier


# ---------------------------------------
# LOAD DATA
# ---------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv('student_career_guidance_dataset_1000.csv')

    return df

df = load_data()


# ---------------------------------------
# PREPROCESSING
# ---------------------------------------

# Find categorical columns except target
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
categorical_cols.remove('Suggested_Career')     # target

# Apply one-hot encoding
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Features and Target
X = df_encoded.drop(columns=['Student_ID', 'Suggested_Career'])
y = df_encoded['Suggested_Career']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42
)

# ---------------------------------------
# GRID SEARCH FOR BEST PARAMETERS
# ---------------------------------------
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=3,
    n_jobs=-1,
    verbose=1,
    scoring='f1_weighted'
)

grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_model = grid_search.best_estimator_


# ---------------------------------------
# TRAIN FINAL MODEL
# ---------------------------------------
model = RandomForestClassifier(
    n_estimators=best_params['n_estimators'],
    max_depth=best_params['max_depth'],
    min_samples_split=best_params['min_samples_split'],
    min_samples_leaf=best_params['min_samples_leaf'],
    random_state=42
)

model.fit(X_train, y_train)

# Model Accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)


# ---------------------------------------
# STREAMLIT UI
# ---------------------------------------
st.title("🎓 Student Career Guidance Predictor")
st.write("Enter student attributes to get career suggestions.")

# Dropdown options from dataset
interests_options = df['Interests'].unique()
strengths_options = df['Strengths'].unique()
skills_options = df['Skills'].unique()
personality_options = df['Personality'].unique()
goals_options = df['Goals'].unique()
budget_options = df['Budget'].unique()
course_options = df['Recommended_Course'].unique()
scholarship_options = df['Scholarship_Option'].unique()

# ---------------------------------------
# SIDEBAR INPUT
# ---------------------------------------
with st.sidebar:
    st.header("Student Input Details")

    marks = st.slider("Marks", 50, 100, 75)
    interests = st.selectbox("Interests", interests_options)
    strengths = st.selectbox("Strengths", strengths_options)
    skills = st.selectbox("Skills", skills_options)
    personality = st.selectbox("Personality", personality_options)
    goals = st.selectbox("Goals", goals_options)
    budget = st.selectbox("Budget", budget_options)
    recommended_course = st.selectbox("Recommended Course", course_options)
    scholarship_option = st.selectbox("Scholarship Option", scholarship_options)


# ---------------------------------------
# CREATE USER INPUT DF
# ---------------------------------------
user_input = pd.DataFrame({
    'Marks': [marks],
    'Interests': [interests],
    'Strengths': [strengths],
    'Skills': [skills],
    'Personality': [personality],
    'Goals': [goals],
    'Budget': [budget],
    'Recommended_Course': [recommended_course],
    'Scholarship_Option': [scholarship_option]
})

# Encode user input
user_encoded = pd.get_dummies(user_input, columns=categorical_cols, drop_first=True)

# Match training columns
missing_cols = set(X_train.columns) - set(user_encoded.columns)
for col in missing_cols:
    user_encoded[col] = 0
user_encoded = user_encoded[X_train.columns]


# ---------------------------------------
# PREDICTION
# ---------------------------------------
if st.button("Predict Suggested Career"):
    prediction = model.predict(user_encoded)
    st.success(f"🎯 Suggested Career: **{prediction[0]}**")

# Display accuracy note
st.write("---")
st.write(f"📌 **Model Accuracy:** {accuracy:.4f}")
st.write("⚠️ Note: Accuracy is low due to dataset limitations. Use predictions carefully.")
