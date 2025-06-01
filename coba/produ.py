import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config
st.set_page_config(page_title="Productivity Score Predictor", layout="wide")

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("./ai_dev_productivity.csv")

df = load_data()

# Calculate productivity score
df["productivity_score"] = (0.5 * df["commits"] + 0.3 * df["task_success"] - 0.2 * df["bugs_reported"])

# Prepare features and target
X = df[["ai_usage_hours", "sleep_hours", "coffee_intake_mg", "cognitive_load", "distractions"]]
y = df["productivity_score"]

# Data preprocessing
X = X.fillna(X.median())
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# App title
st.title("Developer Productivity Score Prediction")

# Sidebar for user input
st.sidebar.header("Input Parameters")

def user_input_features():
    ai_hours = st.sidebar.slider("AI Usage Hours", 0.0, 12.0, 4.0, 0.5)
    sleep = st.sidebar.slider("Sleep Hours", 0.0, 12.0, 7.0, 0.5)
    coffee = st.sidebar.slider("Coffee Intake (mg)", 0, 500, 200, 10)
    cognitive = st.sidebar.slider("Cognitive Load (1-10 scale)", 1, 10, 5)
    distractions = st.sidebar.slider("Distractions (hours)", 0.0, 8.0, 2.0, 0.5)
    
    data = {
        'ai_usage_hours': ai_hours,
        'sleep_hours': sleep,
        'coffee_intake_mg': coffee,
        'cognitive_load': cognitive,
        'distractions': distractions
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# Display user inputs
st.subheader("Your Input Parameters")
st.write(input_df)

# Preprocess input
input_scaled = scaler.transform(input_df)

# Make prediction
prediction = model.predict(input_scaled)

# Display prediction
st.subheader("Predicted Productivity Score")
st.write(f"### {prediction[0]:.2f}")

# Interpretation
st.subheader("Interpretation")
st.write("""
The productivity score is calculated as:
- 50% based on code commits
- 30% based on task success rate
- 20% penalized by bugs reported

Higher scores indicate better productivity.
""")

# Model performance
st.subheader("Model Performance")
st.write(f"R² Score: {model.score(X_test, y_test):.2f}")
st.write("""
R² score indicates how well the model explains the variability in productivity scores.
A score of 1.0 would be perfect prediction.
""")

# Feature importance
st.subheader("Feature Importance (Model Coefficients)")
coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
}).sort_values('Coefficient', ascending=False)

st.write(coefficients)

# Visualization
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x='Coefficient', y='Feature', data=coefficients, palette='viridis', ax=ax)
ax.set_title('Impact of Each Factor on Productivity Score')
st.pyplot(fig)

# Explanation
st.write("""
- **Positive coefficients** indicate factors that increase productivity
- **Negative coefficients** indicate factors that decrease productivity
- The magnitude shows the relative strength of each factor's impact
""")

# Save model option
if st.sidebar.button('Save This Model'):
    pickle.dump(model, open('productivity_predictor.pkl', 'wb'))
    st.sidebar.success('Model saved as productivity_predictor.pkl')

# Data exploration
if st.checkbox('Show Raw Data'):
    st.subheader('Raw Data')
    st.write(df)