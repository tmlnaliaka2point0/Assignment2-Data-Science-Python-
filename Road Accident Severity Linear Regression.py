import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib # Library for saving the model

# 1. DATA PREPARATION

# Synthetic dataset that mimics accident data features
data = {
    'Accident_Severity': ['Minor', 'Serious', 'Fatal', 'Minor', 'Serious', 'Fatal', 'Minor', 'Serious', 'Minor', 'Fatal'],
    'Num_Casualties': [1, 3, 5, 2, 4, 6, 1, 3, 2, 5],
    'Speed_Limit': [30, 50, 70, 30, 60, 70, 40, 50, 30, 70],
    'Weather_Condition': ['Clear', 'Rain', 'Clear', 'Fog', 'Rain', 'Snow', 'Clear', 'Rain', 'Fog', 'Clear'],
    'Road_Surface': ['Dry', 'Wet', 'Dry', 'Dry', 'Wet', 'Ice', 'Dry', 'Wet', 'Dry', 'Dry'],
    'Is_Junction': [0, 1, 1, 0, 1, 0, 0, 1, 0, 1]
}
df = pd.DataFrame(data)

# Convert the categorical 'Accident_Severity' to a numerical target (1, 2, 3)
severity_map = {'Minor': 1, 'Serious': 2, 'Fatal': 3}
df['Accident_Severity_Score'] = df['Accident_Severity'].map(severity_map)

# Define Features (X) and Target (Y)
X = df.drop(['Accident_Severity', 'Accident_Severity_Score'], axis=1)
y = df['Accident_Severity_Score']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define preprocessing steps (StandardScaler for numerical, OneHotEncoder for categorical)
categorical_features = ['Weather_Condition', 'Road_Surface']
numerical_features = ['Num_Casualties', 'Speed_Limit', 'Is_Junction']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'
)

#2. MODEL CREATION AND FITTING 

# Create a pipeline that combines preprocessing and the Linear Regression model
model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                 ('regressor', LinearRegression())])

# Train the model (FITTING)
model_pipeline.fit(X_train, y_train)

# Output for Model Creation/Fitting 1

print("Model trained successfully.")
print(f"R-squared on Test Set: {model_pipeline.score(X_test, y_test):.4f}") 
# Note: R-squared will be high due to the simple synthetic data
print("-" * 40)
print()

# 3. MODEL SAVING

# Save the entire pipeline (preprocessor + model)
MODEL_FILENAME = 'severity_linear_regression_model.pkl'
joblib.dump(model_pipeline, MODEL_FILENAME)

# Output for  Model Saving 2
print(f"Model successfully saved as: {MODEL_FILENAME}")
print("-" * 40)
print()

# 4. MODEL LOADING AND PREDICTION EXAMPLE (Demonstration) 

# Load the saved model
loaded_model = joblib.load(MODEL_FILENAME)

# Define new, hypothetical data for prediction
# Scenario: Heavy Rain, High Speed Limit, High Casualties, at a Junction
new_accident_data = pd.DataFrame({
    'Num_Casualties': [8],
    'Speed_Limit': [60],
    'Weather_Condition': ['Rain'],
    'Road_Surface': ['Wet'],
    'Is_Junction': [1]
})

# Make the prediction
predicted_score = loaded_model.predict(new_accident_data)[0]

# Output for Model Loading and Prediction 3

print(f"Hypothetical Input:\n{new_accident_data.iloc[0].to_dict()}")
print(f"\nModel Loaded: {type(loaded_model)}")
print(f"Predicted Accident Severity Score: {predicted_score:.2f} \n")

# Interpretation based on the score (1=Minor, 2=Serious, 3=Fatal)
if predicted_score >= 2.5:
    severity_interp = "Likely Fatal (Score >= 2.5)"
elif predicted_score >= 1.5:
    severity_interp = "Likely Serious (Score between 1.5 and 2.5)"
else:
    severity_interp = "Likely Minor (Score < 1.5)"
    
print(f"Interpretation: {severity_interp}")
print("-" * 40)