# Assignment2-Data-Science-Python-
Assignment 2 Data Science using Python
#  Road Accident Severity Prediction using Linear Regression

##  Project Summary
This project utilizes **Multiple Linear Regression** within a scikit-learn pipeline to predict the **severity score** of a road accident. The model takes various contributing factors (e.g., casualties, speed limit, weather) and outputs a numerical score (1.0 to 3.0) representing the expected degree of severity.

The full workflow is implemented: data preprocessing (scaling and encoding), model training, **model persistence** using `joblib` (saving the trained model file), and demonstration of real-world prediction.

---

## Model Specification

| Category | Variable Name | Data Type | Role in Model | Example Interpretation |
| :--- | :--- | :--- | :--- | :--- |
| **Dependent (Y)** | `Accident_Severity_Score` | Continuous/Ordinal | Target variable to predict. | 1.0 = Minor, 3.0 = Fatal |
| **Independent (X)** | `Num_Casualties` | Numerical | Direct measure of human impact. | Higher value typically increases severity. |
| **Independent (X)** | `Speed_Limit` | Numerical | Road design factor. | Higher speed limits often correlate with higher severity. |
| **Independent (X)** | `Weather_Condition` | Categorical | Environmental factor. | E.g., 'Rain', 'Snow', 'Clear' (encoded). |
| **Independent (X)** | `Road_Surface` | Categorical | Road condition factor. | E.g., 'Dry', 'Wet', 'Ice' (encoded). |
| **Independent (X)** | `Is_Junction` | Binary | Road feature factor. | 1 if at a junction, 0 otherwise. |

---

## Technical Implementation

### Key Libraries
* `pandas`
* `numpy`
* `scikit-learn` (`LinearRegression`, `Pipeline`, `ColumnTransformer`)
* `joblib` (for model saving and loading)

### Preprocessing Pipeline
To prepare the data for the Linear Regression model, a `ColumnTransformer` handles:
1.  **Standard Scaling** (`StandardScaler`) for numerical features.
2.  **One-Hot Encoding** (`OneHotEncoder`) for categorical features.

### Model Persistence
The complete training pipeline (preprocessor + model) is saved as: **`severity_linear_regression_model.pkl`**.
This file allows the model to be loaded and used for immediate predictions without retraining.

This model provides a **data-driven framework** for traffic safety:

1.  **Resource Prioritization:** By analyzing the model's coefficients, authorities can identify the specific, high-impact factors (e.g., lack of streetlights, presence of certain junction types) that statistically increase accident severity. This ensures **scarce funds** are allocated to the most critical interventions for maximum public safety benefit.
2.  **Proactive Auditing:** The model can be run on data from new or existing road designs to generate a **predictive severity score**, allowing engineers to flag and correct high-risk areas **before** serious accidents occur.
