import pandas as pd
import joblib
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import warnings

# Ignore warnings
warnings.filterwarnings("ignore")

# Load the original dataset
df = pd.read_csv('ExtraaLearn.csv')

# Preprocessing function
def preprocess_data(df, preprocessor):
    # Drop irrelevant columns
    df = df.drop(columns=['ID'])
    
    # Identify categorical and numerical columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    numerical_cols = numerical_cols.drop('status')

    # Transform the data using the loaded preprocessor
    X = df.drop(columns=['status'])
    y = df['status']
    X = preprocessor.transform(X)
    return X, y

# Load the preprocessor and trained model
preprocessor = joblib.load('models/preprocessor.pkl')
model = joblib.load('models/xgb_model.pkl')

# Preprocess the data
X, y = preprocess_data(df, preprocessor)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Make predictions using the loaded model on the test set
y_pred_xgb = model.predict(X_test)

# Print classification report
print("Classification Report:\n", classification_report(y_test, y_pred_xgb))

# ROC-AUC Score
roc_auc_xgb = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
print("ROC-AUC Score (XGBoost):", roc_auc_xgb)

# Take the first few rows (e.g., first 5) for prediction
X_first_rows = X[:5]  # Taking first 5 rows as input for prediction

# Predict for the first rows
y_first_pred = model.predict(X_first_rows)

# Output the predictions along with the actual values for the first few rows
print("\nPredictions for the first 5 rows:")
for i in range(len(X_first_rows)):
    print(f"Row {i+1} - Predicted: {y_first_pred[i]}")

