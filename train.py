import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import joblib
import warnings

# Ignore warnings
warnings.filterwarnings("ignore")

# Load the original dataset
df = pd.read_csv('ExtraaLearn.csv')

# Preprocessing function
def preprocess_data(df):
    # Drop irrelevant columns
    df = df.drop(columns=['ID'])
    
    # Identify categorical and numerical columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    numerical_cols = numerical_cols.drop('status')

    # OneHotEncoder for categorical variables
    preprocessor = ColumnTransformer(
        transformers=[('cat', OneHotEncoder(drop='first'), categorical_cols)],
        remainder='passthrough'
    )

    # Transform the data
    X = df.drop(columns=['status'])
    y = df['status']
    X = preprocessor.fit_transform(X)
    return X, y, preprocessor

# Preprocess the data
X, y, preprocessor = preprocess_data(df)

# Apply SMOTE for handling imbalance
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# Initialize XGBoost Classifier
xgb = XGBClassifier(random_state=42)

# Hyperparameter space for tuning
param_dist = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [10, 20, 30, None],
    'learning_rate': [0.001, 0.01, 0.1, 0.2],
    'min_child_weight': [1, 2, 3, 4],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
    'gamma': [0, 0.1, 0.2, 0.3],
    'scale_pos_weight': [1, 2, 3]
}

# RandomizedSearchCV for hyperparameter tuning
random_search_xgb = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=param_dist,
    n_iter=100,
    cv=5,
    verbose=2,
    random_state=42,
    n_jobs=-1
)

# Fit the model
random_search_xgb.fit(X_train, y_train)

# Print best hyperparameters
print("Best Hyperparameters (XGBoost):", random_search_xgb.best_params_)

# Make predictions on the test set
y_pred = random_search_xgb.predict(X_test)

# Print evaluation metrics
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1-Score:", f1_score(y_test, y_pred))
print("AUC-ROC Score:", roc_auc_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save the trained model
joblib.dump(random_search_xgb.best_estimator_, 'models/xgb_model.pkl')
print("XGBoost model saved successfully!")

# Save the preprocessor to be used during prediction
joblib.dump(preprocessor, 'models/preprocessor.pkl')
print("Preprocessor saved successfully!")
