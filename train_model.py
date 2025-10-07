"""
Personal Diabetes Risk Dashboard - Model Training Script
This script trains a RandomForestRegressor to predict diabetes risk scores (0-100)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
import os

# Set random seed for reproducibility
np.random.seed(42)

def generate_synthetic_data(n_samples=10000):
    """
    Generate synthetic diabetes risk dataset for demonstration
    In production, replace this with actual data loading from CSV
    """
    print("Generating synthetic dataset...")
    
    data = {
        'age': np.random.randint(18, 91, n_samples),
        'gender': np.random.choice(['Male', 'Female', 'Other'], n_samples, p=[0.48, 0.48, 0.04]),
        'ethnicity': np.random.choice(['White', 'Hispanic', 'Black', 'Asian', 'Other'], n_samples),
        'education_level': np.random.choice(['No formal', 'Highschool', 'Graduate', 'Postgraduate'], n_samples),
        'income_level': np.random.choice(['Low', 'Medium', 'High'], n_samples),
        'employment_status': np.random.choice(['Employed', 'Unemployed', 'Retired', 'Student'], n_samples),
        'smoking_status': np.random.choice(['Never', 'Former', 'Current'], n_samples, p=[0.6, 0.25, 0.15]),
        'alcohol_consumption_per_week': np.random.uniform(0, 30, n_samples),
        'physical_activity_minutes_per_week': np.random.randint(0, 601, n_samples),
        'diet_score': np.random.randint(0, 11, n_samples),
        'sleep_hours_per_day': np.random.uniform(3, 12, n_samples),
        'screen_time_hours_per_day': np.random.uniform(0, 12, n_samples),
        'family_history_diabetes': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        'hypertension_history': np.random.choice([0, 1], n_samples, p=[0.75, 0.25]),
        'cardiovascular_history': np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
        'bmi': np.random.uniform(15, 45, n_samples),
        'waist_to_hip_ratio': np.random.uniform(0.7, 1.2, n_samples),
        'systolic_bp': np.random.randint(90, 181, n_samples),
        'diastolic_bp': np.random.randint(60, 121, n_samples),
        'heart_rate': np.random.randint(50, 121, n_samples),
        'cholesterol_total': np.random.uniform(120, 300, n_samples),
        'hdl_cholesterol': np.random.uniform(20, 100, n_samples),
        'ldl_cholesterol': np.random.uniform(50, 200, n_samples),
        'triglycerides': np.random.uniform(50, 500, n_samples),
        'glucose_fasting': np.random.uniform(70, 250, n_samples),
        'glucose_postprandial': np.random.uniform(90, 350, n_samples),
        'insulin_level': np.random.uniform(2, 50, n_samples),
        'hba1c': np.random.uniform(4, 14, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Create realistic diabetes risk score based on key features
    risk_score = (
        df['age'] * 0.3 +
        df['bmi'] * 1.5 +
        df['glucose_fasting'] * 0.2 +
        df['hba1c'] * 4 +
        df['family_history_diabetes'] * 15 +
        (10 - df['diet_score']) * 2 +
        (600 - df['physical_activity_minutes_per_week']) * 0.05 +
        df['hypertension_history'] * 8 +
        df['cardiovascular_history'] * 10 +
        np.random.normal(0, 5, n_samples)  # Add some noise
    )
    
    # Normalize to 0-100 range
    df['diabetes_risk_score'] = np.clip(risk_score, 0, 100).astype(int)
    
    return df

def load_or_generate_data(filepath='data/diabetes_data.csv'):
    """Load data from CSV or generate synthetic data if file doesn't exist"""
    if os.path.exists(filepath):
        print(f"Loading data from {filepath}...")
        return pd.read_csv(filepath)
    else:
        print(f"Data file not found at {filepath}. Generating synthetic data...")
        df = generate_synthetic_data()
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_csv(filepath, index=False)
        print(f"Synthetic data saved to {filepath}")
        return df

def preprocess_data(df):
    """
    Preprocess the dataset:
    - Handle missing values
    - Separate features and target
    - Split into train and test sets
    """
    print("\nPreprocessing data...")
    
    # Drop patient_id as it's not a feature
    X = df.drop(['diabetes_risk_score', 'diagnosed_diabetes', 'diabetes_stage'], axis=1)
    y = df['diabetes_risk_score']
    
    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    print(f"Categorical features: {categorical_cols}")
    print(f"Numerical features: {numerical_cols}")
    
    # Handle missing values (fill with mean for numeric, mode for categorical)
    for col in numerical_cols:
        if X[col].isnull().any():
            X[col].fillna(X[col].mean(), inplace=True)
    
    for col in categorical_cols:
        if X[col].isnull().any():
            X[col].fillna(X[col].mode()[0], inplace=True)
    
    # Split data: 80% train, 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    return X_train, X_test, y_train, y_test, categorical_cols, numerical_cols

def create_preprocessing_pipeline(categorical_cols, numerical_cols):
    """
    Create preprocessing pipeline with encoding and scaling
    """
    # Create transformers for categorical and numerical features
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    numerical_transformer = StandardScaler()
    
    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )
    
    return preprocessor

def train_model(X_train, y_train, preprocessor):
    """
    Train RandomForestRegressor model
    """
    print("\nTraining RandomForestRegressor...")
    
    # Create pipeline with preprocessing and model
    model_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ))
    ])
    
    # Train the model
    model_pipeline.fit(X_train, y_train)
    
    print("Model training completed!")
    
    return model_pipeline

def evaluate_model(model, X_train, X_test, y_train, y_test):
    """
    Evaluate model performance on train and test sets
    """
    print("\nEvaluating model performance...")
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    # Print results
    print("\n" + "="*50)
    print("MODEL PERFORMANCE METRICS")
    print("="*50)
    print(f"Train R²: {train_r2:.4f}")
    print(f"Test R²: {test_r2:.4f}")
    print(f"\nTrain MAE: {train_mae:.2f}")
    print(f"Test MAE: {test_mae:.2f}")
    print(f"\nTrain RMSE: {train_rmse:.2f}")
    print(f"Test RMSE: {test_rmse:.2f}")
    print("="*50)
    
    return y_test_pred

def plot_feature_importance(model, feature_names, top_n=15):
    """
    Plot feature importance from the trained model
    """
    # Get feature importances from the regressor
    regressor = model.named_steps['regressor']
    importances = regressor.feature_importances_
    
    # Create feature names after preprocessing
    preprocessor = model.named_steps['preprocessor']
    cat_features = preprocessor.named_transformers_['cat'].get_feature_names_out()
    num_features = preprocessor.named_transformers_['num'].get_feature_names_out()
    all_features = list(num_features) + list(cat_features)
    
    # Create dataframe of feature importances
    feat_imp_df = pd.DataFrame({
        'feature': all_features,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    # Plot top N features
    plt.figure(figsize=(10, 6))
    top_features = feat_imp_df.head(top_n)
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Importance')
    plt.title(f'Top {top_n} Feature Importances')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    # Save plot
    os.makedirs('models', exist_ok=True)
    plt.savefig('models/feature_importance.png', dpi=150, bbox_inches='tight')
    print("\nFeature importance plot saved to models/feature_importance.png")
    plt.close()
    
    return feat_imp_df

def save_model(model, filepath='models/diabetes_risk_regressor.pkl'):
    """
    Save trained model to disk using joblib
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    print(f"\nModel saved to {filepath}")

def main():
    """
    Main training pipeline
    """
    print("="*50)
    print("DIABETES RISK PREDICTION MODEL TRAINING")
    print("="*50)
    
    # Load data
    df = load_or_generate_data()
    
    # Preprocess data
    X_train, X_test, y_train, y_test, categorical_cols, numerical_cols = preprocess_data(df)
    
    # Create preprocessing pipeline
    preprocessor = create_preprocessing_pipeline(categorical_cols, numerical_cols)
    
    # Train model
    model = train_model(X_train, y_train, preprocessor)
    
    # Evaluate model
    y_test_pred = evaluate_model(model, X_train, X_test, y_train, y_test)
    
    # Plot feature importance
    feature_names = list(numerical_cols) + list(categorical_cols)
    feat_imp_df = plot_feature_importance(model, feature_names)
    
    # Save model
    save_model(model)
    
    print("\n" + "="*50)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*50)
    print("\nNext step: Run 'streamlit run app.py' to launch the dashboard")

if __name__ == "__main__":
    main()