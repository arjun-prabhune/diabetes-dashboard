"""
Configuration file for Diabetes Risk Dashboard
Contains all configurable parameters and settings
"""

import os

# ============================================================================
# FILE PATHS
# ============================================================================

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data paths
DATA_DIR = os.path.join(BASE_DIR, 'data')
DATA_FILE = os.path.join(DATA_DIR, 'diabetes_data.csv')

# Model paths
MODEL_DIR = os.path.join(BASE_DIR, 'models')
MODEL_FILE = os.path.join(MODEL_DIR, 'diabetes_risk_regressor.pkl')
FEATURE_IMPORTANCE_PLOT = os.path.join(MODEL_DIR, 'feature_importance.png')

# ============================================================================
# MODEL PARAMETERS
# ============================================================================

# Random Forest Configuration
RF_PARAMS = {
    'n_estimators': 100,
    'max_depth': 20,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'random_state': 42,
    'n_jobs': -1
}

# Train-test split
TEST_SIZE = 0.2
RANDOM_STATE = 42

# ============================================================================
# DATA GENERATION PARAMETERS (for synthetic data)
# ============================================================================

# Number of synthetic samples to generate
N_SYNTHETIC_SAMPLES = 10000

# ============================================================================
# FEATURE DEFINITIONS
# ============================================================================

# Categorical features
CATEGORICAL_FEATURES = [
    'gender',
    'ethnicity',
    'education_level',
    'income_level',
    'employment_status',
    'smoking_status'
]

# Numerical features
NUMERICAL_FEATURES = [
    'age',
    'alcohol_consumption_per_week',
    'physical_activity_minutes_per_week',
    'diet_score',
    'sleep_hours_per_day',
    'screen_time_hours_per_day',
    'family_history_diabetes',
    'hypertension_history',
    'cardiovascular_history',
    'bmi',
    'waist_to_hip_ratio',
    'systolic_bp',
    'diastolic_bp',
    'heart_rate',
    'cholesterol_total',
    'hdl_cholesterol',
    'ldl_cholesterol',
    'triglycerides',
    'glucose_fasting',
    'glucose_postprandial',
    'insulin_level',
    'hba1c'
]

# Target variable
TARGET_VARIABLE = 'diabetes_risk_score'

# ID column (to be excluded from training)
ID_COLUMN = 'patient_id'

# ============================================================================
# FEATURE VALUE RANGES
# ============================================================================

FEATURE_RANGES = {
    'age': (18, 90),
    'alcohol_consumption_per_week': (0, 30),
    'physical_activity_minutes_per_week': (0, 600),
    'diet_score': (0, 10),
    'sleep_hours_per_day': (3, 12),
    'screen_time_hours_per_day': (0, 12),
    'family_history_diabetes': (0, 1),
    'hypertension_history': (0, 1),
    'cardiovascular_history': (0, 1),
    'bmi': (15, 45),
    'waist_to_hip_ratio': (0.7, 1.2),
    'systolic_bp': (90, 180),
    'diastolic_bp': (60, 120),
    'heart_rate': (50, 120),
    'cholesterol_total': (120, 300),
    'hdl_cholesterol': (20, 100),
    'ldl_cholesterol': (50, 200),
    'triglycerides': (50, 500),
    'glucose_fasting': (70, 250),
    'glucose_postprandial': (90, 350),
    'insulin_level': (2, 50),
    'hba1c': (4, 14)
}

# Categorical feature options
CATEGORICAL_OPTIONS = {
    'gender': ['Male', 'Female', 'Other'],
    'ethnicity': ['White', 'Hispanic', 'Black', 'Asian', 'Other'],
    'education_level': ['No formal', 'Highschool', 'Graduate', 'Postgraduate'],
    'income_level': ['Low', 'Medium', 'High'],
    'employment_status': ['Employed', 'Unemployed', 'Retired', 'Student'],
    'smoking_status': ['Never', 'Former', 'Current']
}

# ============================================================================
# RISK THRESHOLDS
# ============================================================================

RISK_THRESHOLDS = {
    'low': 30,      # Score < 30 = Low Risk
    'moderate': 60  # Score 30-60 = Moderate Risk, >60 = High Risk
}

# ============================================================================
# CLINICAL THRESHOLDS (for recommendations)
# ============================================================================

CLINICAL_THRESHOLDS = {
    # BMI categories
    'bmi_underweight': 18.5,
    'bmi_normal': 25,
    'bmi_overweight': 30,
    
    # Blood glucose (mg/dL)
    'glucose_fasting_normal': 100,
    'glucose_fasting_prediabetes': 125,
    
    # HbA1c (%)
    'hba1c_normal': 5.7,
    'hba1c_prediabetes': 6.5,
    
    # Physical activity (min/week)
    'physical_activity_minimum': 150,
    'physical_activity_optimal': 300,
    
    # Sleep (hours/day)
    'sleep_minimum': 6,
    'sleep_optimal_min': 7,
    'sleep_optimal_max': 9,
    
    # Diet score (0-10)
    'diet_poor': 4,
    'diet_fair': 7,
    
    # Blood pressure (mmHg)
    'systolic_bp_normal': 120,
    'systolic_bp_elevated': 140,
    'diastolic_bp_normal': 80,
    'diastolic_bp_elevated': 90,
    
    # Cholesterol (mg/dL)
    'ldl_optimal': 100,
    'ldl_high': 130,
    'hdl_low': 40,
    'hdl_optimal': 60,
    
    # Screen time (hours/day)
    'screen_time_high': 6,
    
    # Alcohol (drinks/week)
    'alcohol_moderate_limit': 14
}

# ============================================================================
# VISUALIZATION SETTINGS
# ============================================================================

# Color scheme for risk categories
RISK_COLORS = {
    'low': '#28a745',       # Green
    'moderate': '#ffc107',  # Yellow/Amber
    'high': '#dc3545'       # Red
}

# Plotly color scale
PLOTLY_COLOR_SCALE = 'Blues'

# Feature importance plot settings
FEATURE_IMPORTANCE_TOP_N = 15
FIGURE_DPI = 150

# ============================================================================
# STREAMLIT UI SETTINGS
# ============================================================================

# Page configuration
PAGE_CONFIG = {
    'page_title': 'Diabetes Risk Dashboard',
    'page_icon': '🏥',
    'layout': 'wide',
    'initial_sidebar_state': 'expanded'
}

# UI text
UI_TEXT = {
    'app_title': '🏥 Personal Diabetes Risk Dashboard',
    'app_subtitle': 'AI-Powered Diabetes Risk Assessment with Personalized Recommendations',
    'predict_button': '🔍 Calculate Risk Score',
    'low_risk_desc': 'Your diabetes risk is low. Keep up your healthy habits!',
    'moderate_risk_desc': 'Your diabetes risk is moderate. Consider lifestyle improvements.',
    'high_risk_desc': 'Your diabetes risk is high. Please consult a healthcare provider.',
}

# ============================================================================
# RECOMMENDATION TEMPLATES
# ============================================================================

RECOMMENDATIONS = {
    'bmi_obese': "🏃 **Weight Management**: Your BMI indicates obesity. Consider working with a nutritionist and increasing physical activity to achieve a healthier weight.",
    'bmi_overweight': "⚖️ **Weight Management**: Your BMI is in the overweight range. Gradual weight loss through balanced diet and exercise can significantly reduce diabetes risk.",
    'glucose_high': "🩸 **Blood Sugar Control**: Your fasting glucose is elevated (>125 mg/dL). Consult your doctor immediately and consider reducing sugar intake and refined carbohydrates.",
    'glucose_prediabetes': "🍽️ **Diet Monitoring**: Your fasting glucose is in the prediabetic range (100-125 mg/dL). Focus on low-glycemic foods, increase fiber intake, and monitor your blood sugar regularly.",
    'hba1c_diabetes': "⚠️ **HbA1c Alert**: Your HbA1c level (>6.5%) indicates diabetes. Please consult a healthcare provider for proper diagnosis and treatment plan.",
    'hba1c_prediabetes': "📊 **HbA1c Monitoring**: Your HbA1c (5.7-6.5%) suggests prediabetes. Regular monitoring and lifestyle changes can help prevent progression to type 2 diabetes.",
    'exercise_low': "💪 **Exercise More**: You're getting less than 150 minutes of physical activity per week. Aim for at least 30 minutes of moderate exercise 5 days a week to improve insulin sensitivity.",
    'sleep_insufficient': "😴 **Improve Sleep**: You're sleeping less than 6 hours per day. Poor sleep increases diabetes risk. Aim for 7-9 hours of quality sleep each night.",
    'sleep_excessive': "⏰ **Sleep Balance**: You're sleeping more than 9 hours daily. Both too little and too much sleep are associated with increased diabetes risk. Aim for 7-9 hours.",
    'diet_poor': "🥗 **Improve Diet Quality**: Your diet score is low. Focus on whole grains, vegetables, fruits, lean proteins, and healthy fats. Reduce processed foods and added sugars.",
    'smoking_current': "🚭 **Quit Smoking**: Smoking significantly increases diabetes risk and complications. Consider smoking cessation programs and speak with your doctor about quitting strategies.",
    'bp_high': "💉 **Blood Pressure Control**: Your blood pressure is elevated. High blood pressure increases diabetes risk. Reduce sodium intake, manage stress, and consult your doctor.",
    'ldl_high': "🫀 **Cholesterol Management**: Your LDL cholesterol is high. Consider heart-healthy diet changes, regular exercise, and discuss treatment options with your doctor.",
    'hdl_low': "❤️ **Increase HDL Cholesterol**: Your HDL (good cholesterol) is low. Regular aerobic exercise and healthy fats (omega-3s) can help raise HDL levels.",
    'family_history': "👨‍👩‍👧‍👦 **Family History Alert**: You have a family history of diabetes. Regular screening and proactive lifestyle management are especially important for you.",
    'screen_time_high': "📱 **Reduce Screen Time**: Excessive screen time (>6 hours) is associated with sedentary behavior. Take regular breaks, use standing desks, and incorporate movement into your day.",
    'alcohol_high': "🍷 **Moderate Alcohol**: You're consuming more than recommended amounts of alcohol. Excessive alcohol can affect blood sugar control. Consider reducing intake.",
    'maintain_healthy': "✅ **Maintain Healthy Habits**: Your current health metrics are good! Continue with regular exercise, balanced diet, and routine health check-ups.",
    'regular_monitoring': "📅 **Regular Monitoring**: Even with low risk, get annual health screenings including blood glucose, blood pressure, and cholesterol levels.",
    'stay_informed': "🎯 **Stay Informed**: Stay educated about diabetes prevention and maintain your healthy lifestyle choices."
}

# ============================================================================
# MEDICAL DISCLAIMER
# ============================================================================

MEDICAL_DISCLAIMER = """
⚠️ **Medical Disclaimer**: This tool provides risk assessment for educational purposes only. 
It is NOT a substitute for professional medical advice, diagnosis, or treatment. 
Always consult with qualified healthcare providers regarding your health conditions and concerns.
"""

# ============================================================================
# LOGGING SETTINGS
# ============================================================================

LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_all_features():
    """Return list of all feature names"""
    return CATEGORICAL_FEATURES + NUMERICAL_FEATURES

def get_feature_count():
    """Return total number of features"""
    return len(CATEGORICAL_FEATURES) + len(NUMERICAL_FEATURES)

def create_directories():
    """Create necessary directories if they don't exist"""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    print(f"✓ Directories created/verified: {DATA_DIR}, {MODEL_DIR}")

if __name__ == "__main__":
    # Print configuration summary
    print("="*60)
    print("DIABETES RISK DASHBOARD - CONFIGURATION SUMMARY")
    print("="*60)
    print(f"Total Features: {get_feature_count()}")
    print(f"  - Categorical: {len(CATEGORICAL_FEATURES)}")
    print(f"  - Numerical: {len(NUMERICAL_FEATURES)}")
    print(f"\nData File: {DATA_FILE}")
    print(f"Model File: {MODEL_FILE}")
    print(f"\nRisk Thresholds:")
    print(f"  - Low Risk: < {RISK_THRESHOLDS['low']}")
    print(f"  - Moderate Risk: {RISK_THRESHOLDS['low']}-{RISK_THRESHOLDS['moderate']}")
    print(f"  - High Risk: > {RISK_THRESHOLDS['moderate']}")
    print("="*60)
    
    # Create directories
    create_directories()