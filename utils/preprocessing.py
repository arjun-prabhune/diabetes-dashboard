"""
Preprocessing utilities for the Diabetes Risk Dashboard
Contains helper functions for data transformation and validation
"""

import pandas as pd
import numpy as np

def validate_input_ranges(input_data):
    """
    Validate that user inputs are within acceptable ranges
    
    Args:
        input_data (dict): Dictionary of user input values
    
    Returns:
        tuple: (is_valid, error_messages)
    """
    errors = []
    
    # Define valid ranges
    ranges = {
        'age': (18, 90),
        'alcohol_consumption_per_week': (0, 30),
        'physical_activity_minutes_per_week': (0, 600),
        'diet_score': (0, 10),
        'sleep_hours_per_day': (3, 12),
        'screen_time_hours_per_day': (0, 12),
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
    
    # Check each numeric field
    for field, (min_val, max_val) in ranges.items():
        if field in input_data:
            value = input_data[field]
            if value < min_val or value > max_val:
                errors.append(f"{field}: {value} is outside valid range [{min_val}, {max_val}]")
    
    return len(errors) == 0, errors

def create_input_dataframe(input_data):
    """
    Convert user input dictionary to a pandas DataFrame with correct column order
    
    Args:
        input_data (dict): Dictionary of user input values
    
    Returns:
        pd.DataFrame: Single-row dataframe ready for prediction
    """
    # Define the expected column order (excluding patient_id and diabetes_risk_score)
    expected_columns = [
        'age', 'gender', 'ethnicity', 'education_level', 'income_level',
        'employment_status', 'smoking_status', 'alcohol_consumption_per_week',
        'physical_activity_minutes_per_week', 'diet_score', 'sleep_hours_per_day',
        'screen_time_hours_per_day', 'family_history_diabetes', 'hypertension_history',
        'cardiovascular_history', 'bmi', 'waist_to_hip_ratio', 'systolic_bp',
        'diastolic_bp', 'heart_rate', 'cholesterol_total', 'hdl_cholesterol',
        'ldl_cholesterol', 'triglycerides', 'glucose_fasting', 'glucose_postprandial',
        'insulin_level', 'hba1c'
    ]
    
    # Create DataFrame with single row
    df = pd.DataFrame([input_data], columns=expected_columns)
    
    return df

def get_risk_category(risk_score):
    """
    Categorize risk score into Low, Moderate, or High
    
    Args:
        risk_score (float): Predicted diabetes risk score (0-100)
    
    Returns:
        tuple: (category, color, description)
    """
    if risk_score < 30:
        return "Low Risk", "#28a745", "Your diabetes risk is low. Keep up your healthy habits!"
    elif risk_score < 60:
        return "Moderate Risk", "#ffc107", "Your diabetes risk is moderate. Consider lifestyle improvements."
    else:
        return "High Risk", "#dc3545", "Your diabetes risk is high. Please consult a healthcare provider."

def calculate_bmi_category(bmi):
    """
    Categorize BMI value
    
    Args:
        bmi (float): Body Mass Index
    
    Returns:
        str: BMI category
    """
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25:
        return "Normal weight"
    elif bmi < 30:
        return "Overweight"
    else:
        return "Obese"

def generate_recommendations(input_data, risk_score):
    """
    Generate personalized health recommendations based on user input and risk factors
    
    Args:
        input_data (dict): Dictionary of user input values
        risk_score (float): Predicted diabetes risk score
    
    Returns:
        list: List of recommendation strings
    """
    recommendations = []
    
    # BMI recommendations
    if input_data['bmi'] > 30:
        recommendations.append(
            "🏃 **Weight Management**: Your BMI indicates obesity. Consider working with a "
            "nutritionist and increasing physical activity to achieve a healthier weight."
        )
    elif input_data['bmi'] > 25:
        recommendations.append(
            "⚖️ **Weight Management**: Your BMI is in the overweight range. Gradual weight "
            "loss through balanced diet and exercise can significantly reduce diabetes risk."
        )
    
    # Blood glucose recommendations
    if input_data['glucose_fasting'] > 125:
        recommendations.append(
            "🩸 **Blood Sugar Control**: Your fasting glucose is elevated (>125 mg/dL). "
            "Consult your doctor immediately and consider reducing sugar intake and refined carbohydrates."
        )
    elif input_data['glucose_fasting'] > 100:
        recommendations.append(
            "🍽️ **Diet Monitoring**: Your fasting glucose is in the prediabetic range (100-125 mg/dL). "
            "Focus on low-glycemic foods, increase fiber intake, and monitor your blood sugar regularly."
        )
    
    # HbA1c recommendations
    if input_data['hba1c'] > 6.5:
        recommendations.append(
            "⚠️ **HbA1c Alert**: Your HbA1c level (>6.5%) indicates diabetes. Please consult "
            "a healthcare provider for proper diagnosis and treatment plan."
        )
    elif input_data['hba1c'] > 5.7:
        recommendations.append(
            "📊 **HbA1c Monitoring**: Your HbA1c (5.7-6.5%) suggests prediabetes. Regular monitoring "
            "and lifestyle changes can help prevent progression to type 2 diabetes."
        )
    
    # Physical activity recommendations
    if input_data['physical_activity_minutes_per_week'] < 150:
        recommendations.append(
            "💪 **Exercise More**: You're getting less than 150 minutes of physical activity per week. "
            "Aim for at least 30 minutes of moderate exercise 5 days a week to improve insulin sensitivity."
        )
    
    # Sleep recommendations
    if input_data['sleep_hours_per_day'] < 6:
        recommendations.append(
            "😴 **Improve Sleep**: You're sleeping less than 6 hours per day. Poor sleep increases "
            "diabetes risk. Aim for 7-9 hours of quality sleep each night."
        )
    elif input_data['sleep_hours_per_day'] > 9:
        recommendations.append(
            "⏰ **Sleep Balance**: You're sleeping more than 9 hours daily. Both too little and too "
            "much sleep are associated with increased diabetes risk. Aim for 7-9 hours."
        )
    
    # Diet recommendations
    if input_data['diet_score'] < 5:
        recommendations.append(
            "🥗 **Improve Diet Quality**: Your diet score is low. Focus on whole grains, vegetables, "
            "fruits, lean proteins, and healthy fats. Reduce processed foods and added sugars."
        )
    
    # Smoking recommendations
    if input_data['smoking_status'] == 'Current':
        recommendations.append(
            "🚭 **Quit Smoking**: Smoking significantly increases diabetes risk and complications. "
            "Consider smoking cessation programs and speak with your doctor about quitting strategies."
        )
    
    # Blood pressure recommendations
    if input_data['systolic_bp'] > 140 or input_data['diastolic_bp'] > 90:
        recommendations.append(
            "💉 **Blood Pressure Control**: Your blood pressure is elevated. High blood pressure "
            "increases diabetes risk. Reduce sodium intake, manage stress, and consult your doctor."
        )
    
    # Cholesterol recommendations
    if input_data['ldl_cholesterol'] > 130:
        recommendations.append(
            "🫀 **Cholesterol Management**: Your LDL cholesterol is high. Consider heart-healthy "
            "diet changes, regular exercise, and discuss treatment options with your doctor."
        )
    
    if input_data['hdl_cholesterol'] < 40:
        recommendations.append(
            "❤️ **Increase HDL Cholesterol**: Your HDL (good cholesterol) is low. Regular aerobic "
            "exercise and healthy fats (omega-3s) can help raise HDL levels."
        )
    
    # Family history recommendations
    if input_data['family_history_diabetes'] == 1:
        recommendations.append(
            "👨‍👩‍👧‍👦 **Family History Alert**: You have a family history of diabetes. Regular screening "
            "and proactive lifestyle management are especially important for you."
        )
    
    # Screen time recommendations
    if input_data['screen_time_hours_per_day'] > 6:
        recommendations.append(
            "📱 **Reduce Screen Time**: Excessive screen time (>6 hours) is associated with sedentary "
            "behavior. Take regular breaks, use standing desks, and incorporate movement into your day."
        )
    
    # Alcohol recommendations
    if input_data['alcohol_consumption_per_week'] > 14:
        recommendations.append(
            "🍷 **Moderate Alcohol**: You're consuming more than recommended amounts of alcohol. "
            "Excessive alcohol can affect blood sugar control. Consider reducing intake."
        )
    
    # If no specific recommendations, provide general advice
    if len(recommendations) == 0:
        recommendations.append(
            "✅ **Maintain Healthy Habits**: Your current health metrics are good! Continue with "
            "regular exercise, balanced diet, and routine health check-ups."
        )
        recommendations.append(
            "📅 **Regular Monitoring**: Even with low risk, get annual health screenings including "
            "blood glucose, blood pressure, and cholesterol levels."
        )
        recommendations.append(
            "🎯 **Stay Informed**: Stay educated about diabetes prevention and maintain your "
            "healthy lifestyle choices."
        )
    
    # Limit to top 5 recommendations
    return recommendations[:5]

def get_feature_interpretation(feature_name, value):
    """
    Provide human-readable interpretation of feature values
    
    Args:
        feature_name (str): Name of the feature
        value: Value of the feature
    
    Returns:
        str: Interpretation string
    """
    interpretations = {
        'bmi': f"BMI of {value:.1f} - {calculate_bmi_category(value)}",
        'glucose_fasting': f"{value:.0f} mg/dL - {'Normal' if value < 100 else 'Prediabetes' if value < 126 else 'Diabetes range'}",
        'hba1c': f"{value:.1f}% - {'Normal' if value < 5.7 else 'Prediabetes' if value < 6.5 else 'Diabetes range'}",
        'physical_activity_minutes_per_week': f"{value} minutes/week - {'Below recommended' if value < 150 else 'Good' if value < 300 else 'Excellent'}",
        'sleep_hours_per_day': f"{value:.1f} hours - {'Insufficient' if value < 7 else 'Optimal' if value <= 9 else 'Excessive'}",
        'diet_score': f"{value}/10 - {'Poor' if value < 4 else 'Fair' if value < 7 else 'Good'}",
        'systolic_bp': f"{value} mmHg - {'Normal' if value < 120 else 'Elevated' if value < 130 else 'High'}",
        'diastolic_bp': f"{value} mmHg - {'Normal' if value < 80 else 'Elevated' if value < 90 else 'High'}",
    }
    
    return interpretations.get(feature_name, f"{value}")