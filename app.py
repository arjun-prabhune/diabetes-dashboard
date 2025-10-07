"""
Personal Diabetes Risk Dashboard - Streamlit Application
A comprehensive web application for predicting diabetes risk and providing personalized recommendations
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px

# Add utils directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))
from preprocessing import (
    validate_input_ranges, 
    create_input_dataframe, 
    get_risk_category,
    generate_recommendations,
    get_feature_interpretation
)

# Page configuration
st.set_page_config(
    page_title="Diabetes Risk Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .risk-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin: 20px 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    .recommendation {
        background-color: #e7f3ff;
        padding: 15px;
        border-left: 4px solid #1f77b4;
        margin: 10px 0;
        border-radius: 4px;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained model from disk"""
    model_path = 'models/diabetes_risk_regressor.pkl'
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}. Please run train_model.py first.")
        st.stop()
    return joblib.load(model_path)

def create_gauge_chart(risk_score):
    """Create a gauge chart for risk score visualization"""
    # Determine color based on risk level
    if risk_score < 30:
        color = "#28a745"
    elif risk_score < 60:
        color = "#ffc107"
    else:
        color = "#dc3545"
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = risk_score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Diabetes Risk Score", 'font': {'size': 24}},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': '#d4edda'},
                {'range': [30, 60], 'color': '#fff3cd'},
                {'range': [60, 100], 'color': '#f8d7da'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))
    
    fig.update_layout(
        height=350,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="white",
        font={'color': "darkblue", 'family': "Arial"}
    )
    
    return fig

def create_feature_importance_chart(model, input_df):
    """Create feature importance visualization"""
    try:
        # Get the regressor from pipeline
        regressor = model.named_steps['regressor']
        
        # Get feature names after preprocessing
        preprocessor = model.named_steps['preprocessor']
        
        # Transform the input to get feature names
        input_transformed = preprocessor.transform(input_df)
        
        # Get feature names
        cat_features = preprocessor.named_transformers_['cat'].get_feature_names_out()
        num_features = preprocessor.named_transformers_['num'].get_feature_names_out()
        all_features = list(num_features) + list(cat_features)
        
        # Get feature importances
        importances = regressor.feature_importances_
        
        # Create dataframe
        feat_imp_df = pd.DataFrame({
            'Feature': all_features,
            'Importance': importances
        }).sort_values('Importance', ascending=False).head(15)
        
        # Create bar chart
        fig = px.bar(
            feat_imp_df, 
            x='Importance', 
            y='Feature',
            orientation='h',
            title='Top 15 Feature Importances',
            color='Importance',
            color_continuous_scale='Blues'
        )
        
        fig.update_layout(
            height=500,
            showlegend=False,
            yaxis={'categoryorder': 'total ascending'}
        )
        
        return fig
    except Exception as e:
        st.warning(f"Could not generate feature importance chart: {str(e)}")
        return None

def create_shap_explanation(model, input_df):
    """Create SHAP explanation plot"""
    try:
        import shap
        
        # Get the regressor from pipeline
        regressor = model.named_steps['regressor']
        preprocessor = model.named_steps['preprocessor']
        
        # Transform input data
        input_transformed = preprocessor.transform(input_df)
        
        # Create SHAP explainer
        explainer = shap.TreeExplainer(regressor)
        shap_values = explainer.shap_values(input_transformed)
        
        # Get feature names
        cat_features = preprocessor.named_transformers_['cat'].get_feature_names_out()
        num_features = preprocessor.named_transformers_['num'].get_feature_names_out()
        all_features = list(num_features) + list(cat_features)
        
        # Create force plot
        fig, ax = plt.subplots(figsize=(12, 3))
        shap.plots.waterfall(
            shap.Explanation(
                values=shap_values[0],
                base_values=explainer.expected_value,
                data=input_transformed[0],
                feature_names=all_features
            ),
            max_display=10,
            show=False
        )
        
        return fig
    except ImportError:
        st.info("Install SHAP library for advanced explanations: pip install shap")
        return None
    except Exception as e:
        st.warning(f"Could not generate SHAP explanation: {str(e)}")
        return None

def sidebar_inputs():
    """Create sidebar for user input collection"""
    st.sidebar.title("📋 Patient Information")
    st.sidebar.markdown("---")
    
    input_data = {}
    
    # Demographic Information
    st.sidebar.subheader("👤 Demographics")
    input_data['age'] = st.sidebar.slider("Age", 18, 90, 45, help="Your current age in years")
    input_data['gender'] = st.sidebar.selectbox("Gender", ['Male', 'Female', 'Other'])
    input_data['ethnicity'] = st.sidebar.selectbox(
        "Ethnicity", 
        ['White', 'Hispanic', 'Black', 'Asian', 'Other']
    )
    input_data['education_level'] = st.sidebar.selectbox(
        "Education Level",
        ['No formal', 'Highschool', 'Graduate', 'Postgraduate']
    )
    input_data['income_level'] = st.sidebar.selectbox("Income Level", ['Low', 'Medium', 'High'])
    input_data['employment_status'] = st.sidebar.selectbox(
        "Employment Status",
        ['Employed', 'Unemployed', 'Retired', 'Student']
    )
    
    st.sidebar.markdown("---")
    
    # Lifestyle Factors
    st.sidebar.subheader("🏃 Lifestyle")
    input_data['smoking_status'] = st.sidebar.selectbox(
        "Smoking Status",
        ['Never', 'Former', 'Current']
    )
    input_data['alcohol_consumption_per_week'] = st.sidebar.slider(
        "Alcohol (drinks/week)", 0.0, 30.0, 5.0, 0.5
    )
    input_data['physical_activity_minutes_per_week'] = st.sidebar.slider(
        "Physical Activity (min/week)", 0, 600, 150, 10
    )
    input_data['diet_score'] = st.sidebar.slider(
        "Diet Quality Score (0-10)", 0, 10, 6,
        help="0=Poor, 10=Excellent"
    )
    input_data['sleep_hours_per_day'] = st.sidebar.slider(
        "Sleep (hours/day)", 3.0, 12.0, 7.0, 0.5
    )
    input_data['screen_time_hours_per_day'] = st.sidebar.slider(
        "Screen Time (hours/day)", 0.0, 12.0, 4.0, 0.5
    )
    
    st.sidebar.markdown("---")
    
    # Medical History
    st.sidebar.subheader("🏥 Medical History")
    input_data['family_history_diabetes'] = st.sidebar.selectbox(
        "Family History of Diabetes", [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )
    input_data['hypertension_history'] = st.sidebar.selectbox(
        "History of Hypertension", [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )
    input_data['cardiovascular_history'] = st.sidebar.selectbox(
        "History of Cardiovascular Disease", [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )
    
    st.sidebar.markdown("---")
    
    # Physical Measurements
    st.sidebar.subheader("📏 Physical Measurements")
    input_data['bmi'] = st.sidebar.number_input(
        "BMI (Body Mass Index)", 15.0, 45.0, 25.0, 0.1
    )
    input_data['waist_to_hip_ratio'] = st.sidebar.number_input(
        "Waist-to-Hip Ratio", 0.7, 1.2, 0.85, 0.01
    )
    input_data['systolic_bp'] = st.sidebar.number_input(
        "Systolic Blood Pressure (mmHg)", 90, 180, 120, 1
    )
    input_data['diastolic_bp'] = st.sidebar.number_input(
        "Diastolic Blood Pressure (mmHg)", 60, 120, 80, 1
    )
    input_data['heart_rate'] = st.sidebar.number_input(
        "Resting Heart Rate (bpm)", 50, 120, 70, 1
    )
    
    st.sidebar.markdown("---")
    
    # Laboratory Values
    st.sidebar.subheader("🔬 Laboratory Values")
    input_data['cholesterol_total'] = st.sidebar.number_input(
        "Total Cholesterol (mg/dL)", 120.0, 300.0, 180.0, 1.0
    )
    input_data['hdl_cholesterol'] = st.sidebar.number_input(
        "HDL Cholesterol (mg/dL)", 20.0, 100.0, 50.0, 1.0
    )
    input_data['ldl_cholesterol'] = st.sidebar.number_input(
        "LDL Cholesterol (mg/dL)", 50.0, 200.0, 100.0, 1.0
    )
    input_data['triglycerides'] = st.sidebar.number_input(
        "Triglycerides (mg/dL)", 50.0, 500.0, 150.0, 1.0
    )
    input_data['glucose_fasting'] = st.sidebar.number_input(
        "Fasting Glucose (mg/dL)", 70.0, 250.0, 95.0, 1.0
    )
    input_data['glucose_postprandial'] = st.sidebar.number_input(
        "Post-meal Glucose (mg/dL)", 90.0, 350.0, 120.0, 1.0
    )
    input_data['insulin_level'] = st.sidebar.number_input(
        "Insulin Level (μU/mL)", 2.0, 50.0, 10.0, 0.5
    )
    input_data['hba1c'] = st.sidebar.number_input(
        "HbA1c (%)", 4.0, 14.0, 5.5, 0.1
    )
    input_data['diabetes_stage'] = st.sidebar.selectbox(
        "Diabetes Stage",
        ['No Diabetes', 'Pre-Diabetes', 'Type 1', 'Type 2', 'Gestational']
    )
    input_data['diagnosed_diabetes'] = st.sidebar.selectbox(
        "Diagnosed Diabetes",
        [0, 1]
    )

    
    return input_data

def display_risk_result(risk_score):
    """Display risk score with visual indicators"""
    category, color, description = get_risk_category(risk_score)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown(f"""
        <div class="risk-box" style="background-color: {color}20; border: 2px solid {color};">
            <h2 style="color: {color}; margin: 0;">{category}</h2>
            <h1 style="color: {color}; margin: 10px 0; font-size: 3rem;">{risk_score:.1f}/100</h1>
            <p style="margin: 0; font-size: 1.1rem;">{description}</p>
        </div>
        """, unsafe_allow_html=True)

def main():
    """Main application function"""
    
    # Header
    st.markdown('<div class="main-header">🏥 Personal Diabetes Risk Dashboard</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">AI-Powered Diabetes Risk Assessment with Personalized Recommendations</div>',
        unsafe_allow_html=True
    )
    
    # Load model
    model = load_model()
    
    # Sidebar inputs
    input_data = sidebar_inputs()
    
    # Predict button
    st.sidebar.markdown("---")
    predict_button = st.sidebar.button("🔍 Calculate Risk Score", type="primary", use_container_width=True)
    
    if predict_button:
        # Validate inputs
        is_valid, errors = validate_input_ranges(input_data)
        
        if not is_valid:
            st.error("⚠️ Input Validation Errors:")
            for error in errors:
                st.write(f"- {error}")
            return
        
        # Create input dataframe
        input_df = create_input_dataframe(input_data)
        
        # Make prediction
        with st.spinner("Analyzing your health data..."):
            risk_score = model.predict(input_df)[0]
            risk_score = np.clip(risk_score, 0, 100)  # Ensure score is between 0-100
        
        # Display results
        st.success("✅ Analysis Complete!")
        
        # Risk Score Display
        st.markdown("## 📊 Your Diabetes Risk Assessment")
        display_risk_result(risk_score)
        
        # Gauge chart
        col1, col2 = st.columns([1, 1])
        
        with col1:
            gauge_fig = create_gauge_chart(risk_score)
            st.plotly_chart(gauge_fig, use_container_width=True)
        
        with col2:
            st.markdown("### 📈 Risk Score Interpretation")
            st.markdown("""
            - **0-30**: Low risk - Continue healthy habits
            - **30-60**: Moderate risk - Lifestyle changes recommended
            - **60-100**: High risk - Medical consultation advised
            """)
            
            st.markdown("### 🎯 Key Metrics")
            st.metric("Risk Score", f"{risk_score:.1f}/100")
            category, _, _ = get_risk_category(risk_score)
            st.metric("Risk Category", category)
        
        st.markdown("---")
        
        # Feature Importance
        st.markdown("## 🔍 Feature Importance Analysis")
        st.markdown("Understanding which factors contribute most to diabetes risk prediction:")
        
        importance_fig = create_feature_importance_chart(model, input_df)
        if importance_fig:
            st.plotly_chart(importance_fig, use_container_width=True)
        
        st.markdown("---")
        
        # SHAP Explanation
        st.markdown("## 🧠 AI Model Explanation (SHAP)")
        st.markdown("How each of your features influences your risk score:")
        
        shap_fig = create_shap_explanation(model, input_df)
        if shap_fig:
            st.pyplot(shap_fig)
        else:
            st.info("💡 SHAP explanations require the `shap` library. Install it with: `pip install shap`")
        
        st.markdown("---")
        
        # Personalized Recommendations
        st.markdown("## 💡 Personalized Health Recommendations")
        recommendations = generate_recommendations(input_data, risk_score)
        
        for i, rec in enumerate(recommendations, 1):
            st.markdown(
                f'''
                <div style="
                    background-color:#06402B;
                    padding:15px;
                    border-left:4px solid #007bff;
                    border-radius:6px;
                    margin:10px 0;
                ">
                    <strong>Recommendation {i}:</strong><br>{rec}
                </div>
                ''',
                unsafe_allow_html=True
            )

        
        st.markdown("---")
        
        # Key Health Indicators Summary
        st.markdown("## 📋 Your Health Indicators Summary")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### Metabolic Health")
            st.write(f"**BMI**: {get_feature_interpretation('bmi', input_data['bmi'])}")
            st.write(f"**Fasting Glucose**: {get_feature_interpretation('glucose_fasting', input_data['glucose_fasting'])}")
            st.write(f"**HbA1c**: {get_feature_interpretation('hba1c', input_data['hba1c'])}")
            st.write(f"**HbA1c**: {get_feature_interpretation('hba1c', input_data['hba1c'])}")
        
        with col2:
            st.markdown("### Lifestyle Factors")
            st.write(f"**Physical Activity**: {get_feature_interpretation('physical_activity_minutes_per_week', input_data['physical_activity_minutes_per_week'])}")
            st.write(f"**Sleep**: {get_feature_interpretation('sleep_hours_per_day', input_data['sleep_hours_per_day'])}")
            st.write(f"**Diet Score**: {get_feature_interpretation('diet_score', input_data['diet_score'])}")
        
        with col3:
            st.markdown("### Cardiovascular Health")
            st.write(f"**Systolic BP**: {get_feature_interpretation('systolic_bp', input_data['systolic_bp'])}")
            st.write(f"**Diastolic BP**: {get_feature_interpretation('diastolic_bp', input_data['diastolic_bp'])}")
            st.write(f"**Heart Rate**: {input_data['heart_rate']} bpm")
        
        st.markdown("---")
        
        # Disclaimer
        st.warning("""
        ⚠️ **Medical Disclaimer**: This tool provides risk assessment for educational purposes only. 
        It is NOT a substitute for professional medical advice, diagnosis, or treatment. 
        Always consult with qualified healthcare providers regarding your health conditions and concerns.
        """)
    
    else:
        # Welcome message
        st.info("""
        👈 **Get Started**: Fill in your health information in the sidebar and click 
        "Calculate Risk Score" to receive your personalized diabetes risk assessment.
        
        This dashboard uses machine learning to predict your diabetes risk based on:
        - 📊 Demographic information
        - 🏃 Lifestyle factors
        - 🏥 Medical history
        - 📏 Physical measurements
        - 🔬 Laboratory values
        """)
        
        # Display sample information
        st.markdown("## 📖 About This Dashboard")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### What You'll Get:
            - ✅ Personalized risk score (0-100)
            - ✅ Visual risk assessment
            - ✅ Feature importance analysis
            - ✅ AI-powered explanations
            - ✅ Actionable health recommendations
            """)
        
        with col2:
            st.markdown("""
            ### Key Features:
            - 🤖 Machine Learning prediction
            - 📊 Interactive visualizations
            - 🔍 SHAP explainability
            - 💡 Personalized advice
            - 📱 User-friendly interface
            """)
        
        st.markdown("---")
        st.markdown("""
        ### 🔐 Privacy Note
        All calculations are performed locally. Your health data is not stored or transmitted anywhere.
        """)

if __name__ == "__main__":
    main()