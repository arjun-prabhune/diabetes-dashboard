# 🏥 Personal Diabetes Risk Dashboard

An **AI-powered health analytics web application** that predicts an individual's **diabetes risk score (0–100)** based on their lifestyle, medical history, and lab values.  
The system uses a **machine learning regression model (Random Forest)** trained on health data and provides **personalized recommendations** with **explainable AI visualizations**.

---

## 🌟 Features

### 🤖 **Machine Learning Prediction**
- Predicts a patient’s **diabetes risk score (0–100)** using demographic, lifestyle, and medical factors.
- Built with **Scikit-learn** and trained using a **RandomForestRegressor**.

### 📊 **Interactive Streamlit Dashboard**
- User-friendly interface for entering health data.
- Displays **risk results**, **category indicators**, and **visual analytics** in real-time.

### 💡 **Personalized Health Recommendations**
- Provides actionable, tailored suggestions based on the predicted risk and individual inputs.
- Encourages lifestyle improvement across activity, diet, sleep, and glucose management.

### 🔍 **Model Explainability**
- Integrated **SHAP explainability** to visualize how each factor influences the model’s prediction.
- Displays **feature importance charts** and **SHAP waterfall plots** for transparency.

### 🎨 **Beautiful Data Visualizations**
- Dynamic **gauge chart** for risk scoring (Plotly).
- Clean, responsive layout with custom CSS styling.

---

## 🧠 Project Architecture

diabetes-dashboard/
│
├── data/
│ └── diabetes_data.csv # Dataset (not included for privacy)
│
├── models/
│ └── diabetes_risk_regressor.pkl # Trained ML model
│
├── utils/
│ └── preprocessing.py # Data validation and input utilities
│
├── app.py # Streamlit web application
├── train_model.py # Model training and evaluation script
└── README.md # Project documentation

yaml
Copy code

---

## ⚙️ Tech Stack

| Category | Tools / Libraries |
|-----------|------------------|
| **Frontend / UI** | [Streamlit](https://streamlit.io/), Plotly, CSS |
| **Machine Learning** | Scikit-learn, RandomForestRegressor |
| **Data Processing** | pandas, numpy |
| **Explainability** | SHAP |
| **Visualization** | matplotlib, seaborn, plotly.express |
| **Model Persistence** | joblib |

---

## 🚀 How to Run Locally

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/<your-username>/diabetes-dashboard.git
cd diabetes-dashboard
2️⃣ Create a Virtual Environment
bash
Copy code
python3 -m venv venv
source venv/bin/activate   # On Mac/Linux
venv\Scripts\activate      # On Windows
3️⃣ Install Dependencies
bash
Copy code
pip install -r requirements.txt
4️⃣ Train the Model
bash
Copy code
python train_model.py
This will train a RandomForest model and save it as models/diabetes_risk_regressor.pkl.

5️⃣ Run the Streamlit App
bash
Copy code
streamlit run app.py
Then open http://localhost:8501 in your browser.

📈 Sample Output
Example Visualization	Description
Interactive risk gauge showing predicted diabetes risk
Explains how each feature influences the prediction
Highlights most impactful lifestyle/biometric factors

💬 Example Use Case
A 45-year-old individual inputs lifestyle and biometric data into the dashboard.
The ML model predicts a Diabetes Risk Score of 72/100, categorizing them as High Risk.
The app then displays:

🚨 Red gauge indicator for risk level

📈 SHAP plot showing high BMI and elevated fasting glucose as key contributors

💡 Personalized recommendations for diet and exercise

🧩 Model Evaluation
Metric	Training	Testing
R²	0.998	0.992
MAE	0.27	0.63
RMSE	0.37	0.83

⚠️ Note: Extremely high R² values may indicate strong correlations or limited data noise (synthetic dataset).
In real-world clinical data, performance will be lower due to biological variability.

🩺 Medical Disclaimer
⚠️ This tool is intended for educational and research purposes only.
It is not a substitute for professional medical advice, diagnosis, or treatment.
Always consult a qualified healthcare professional for medical concerns.

✨ Future Improvements
🔁 Integrate real-world EMR (Electronic Medical Record) data.

🌐 Deploy as a public health web app on Streamlit Cloud or Azure.

🧬 Add early warning detection using temporal data (e.g., HbA1c over time).

🧠 Experiment with neural network or gradient boosting models.

⚖️ Implement bias and fairness checks across demographics.

👨‍💻 Author
Arjun Prabhune
📧 arjun.prabhune@gmail.com
🔗 arjun-prabhune
💻 arjun-prabhune

⭐ Acknowledgements
Streamlit for intuitive dashboarding

Scikit-learn for powerful machine learning tools

SHAP for model explainability

Plotly for dynamic visualization

🧾 License
This project is licensed under the MIT License.

Empowering individuals with data-driven insights for better health.
