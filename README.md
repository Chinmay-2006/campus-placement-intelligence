# Campus Placement & CTC Intelligence Hub

A two-stage machine learning system for campus placement prediction and expected starting CTC estimation.

## 1. Project Overview

The system consists of two machine learning stages.

Stage 1 — Classification

Predicts:

- Placement status
- Placement probability

Stage 2 — Regression

Predicts:

- Expected starting CTC in LPA

The regression model is trained only on placed-student records.

## 2. Technology Stack

Python

Pandas

NumPy

Scikit-learn

Streamlit

Matplotlib

Seaborn

Jupyter

Joblib

Git/GitHub

## 3. Dataset

The dataset contains 100,000 student records and 26 columns.

The model uses 23 input features.

Student ID is excluded.

Salary package is excluded from classification.

Placement status is excluded from regression.

## 4. Machine Learning Architecture

Student Profile
↓
Preprocessing Pipeline
↓
Stage 1 — Classification
↓
Placement Prediction
↓
If Placed
↓
Stage 2 — CTC Regression
↓
Expected Starting CTC

## 5. Features

The system uses academic, technical, communication, activity and profile features including:

- Age
- Gender
- CGPA
- Branch
- College Tier
- Internships
- Projects
- Certifications
- Coding Skill Score
- Aptitude Score
- Communication Skill Score
- Logical Reasoning Score
- Hackathons
- GitHub Repositories
- LinkedIn Connections
- Mock Interview Score
- Attendance
- Backlogs
- Extracurricular Score
- Leadership Score
- Volunteer Experience
- Sleep Hours
- Study Hours Per Day

## 6. Preprocessing

Numerical features are standardized using StandardScaler.

Categorical features are encoded using OneHotEncoder.

Preprocessing is included inside the Scikit-learn pipeline.

## 7. Classification

The classification stage compares:

- Logistic Regression
- Random Forest

Metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

## 8. Regression

The regression stage compares:

- Linear Regression
- Random Forest Regression

Metrics:

- MAE
- RMSE
- R²

## 9. Explainability

For linear classification, model coefficients are used to show feature effects.

For tree-based classification, feature importance is used where supported.

These values describe model behavior and should not be interpreted as causal relationships.

## 10. Streamlit Application

The application provides:

- Student prediction
- Placement probability
- Expected CTC
- What-if analysis
- Model comparison
- Cross-validation results
- Classification report
- Confusion matrix
- Regression evaluation
- Feature effects
- Dataset analytics
- Prediction history

## 11. Project Structure

campus-intelligence/

├── data/

├── models/

├── notebooks/

├── reports/

├── src/

├── app.py

├── requirements.txt

├── README.md

└── .gitignore

## 12. Running the Project

Install dependencies:

pip install -r requirements.txt

Run the application:

streamlit run app.py

## 13. Limitations

The dataset is synthetic.

Model performance should therefore not be interpreted as real-world placement prediction accuracy.

Predictions are estimates rather than guaranteed placement or salary outcomes.

## 14. Team

- Chinmay Patil
- Sanika Mhatre
- Siddharth Parchande
- Dhruva
