# Loan Approval Prediction — Machine Learning Project

I built this project because I wanted to go beyond just training a model in a notebook and closing the tab. I wanted to build something complete — something a real person could actually open and use.


## What This Project Does

It predicts whether a loan application is likely to get approved or rejected. You enter your income, loan amount, credit history, and a few personal details — and the model gives you an instant decision with a confidence score.

The app also shows your estimated monthly EMI and debt-to-income ratio, so it feels like a real financial tool, not just a machine learning demo.
---

## The Dataset

Source: Loan Prediction Dataset (commonly used for classification practice)
Target: Loan Status — Approved or Rejected
Features used: 18, after preprocessing and one-hot encoding

## What I Built

### Data Preprocessing
- Handled missing values carefully instead of just dropping rows
- Applied one-hot encoding to categorical columns like Gender, Marital Status, Education, Employment Type, and Property Area
- Scaled all features using StandardScaler so the model trains fairly across different value ranges

### Features the Model Uses
Applicant Income, Co-Applicant Income, Loan Amount, Loan Term, Credit History, Total Income, Number of Dependents, Gender, Marital Status, Education, Self Employment Status, and Property Area.

### Model
**Logistic Regression** — a clean, interpretable classification model. I chose it because it works well for binary decisions like approve or reject, and the results are easy to explain to someone who is not a data scientist.

## Tech Stack

Python, Pandas, NumPy, Scikit-learn, Streamlit, Joblib

---

## Project Structure

loan-approval-prediction/
app.py                  # Streamlit web application
Logistic_reg.pkl        # Trained Logistic Regression model
scaler.pkl              # Fitted StandardScaler
requirements.txt        # Python dependencies
README.md

## Live Demo

Deployed on Streamlit Cloud — [click here to try it live]([https://your-app-link.streamlit.app](https://loanapprovalpredictor-by-sourabh.streamlit.app/))

---

## What I Learned

I started this thinking the hardest part would be the model. It was not. The hardest part was making the app feel like something real — handling user inputs cleanly, converting values so the model reads them correctly, showing results in a way that actually means something to the person looking at the screen.

I also learned that credit history is by far the most powerful feature in this dataset. The model leans heavily on it, which honestly makes sense in real life too.


## About Me

I am a beginner in data science who believes that finishing and deploying a project teaches you more than ten incomplete notebooks. This project is proof of that belief.

---

## Contact

- GitHub: [username](https://github.com/sourabh9098))
- LinkedIn: [linkedin](www.linkedin.com/in/sourabh9098)
- Email: www.sourabh555@gmail.com
