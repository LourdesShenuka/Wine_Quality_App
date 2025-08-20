# Wine Quality Prediction App

An end-to-end **Machine Learning Project** that predicts wine quality (Good / Bad) based on its chemical properties.  
This project demonstrates the full ML workflow: **data preprocessing, model training, evaluation, and deployment** using **Streamlit**.

---

## Features
- **Exploratory Data Analysis (EDA)**
  - Dataset overview
  - Histograms, scatter plots, and correlation heatmap  
- **Machine Learning**
  - Logistic Regression & Random Forest
  - Cross-validation for model selection
  - Performance metrics (Accuracy, F1-score, Confusion Matrix)
- **Streamlit Web App**
  - Interactive data exploration
  - Model performance dashboard
  - Real-time predictions from user input

---

## Project Structure

wine-quality-app/
│
├── data/
│ └── WineQT.csv # Dataset (from Kaggle)
│
├── model_training.ipynb # Jupyter Notebook for training & EDA
├── app.py # Streamlit app
├── model.pkl # Saved trained model
├── features.json # Feature names used by the model
├── metrics.json # Training metrics
├── requirements.txt # Dependencies
└── README.md # Project documentation


---

## Installation & Setup

### Clone the Repository
```bash
git clone https://github.com/your-username/wine-quality-app.git
cd wine-quality-app

## Create Virtual Environment
python -m venv .venv


## Activate it:

# Windows (PowerShell)

.venv\Scripts\Activate.ps1

# Mac/Linux

source .venv/bin/activate

## Install Dependencies
pip install -r requirements.txt

## Run the App Locally
streamlit run app.py

* The app will open in your browser at http://localhost:8501

## Deploy on Streamlit Cloud
1. Push this repo to GitHub.
2. Go to Streamlit Cloud.
3. Connect your GitHub repo.
4. Set app.py as the entry point.
5. The app will be live online

## Example Prediction

Input:

alcohol = 10.2
pH = 3.3
sulphates = 0.6
volatile acidity = 0.5
...


Output:

Prediction: Good Wine 🍷✅
Confidence: 78.5%

Dataset

Source: Kaggle - Wine Quality Dataset

Description: Contains physicochemical tests of red/white wine and quality scores (0–10).


* Author: Lourdes shenuka

* Horizon Campus — Intelligent Systems Assignment