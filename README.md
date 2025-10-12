# 🫀 Predicting Heart Disease Type from Clinical Measurements

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Jupyter](https://img.shields.io/badge/Notebook-Jupyter-orange.svg)](https://jupyter.org/)

---

## 📘 Project Overview
This project predicts the **presence and severity of heart disease** using clinical measurements.  
We leverage the **UCI Heart Disease Dataset**, applying **feature engineering**, **logistic regression**, and **SVM** to classify patients into heart disease risk categories.

The goal is to demonstrate a complete ML workflow — data preprocessing, feature engineering, model training, and evaluation.

---

## 📂 Dataset Information

**Source:** UCI Heart Disease Dataset  
**Target Variable:** `num`

| Value | Meaning |
|--------|----------|
| 0 | No heart disease |
| 1–4 | Presence of heart disease (increasing severity) |

To improve model performance, we convert this to a **binary classification** problem:
- `0` → No disease  
- `1` → Disease present (classes 1–4 combined)

<details>
<summary>🧾 Key Features</summary>

| Feature | Description |
|----------|--------------|
| `age` | Age of the patient |
| `sex` | Gender (1 = male, 0 = female) |
| `cp` | Chest pain type (4 categories) |
| `trestbps` | Resting blood pressure |
| `chol` | Serum cholesterol (mg/dl) |
| `fbs` | Fasting blood sugar > 120 mg/dl |
| `restecg` | Resting electrocardiographic results |
| `thalch` | Maximum heart rate achieved |
| `exang` | Exercise-induced angina |
| `oldpeak` | ST depression induced by exercise |
| `slope`, `ca`, `thal` | Additional diagnostic features |
</details>

---

## ⚙️ Methodology

### 1️⃣ Data Cleaning & Missing Value Handling
- Inspected missing values per column.
- Used **proximity-based imputation** for categorical features:
  ```python
  df.groupby(group_cols)[target_col].transform(
      lambda x: x.fillna(x.mode().iloc[0]) if not x.mode().empty else x
  )
Applied median imputation for numerical columns.

Dropped redundant or high-missing columns:

id (identifier)

dataset (non-informative)

ca, thal (high missingness)

Converted num (0–4) into a binary classification target:

df['num'] = df['num'].apply(lambda x: 1 if x > 0 else 0)

2️⃣ Feature Engineering

One-hot encoded categorical columns:
sex, cp, fbs, restecg, exang

Scaled numeric columns using StandardScaler for model compatibility.

Ensured features were clean, numeric, and model-ready.

Example:

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[numeric_cols])

3️⃣ Model Training

Two machine learning models were implemented and compared:

Logistic Regression (linear baseline)

Support Vector Machine (SVM) (nonlinear kernel-based model)

Dataset split:

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

4️⃣ Model Evaluation

We evaluated both models using the following metrics:

Accuracy Score

Confusion Matrix

Classification Report (Precision, Recall, F1-score)

📊 Results
Model	Accuracy	Comment
Logistic Regression	~0.75	Simple, interpretable baseline
SVM (RBF Kernel)	~0.77	Slightly better, captures nonlinear boundaries

⚠️ Note: Accuracy may vary slightly depending on random seed and preprocessing details.

📈 Insights

Feature scaling and encoding significantly improve performance.

The dataset contains overlapping features, limiting separability.

Binary target conversion increases model consistency and interpretability.

SVM performs marginally better due to its nonlinear kernel behavior.

🧠 Technologies Used
Category	Tools
Language	Python
Libraries	NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn
Environment	Jupyter Notebook (.ipynb)
🪜 Project Structure
├── featureEngineering.ipynb    # Handles missing values, encoding, scaling
├── disease_pred.ipynb           # Trains Logistic Regression and SVM models
├── heart_disease_uci.csv        # Original dataset
├── README.md                    # Project documentation

🚀 How to Run

Clone this repository:

git clone https://github.com/<your-username>/heart-disease-prediction.git
cd heart-disease-prediction


Install dependencies:

pip install -r requirements.txt


Open Jupyter Notebook and run the files in order:

featureEngineering.ipynb

disease_pred.ipynb

Review printed metrics, confusion matrix, and accuracy results.

🧭 Future Improvements

🔧 Hyperparameter tuning with GridSearchCV for better SVM and Logistic Regression performance.

📉 Feature selection using Mutual Information or Recursive Feature Elimination (RFE).

🌲 Experiment with ensemble methods like Random Forest and XGBoost.

🔁 Use K-Fold Cross-Validation for more robust accuracy estimation.

✍️ Author

PESU_Student
📍 Department of Computer Science
🎓 PES University