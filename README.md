# 📘 Machine Learning Assignment 02  
## Student Academic Performance Classification  

**Name:** DASAWANT PARTH NARESH MANASI  
**Student ID:** 2025AB05066  

---

## 📌 Project Overview

This project focuses on building and evaluating multiple supervised machine learning classification models on a structured academic dataset.

The objective is to compare different classification algorithms using standard evaluation metrics and identify the most effective model for prediction.

The problem is strictly formulated as a **classification task**.

---

## 📂 Dataset Information

- Total Records: 5000  
- Structured academic features  
- Target Variable: Classification-based output  

The dataset consists of structured numerical and categorical attributes representing academic performance indicators.

---

## 🧠 Machine Learning Models Implemented

The following six classification algorithms were trained and evaluated:

1. Logistic Regression  
2. Decision Tree Classifier  
3. k-Nearest Neighbors (kNN)  
4. Naive Bayes (Gaussian)  
5. Random Forest (Ensemble Model)  
6. XGBoost (Gradient Boosting Ensemble)  

---

## 📊 Evaluation Metrics

Each model was evaluated using:

- Accuracy  
- Precision  
- Recall  
- F1 Score  
- Matthews Correlation Coefficient (MCC)  
- AUC (Area Under ROC Curve)  

---

## 📈 Model Performance Comparison

| Model                  | Accuracy  | Precision | Recall    | F1 Score  | MCC       | AUC       |
|------------------------|-----------|-----------|-----------|-----------|-----------|-----------|
| Logistic Regression    | 0.921410  | 0.922178  | 0.921410  | 0.921590  | 0.905045  | 0.993408  |
| Decision Tree          | 0.907088  | 0.907396  | 0.907088  | 0.907167  | 0.887625  | 0.962472  |
| kNN                    | 0.916636  | 0.917409  | 0.916636  | 0.916805  | 0.899204  | 0.981247  |
| Naive Bayes            | 0.897907  | 0.900702  | 0.897907  | 0.898075  | 0.877263  | 0.990201  |
| Random Forest          | 0.920676  | 0.920917  | 0.920676  | 0.920706  | 0.904045  | 0.992435  |
| XGBoost                | **0.923981** | **0.924213** | **0.923981** | **0.923982** | **0.908031** | **0.993976** |

---

## 🔎 Performance Analysis

### 🥇 Best Performing Model: XGBoost

- Highest Accuracy (92.39%)
- Highest MCC (0.908)
- Highest AUC (0.9939)
- Strong generalization performance
- Effectively captures complex feature interactions

---

## 📌 Key Observations

- XGBoost achieved the best overall performance.
- Logistic Regression provided a strong baseline model.
- Random Forest delivered stable and reliable results.
- Decision Tree showed slightly lower generalization compared to ensemble methods.
- Naive Bayes performed well but is limited by its feature independence assumption.
- kNN performed competitively but may scale poorly for larger datasets.

---

## 🛠 Technologies Used

- Python  
- NumPy  
- Pandas  
- Scikit-learn  
- XGBoost  
- Matplotlib  
- Seaborn  

---

## ⚙️ Project Workflow

1. Data Cleaning and Preprocessing  
2. Feature Encoding and Scaling  
3. Train-Test Split  
4. Model Training  
5. Model Evaluation  
6. Comparative Performance Analysis  

---

## 📌 Conclusion

The experimental results demonstrate that ensemble-based boosting techniques, particularly **XGBoost**, provide superior classification performance on structured academic datasets.

Although traditional models such as Logistic Regression remain strong baselines, advanced ensemble methods offer improved generalization and robustness.

This assignment emphasizes:

- Comparing multiple models  
- Evaluating with diverse performance metrics  
- Using MCC and AUC for balanced performance assessment  

---

## 🚀 Future Improvements

- Hyperparameter tuning using GridSearchCV  
- K-Fold Cross Validation  
- Feature Importance Analysis  
- Model Explainability using SHAP  
- Deployment using Flask or FastAPI  

---

## 👨‍🎓 Author

DASAWANT PARTH NARESH MANASI
 
Machine Learning Assignment – 02  
