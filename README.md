# 📘 Machine Learning Assignment 02  
## Dry Bean Classification using Machine Learning  

**Name:** DASAWANT PARTH NARESH MANASI  
**Student ID:** 2025AB05066  

---

## 📌 Project Overview

This project focuses on building and evaluating multiple supervised machine learning classification models on the Dry Bean Dataset, a structured agricultural dataset containing morphological measurements of bean grains.

The objective is to compare different classification algorithms using standard evaluation metrics and identify the most effective model for multiclass bean variety prediction.

The problem is formulated as a multiclass classification task.

---

## 📂 Dataset Information

- Dataset Name: Dry Bean Dataset  
- Domain: Agriculture / Biology  
- Total Records: 13,611  
- Total Features: 16 input features  
- Target Classes: 7 bean varieties  
- Feature Type: Integer and Continuous  
- Missing Values: None  

The dataset contains geometric and shape-based features extracted from bean grain images using computer vision techniques.

---

## 🫘 Target Classes

The classification target consists of seven dry bean varieties:

- Seker  
- Barbunya  
- Bombay  
- Cali  
- Dermosan  
- Horoz  
- Sira  

---

## 📐 Feature Description

The features represent morphological and geometric characteristics of bean grains.

| Feature | Description |
|--------|------------|
| Area | Pixel area of the bean region |
| Perimeter | Length of bean boundary |
| MajorAxisLength | Longest axis of the bean |
| MinorAxisLength | Shortest axis perpendicular to major axis |
| AspectRatio | Ratio of major to minor axis |
| Eccentricity | Elliptical eccentricity |
| ConvexArea | Pixel count of convex hull |
| EquivDiameter | Diameter of circle with same area |
| Extent | Ratio of bean area to bounding box |
| Solidity | Ratio of area to convex area |
| Roundness | (4π × Area) / Perimeter² |
| Compactness | Equivalent diameter / Major axis |
| ShapeFactor1 | Shape descriptor 1 |
| ShapeFactor2 | Shape descriptor 2 |
| ShapeFactor3 | Shape descriptor 3 |
| ShapeFactor4 | Shape descriptor 4 |

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

| Model | Accuracy | Precision | Recall | F1 Score | MCC | AUC |
|------|----------|----------|--------|---------|--------|--------|
| Logistic Regression | 0.921410 | 0.922178 | 0.921410 | 0.921590 | 0.905045 | 0.993408 |
| Decision Tree | 0.907088 | 0.907396 | 0.907088 | 0.907167 | 0.887625 | 0.962472 |
| kNN | 0.916636 | 0.917409 | 0.916636 | 0.916805 | 0.899204 | 0.981247 |
| Naive Bayes | 0.897907 | 0.900702 | 0.897907 | 0.898075 | 0.877263 | 0.990201 |
| Random Forest | 0.920676 | 0.920917 | 0.920676 | 0.920706 | 0.904045 | 0.992435 |
| **XGBoost** | **0.923981** | **0.924213** | **0.923981** | **0.923982** | **0.908031** | **0.993976** |

---

## 🔎 Performance Analysis

### 🥇 Best Performing Model: XGBoost

- Highest Accuracy (92.39%)  
- Highest MCC (0.908)  
- Highest AUC (0.9939)  
- Strong generalization performance  
- Captures complex feature interactions  

---

## 📌 Key Observations

- XGBoost achieved the best overall performance.  
- Logistic Regression provided a strong baseline.  
- Random Forest delivered stable results.  
- Decision Tree showed lower generalization than ensembles.  
- Naive Bayes is limited by feature independence assumption.  
- kNN performed well but may scale poorly for large datasets.  

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
2. Feature Scaling and Preparation  
3. Train-Test Split  
4. Model Training  
5. Model Evaluation  
6. Comparative Analysis  

---

## 📌 Conclusion

The experimental results demonstrate that ensemble boosting methods, particularly XGBoost, provide superior multiclass classification performance on the Dry Bean morphological dataset.

While traditional models such as Logistic Regression remain strong baselines, advanced ensemble methods offer improved generalization and robustness for structured agricultural data.

---

## 🚀 Future Improvements

- Hyperparameter tuning (GridSearchCV)  
- K-Fold Cross Validation  
- Feature Importance Analysis  
- Model Explainability (SHAP)  
- Deployment (Flask / FastAPI)  

---

## 📄 Dataset Reference

M. Koklu, I. A. Özkan (2020)  
Multiclass Classification of Dry Beans Using Computer Vision and Machine Learning Techniques  
Computers and Electronics in Agriculture  

---

## 👨‍🎓 Author

DASAWANT PARTH NARESH MANASI  
Machine Learning Assignment – 02  