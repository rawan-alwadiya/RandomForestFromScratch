# **RandomForestFromScratch: Random Forest from Scratch on Seeds Dataset**

RandomForestFromScratch is a machine learning project that implements the **Random Forest algorithm entirely from scratch**, without using scikit-learn’s built-in classifiers.  
It demonstrates how machine learning fundamentals can be implemented from first principles, including **exploratory data analysis**, **feature scaling**, **algorithm implementation**, **evaluation with custom-built metrics**, and **deployment with Streamlit**.

---

## **Demo**

- 🎥 [View LinkedIn Demo Post](https://www.linkedin.com/posts/rawan-alwadeya-17948a305_machinelearning-fromscratch-randomforest-activity-7364025324336005120-zTf2?utm_source=share&utm_medium=member_desktop&rcm=ACoAAE3YzG0BAZw48kimDDr_guvq8zXgSjDgk_I)  
- 🌐 [Try the App Live on Streamlit](https://randomforestfromscratch-xmpa5jh2wpyu8hngsuq8sr.streamlit.app/)

![App Demo](https://github.com/rawan-alwadiya/RandomForestFromScratch/blob/main/RandomForestFromScratch.png)

---

## **Project Overview**

The workflow includes:  
- **Exploratory Data Analysis (EDA)**  
- **Feature scaling (standardization)**  
- Implementation of a **Random Forest classifier from scratch**  
- Evaluation using **custom-built metrics**  
- Deployment of the trained model via a **Streamlit web application**

---

## **Objective**

Develop and deploy a machine learning model built from first principles to classify three varieties of wheat seeds, demonstrating the **end-to-end pipeline** of data exploration, model construction, evaluation, and deployment.

---

## **Dataset**

- **Source**: [Seeds Dataset (UCI Repository)](https://archive.ics.uci.edu/ml/machine-learning-databases/00236/seeds_dataset.txt)  
- **Samples**: 210  
- **Features**: 7 numerical features (Area, Perimeter, Compactness, Kernel Length, Kernel Width, Asymmetry, Groove Length)  
- **Target**: 3 wheat varieties (Kama, Rosa, Canadian)  

---

## **Project Workflow**

- **EDA & Visualization**: Feature distributions, class separability  
- **Preprocessing**: Feature scaling (standardization)  
- **Modeling**: Implementation of Random Forest from scratch  
  - Decision trees (Gini index, information gain)  
  - Bootstrap sampling  
  - Random feature selection at splits  
  - Majority voting for final predictions  
- **Evaluation Metrics (from scratch)**:  
  - Accuracy  
  - Precision  
  - Recall  
  - F1-score  
  - Confusion Matrix  
  - Classification Report  
- **Deployment**: Streamlit web app with interactive sliders for seed features  

---

## **Performance Results**

**Random Forest Classifier (Scratch Implementation):**  
- **Accuracy**: 0.90  
- **Precision**: 0.91  
- **Recall**: 0.91  
- **F1-score**: 0.91  

---

## **Project Links**
 
- **Kaggle Notebook**: [View on Kaggle](https://www.kaggle.com/code/rawanalwadeya/randomforestfromscratch-built-from-scratch?scriptVersionId=257175423)  
- **Live Streamlit App**: [Try it Now](https://randomforestfromscratch-xmpa5jh2wpyu8hngsuq8sr.streamlit.app/)  

---

## **Tech Stack**

**Languages & Libraries**:  
- Python, Pandas, NumPy  
- Matplotlib, Seaborn  
- Streamlit (Deployment)  

**Techniques**:  
- Random Forest from scratch (Decision Trees, Bootstrap, Random Features, Voting)  
- Custom Evaluation Metrics  
- Streamlit Deployment  
